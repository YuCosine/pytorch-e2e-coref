import os
import sys
from collections import deque
import time
import re
import shutil
import subprocess
import numpy as np
import random
from itertools import chain
import torch
from torch import nn, optim
import torch.utils.data as tud
import argparse

from model import Model
from util import initialize_from_env, set_log_file, TensorboardWriter
from data_utils import PrpDataset
from model_utils import OptimizerBase
import metrics


parser = argparse.ArgumentParser(description='train or test model')
parser.add_argument('model', type=str,
                    help='model name to train or test')
parser.add_argument('mode', type=str,
                    help='train or eval')
parser.add_argument('--log_dir', type=str, default='logs', 
                    help='dir of training log')


class Runner:
    def __init__(self, config):
        self.config = config
        self.model = Model(config).cuda()

        self.optimizer = OptimizerBase.from_opt(self.model, self.config)

        self.epoch_idx = 0
        self.max_f1 = 0.
        self.max_f1_epoch_idx = 0

        if self.config["max_ckpt_to_keep"] > 0:
            self.ckpt_queue = deque([], maxlen=config["max_ckpt_to_keep"])

        self.writer = TensorboardWriter(config["log_dir"])

    @staticmethod
    def compute_ant_loss(
        # self,
        # [cand_num]
        cand_mention_scores,
        # [top_cand_num]
        top_start_idxes,
        # [top_cand_num]
        top_end_idxes,
        # [top_cand_num]
        top_span_cluster_ids,
        # [top_span_num, pruned_ant_num]
        top_ant_idxes_of_spans,
        # [top_cand_num, pruned_ant_num]
        top_ant_cluster_ids_of_spans,
        # # [top_cand_num, 1 + pruned_ant_num]
        top_ant_scores_of_spans,
        # 4 * [top_cand_num, 1 + pruned_ant_num]
        # list_of_top_ant_scores_of_spans,
        # [top_span_num, pruned_ant_num]
        top_ant_mask_of_spans,
        # [top_span_num, 1 + top_span_num], [top_span_num, top_span_num]
        full_fast_ant_scores_of_spans, full_ant_mask_of_spans
    ):

        top_span_num = top_span_cluster_ids.size(0)

        # [top_cand_num, 1]
        non_dummy_indicator = (top_span_cluster_ids > 0).view(-1, 1)

        top_ant_cluster_ids_of_spans.masked_fill_(~top_ant_mask_of_spans, -float('inf'))
        # [top_cand_num, pruned_ant_num]
        same_cluster_indicator = top_ant_cluster_ids_of_spans == top_span_cluster_ids.view(-1, 1)

        # [top_cand_num, pruned_ant_num]
        pairwise_labels = same_cluster_indicator & non_dummy_indicator

        # [top_cand_num, 1 + pruned_ant_num]
        top_antecedent_labels = torch.cat(
            (
                # [top_cand_num, 1]
                ~(pairwise_labels.any(dim=1, keepdim=True)),
                # [top_cand_num, pruned_ant_num]
                pairwise_labels
            ), dim=1
        )

        loss = -(
                torch.logsumexp(
                    top_ant_scores_of_spans.masked_fill_(~top_ant_cluster_ids_of_spans, -float('inf')),
                    dim=1
                ) - torch.logsumexp(top_ant_scores_of_spans, dim=1)
            ).sum()

        return loss

    @staticmethod
    def predict(
        # [cand_num]
        cand_mention_scores,
        # [top_cand_num]
        top_start_idxes,
        # [top_cand_num]
        top_end_idxes,
        # [top_cand_num]
        top_span_cluster_ids,
        # [top_span_num, pruned_ant_num]
        top_ant_idxes_of_spans,
        # [top_cand_num, pruned_ant_num]
        top_ant_cluster_ids_of_spans,
        # # [top_cand_num, 1 + pruned_ant_num]
        top_ant_scores_of_spans,
        # 4 * [top_cand_num, 1 + pruned_ant_num]
        # list_of_top_ant_scores_of_spans,
        # [top_span_num, pruned_ant_num]
        top_ant_mask_of_spans,
        # # [doc_len, pos_tag_num]
        # pos_tag_logits
    ):
        # (
        #
        # ) = self.model(*input_tensors)

        predicted_ant_idxes = []

        for span_idx, loc in enumerate(torch.argmax(top_ant_scores_of_spans, dim=1) - 1):
            if loc < 0:
                predicted_ant_idxes.append(-1)
            else:
                predicted_ant_idxes.append(top_ant_idxes_of_spans[span_idx, loc].item())

        span_to_predicted_cluster_id = {}
        predicted_clusters = []

        for span_idx, ant_idx in enumerate(predicted_ant_idxes):
            if ant_idx < 0:
                continue

            assert span_idx > ant_idx

            ant_span = top_start_idxes[ant_idx].item(), top_end_idxes[ant_idx].item()

            if ant_span in span_to_predicted_cluster_id:
                predicted_cluster_id = span_to_predicted_cluster_id[ant_span]
            else:
                predicted_cluster_id = len(predicted_clusters)
                predicted_clusters.append([ant_span])
                span_to_predicted_cluster_id[ant_span] = predicted_cluster_id

            span = top_start_idxes[span_idx].item(), top_end_idxes[span_idx].item()
            predicted_clusters[predicted_cluster_id].append(span)
            span_to_predicted_cluster_id[span] = predicted_cluster_id

        predicted_clusters = [tuple(cluster) for cluster in predicted_clusters]
        span_to_predicted_cluster = {
            span: predicted_clusters[cluster_id]
            for span, cluster_id in span_to_predicted_cluster_id.items()
        }

        # [top_cand_num], [top_cand_num], [top_cand_num]
        return top_start_idxes, top_end_idxes, predicted_ant_idxes, \
               predicted_clusters, span_to_predicted_cluster


    def test_gpu(self, dataset):
        example_idx, input_tensors = dataset[0]
        loss = self.model.compute_loss(input_tensors)
        print(loss.item())
        self.optimizer.zero_grad()
        # torch.cuda.empty_cache()
        loss.backward()
        self.optimizer.step()


    def train(self, data_loaders):
        if os.path.exists(self.config['log_dir']) or self.config['loads_ckpt'] or self.config['loads_best_ckpt']:
            self.load_ckpt()

        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)

        start_epoch_idx = self.epoch_idx

        for epoch_idx in range(start_epoch_idx, self.config['num_epochs']):
            self.epoch_idx = epoch_idx

            print(f'starting epoch {epoch_idx}')
            print('training')

            self.model.train()

            avg_epoch_loss = 0.
            batch_num = 0
            next_logging_pct = .5
            next_evaluating_pct = self.config["next_evaluating_pct"] 
            start_time = time.time()

            for example_idx, input_tensors, cand_mention_labels in data_loaders['train']:
                input_tensors = [t.cuda() for t in input_tensors]
                cand_mention_labels = cand_mention_labels.cuda()
                batch_num += 1
                pct = batch_num / len(data_loaders['train'])

                self.optimizer.zero_grad()

                # print(example_idx)
                (
                    # [cand_num]
                    cand_mention_scores,
                    # [top_cand_num]
                    top_start_idxes,
                    # [top_cand_num]
                    top_end_idxes,
                    # [top_cand_num]
                    top_span_cluster_ids,
                    # [top_span_num, pruned_ant_num]
                    top_ant_idxes_of_spans,
                    # [top_cand_num, pruned_ant_num]
                    top_ant_cluster_ids_of_spans,
                    # # [top_cand_num, 1 + pruned_ant_num]
                    top_ant_scores_of_spans,
                    # 4 * [top_cand_num, 1 + pruned_ant_num]
                    # list_of_top_ant_scores_of_spans,
                    # [top_span_num, pruned_ant_num]
                    top_ant_mask_of_spans,
                    # [top_span_num, 1 + top_span_num], [top_span_num, top_span_num]
                    full_fast_ant_scores_of_spans, full_ant_mask_of_spans
                ) = self.model(*input_tensors)

                ant_loss = Runner.compute_ant_loss(
                    # [cand_num]
                    cand_mention_scores,
                    # [top_cand_num]
                    top_start_idxes,
                    # [top_cand_num]
                    top_end_idxes,
                    # [top_cand_num]
                    top_span_cluster_ids,
                    # [top_span_num, pruned_ant_num]
                    top_ant_idxes_of_spans,
                    # [top_cand_num, pruned_ant_num]
                    top_ant_cluster_ids_of_spans,
                    # # [top_cand_num, 1 + pruned_ant_num]
                    top_ant_scores_of_spans,
                    # 4 * [top_cand_num, 1 + pruned_ant_num]
                    # list_of_top_ant_scores_of_spans,
                    # [top_span_num, pruned_ant_num]
                    top_ant_mask_of_spans,
                    # [top_span_num, 1 + top_span_num], [top_span_num, top_span_num]
                    full_fast_ant_scores_of_spans, full_ant_mask_of_spans
                )

                loss = ant_loss

                avg_epoch_loss += loss.item()
                loss.backward()

                self.optimizer.step()

                if pct >= next_logging_pct:
                    na_str = 'N/A'

                    print(
                        f'{int(pct)}%, time: {time.time() - start_time} avg_epoch_loss: {avg_epoch_loss / batch_num}'
                    )

                    next_logging_pct += self.config["next_logging_pct"]

                    iter_now = int(len(PrpDataset) * (epoch_idx + pct))
                    self.writer.add_scalar('Train/loss', avg_epoch_loss / batch_num, iter_now)


                if pct >= next_evaluating_pct:
                    iter_now = int(len(PrpDataset) * (epoch_idx + pct))
                    avg_f1 = self.evaluate(data_loaders['val'], iter_now)

                    if avg_f1 > self.max_f1:
                        self.max_f1 = avg_f1
                        self.max_f1_epoch_idx = epoch_idx + pct

                        ckpt_path = self.save_ckpt_best()

                    next_evaluating_pct += self.config["next_evaluating_pct"]

            avg_epoch_loss /= batch_num

            na_str = 'N/A'

            print(
                f'100%,\ttime:\t{time.time() - start_time} avg_epoch_loss:\t{avg_epoch_loss}'
            )


            iter_now = int(len(PrpDataset) * (epoch_idx + pct))
            avg_f1 = self.evaluate(data_loaders['val'], iter_now)
            ckpt_path = self.save_ckpt()

            if self.config["max_ckpt_to_keep"] > 0:
                if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                    todel = self.checkpoint_queue.popleft()
                    os.remove(todel)
                self.checkpoint_queue.append(ckpt_path)

            if avg_f1 > self.max_f1:
                self.max_f1 = avg_f1
                self.max_f1_epoch_idx = epoch_idx + 1

                best_ckpt_path = ckpt_path.replace(f'{self.epoch_idx}.ckpt', 'best.ckpt')
                shutil.copyfile(ckpt_path, best_ckpt_path)
                print(f'Saving {best_ckpt_path}.')
            elif epoch_idx - self.max_f1_epoch_idx > self.config["early_stop_epoch"]:
                print('Early stop.')
                break


    def evaluate(self, data_loader, iter_now=0, name='test', saves_results=False):
        # from collections import Counter
        # span_len_cnts = Counter()

        with torch.no_grad():
            print('evaluating')
            evaluator = metrics.CorefEvaluator()
            pr_coref_evaluator = metrics.PrCorefEvaluator()

            self.model.eval()
            batch_num = 0
            avg_loss = 0.
            next_logging_pct = 10.
            start_time = time.time()
            cluster_predictions = {}

            for example_idx, input_tensors, cand_mention_labels in data_loader:
                input_tensors = [t.cuda() for t in input_tensors]
                cand_mention_labels = cand_mention_labels.cuda()
                batch_num += 1
                pct = batch_num / len(data_loader)

                (
                    # [cand_num]
                    cand_mention_scores,
                    # [top_cand_num]
                    top_start_idxes,
                    # [top_cand_num]
                    top_end_idxes,
                    # [top_cand_num]
                    top_span_cluster_ids,
                    # [top_span_num, pruned_ant_num]
                    top_ant_idxes_of_spans,
                    # [top_cand_num, pruned_ant_num]
                    top_ant_cluster_ids_of_spans,
                    # # [top_cand_num, 1 + pruned_ant_num]
                    top_ant_scores_of_spans,
                    # 4 * [top_cand_num, 1 + pruned_ant_num]
                    # list_of_top_ant_scores_of_spans,
                    # [top_span_num, pruned_ant_num]
                    top_ant_mask_of_spans,
                    # [doc_len, pos_tag_num]
                    pos_tag_logits,
                    # [top_span_num, 1 + top_span_num], [top_span_num, top_span_num]
                    full_fast_ant_scores_of_spans, full_ant_mask_of_spans
                ) = self.model(*input_tensors)


                ant_loss = Runner.compute_ant_loss(
                    # [cand_num]
                    cand_mention_scores,
                    # [top_cand_num]
                    top_start_idxes,
                    # [top_cand_num]
                    top_end_idxes,
                    # [top_cand_num]
                    top_span_cluster_ids,
                    # [top_span_num, pruned_ant_num]
                    top_ant_idxes_of_spans,
                    # [top_cand_num, pruned_ant_num]
                    top_ant_cluster_ids_of_spans,
                    # # [top_cand_num, 1 + pruned_ant_num]
                    top_ant_scores_of_spans,
                    # 4 * [top_cand_num, 1 + pruned_ant_num]
                    # list_of_top_ant_scores_of_spans,
                    # [top_span_num, pruned_ant_num]
                    top_ant_mask_of_spans,
                    # [top_span_num, 1 + top_span_num], [top_span_num, top_span_num]
                    full_fast_ant_scores_of_spans, full_ant_mask_of_spans
                )

                loss = ant_loss

                avg_loss += loss.item()

                (
                    top_start_idxes, top_end_idxes, predicted_ant_idxes,
                    predicted_clusters, span_to_predicted_cluster
                ) = Runner.predict(
                    # [cand_num]
                    cand_mention_scores,
                    # [top_cand_num]
                    top_start_idxes,
                    # [top_cand_num]
                    top_end_idxes,
                    # [top_cand_num]
                    top_span_cluster_ids,
                    # [top_span_num, pruned_ant_num]
                    top_ant_idxes_of_spans,
                    # [top_cand_num, pruned_ant_num]
                    top_ant_cluster_ids_of_spans,
                    # # [top_cand_num, 1 + pruned_ant_num]
                    # top_ant_scores_of_spans,
                    # 4 * [top_cand_num, 1 + pruned_ant_num]
                    # list_of_top_ant_scores_of_spans,
                    # [top_span_num, pruned_ant_num]
                    top_ant_mask_of_spans
                )

                gold_clusters = data_loader.get_gold_clusters(example_idx)
                gold_clusters = [
                    tuple(tuple(span) for span in cluster)
                    for cluster in gold_clusters
                ]
                span_to_gold_cluster = {
                    span: cluster
                    for cluster in gold_clusters
                    for span in cluster
                }

                evaluator.update(
                    predicted=predicted_clusters,
                    gold=gold_clusters,
                    mention_to_predicted=span_to_predicted_cluster,
                    mention_to_gold=span_to_gold_cluster
                )
                cluster_predictions[data_loader.get_doc_key(example_idx)] = predicted_clusters

                pr_coref_evaluator.update(predicted_clusters, data_loader.get_pronoun_info(example_idx), 
                                          data_loader.get_sentences(example_idx))

                if pct >= next_logging_pct:
                    print(
                        f'{int(pct)}%,\ttime:\t{time.time() - start_time}'
                    )
                    next_logging_pct += 5.

            epoch_precision, epoch_recall, epoch_f1 = evaluator.get_prf()
            avg_loss = avg_loss / batch_num

            na_str = 'N/A'

            print(
                f'avg_valid_time:\t{time.time() - start_time}\n'
                f'avg loss:\t{avg_loss}\n'
                f'Coref average precision:\t{epoch_precision}\n'
                f'Coref average recall:\t{epoch_recall}\n'
                f'Coref average f1:\t{epoch_f1}\n'
            )

            self.writer.add_scalar('Val/loss', avg_loss, iter_now)
            self.writer.add_scalar('Val/coref precision', epoch_precision, iter_now)
            self.writer.add_scalar('Val/coref recall', epoch_recall, iter_now)
            self.writer.add_scalar('Val/coref f1', epoch_f1, iter_now)

            pr_coref_results = pr_coref_evaluator.get_prf()

            print(
                f'avg_valid_time:\t{time.time() - start_time}\n'
                f'Pronoun Coref average precision:\t{pr_coref_results["p"]}\n'
                f'Pronoun Coref average recall:\t{pr_coref_results["r"]}\n'
                f'Pronoun Coref average f1:\t{pr_coref_results["f"]}\n'
            )

            self.writer.add_scalar('Val/pronoun coref precision', pr_coref_results['p'], iter_now)
            self.writer.add_scalar('Val/pronoun coref recall', pr_coref_results['r'], iter_now)
            self.writer.add_scalar('Val/pronoun coref f1', pr_coref_results['f'], iter_now)

            return pr_coref_results['f']

    def get_ckpt(self):
        return {
            'epoch_idx': self.epoch_idx,
            'max_f1': self.max_f1,
            'seed': self.config['random_seed'],
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'embedder_optimizer': self.embedder_optimizer.state_dict() if not self.config['freezes_embeddings'] else None,
        }

    def set_ckpt(self, ckpt_dict):
        if not self.config['restarts']:
            self.epoch_idx = ckpt_dict['epoch_idx'] + 1

        if not self.config['resets_max_f1']:
            self.max_f1 = ckpt_dict['max_f1']

        model_state_dict = self.model.state_dict()
        model_state_dict.update(
            {
                name: param
                for name, param in ckpt_dict['model'].items()
                if name in model_state_dict
            }
        )

        self.model.load_state_dict(model_state_dict)
        print('loaded model')
        del model_state_dict

        if not (self.config['uses_new_optimizer'] or self.config['sets_new_lr']):
            #     if ckpt_dict['embedder_optimizer'] and not self.config['freezes_embeddings']:
            #         self.embedder_optimizer.load_state_dict(ckpt_dict['embedder_optimizer'])
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            print('loaded optimizer')

        del ckpt_dict

        torch.cuda.empty_cache()

    ckpt = property(get_ckpt, set_ckpt)

    def save_ckpt(self):
        ckpt_path = f'{self.config["log_dir"]}/epoch_{self.epoch_idx}.ckpt'
        print(f'saving checkpoint {ckpt_path}')
        torch.save(self.ckpt, f=ckpt_path)
        return ckpt_path

    def save_best_ckpt(self):
        ckpt_path = f'{self.config["log_dir"]}/best.ckpt'
        print(f'saving checkpoint {ckpt_path}')
        torch.save(self.ckpt, f=ckpt_path)
        return ckpt_path

    def load_ckpt(self, ckpt_path=None):
        if not ckpt_path:
            if self.config['loads_best_ckpt']:
                ckpt_path = f'{self.config["log_dir"]}/best.ckpt'
            else:
                ckpt_paths = [path for path in os.listdir(f'{self.config["log_dir"]}/') if path.endswith('.ckpt')]
                if len(ckpt_paths) == 0:
                    print(f'No .ckpt found in {self.config["log_dir"]}')
                    return
                sort_func = lambda x:int(re.search(r"(\d+)", x).groups(0))
                ckpt_path = f'{self.config["log_dir"]}/{sorted(ckpt_paths, key=sort_func)}'

        print(f'loading checkpoint {ckpt_path}')

        self.ckpt = torch.load(ckpt_path)


if __name__ == '__main__':
    args = parser.parse_args()
    if len(sys.argv) == 1:
        sys.argv.append(args.model)
    else:
        sys.argv[1] = args.model  
    if len(sys.argv) == 2:
        sys.argv.append(args.mode)
    else:
        sys.argv[2] = args.mode  

    # initialization
    config = initialize_from_env()
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed(config['random_seed'])
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    # set log file
    config["log_dir"] = os.path.join(args.log_dir, args.model)
    if not os.path.exists(config["log_dir"]):
      os.makedirs(config["log_dir"])
    
    log_file = os.path.join(config["log_dir"], f'{args.mode}.log')
    set_log_file(log_file)    

    config['training'] = args.mode == 'train'
    config['validating'] = args.mode == 'eval'
    config['debugging'] = args.mode == 'debug'

    # prepare dataset
    if config['training']:
        splits = {'train': 'train', 'eval': 'val'}
    elif config['validating']: 
        splits = {'eval': 'test'}
    elif config['debugging']:
        splits = {'train': 'val', 'eval': 'test'}
    datasets = {
        split: PrpDataset(splits[split], config)
        for split in splits
    }
    data_loaders = {
        split: tud.DataLoader(
            dataset=datasets[split],
            batch_size=1,
            shuffle=(split == 'train' and config['training']),
            # pin_memory=True,
            collate_fn=PrpDataset.collate_fn,
            num_workers=4
        )
        for split in splits
    }
    if not config['validating']:
        config['train_steps'] = len(datasets['train']) * config['num_epochs']
        config['warmup_steps'] = config['train_steps'] * 0.1

    runner = Runner(config)

    if config['training'] or config['debugging']:
        runner.train(data_loaders)
    elif config['validating']:
        runner.evaluate(data_loaders['eval'])
