import os
import math
from tqdm import tqdm
from collections import defaultdict, Counter
from itertools import chain
import time
import json
# import Levenshtein
import csv
import bisect
# from PIL import Image
import numpy as np
import random
import pdb
import h5py
import torch
import torch.utils.data as tud
from transformers import AutoTokenizer

import util


# names = 'test', 'dev'


# def build_mask_batch(
#         # [batch_size], []
#         len_batch, max_len
# ):
#     batch_size, = len_batch.shape
#     # [batch_size, max_len]
#     idxes_batch = np.arange(max_len).reshape(1, -1).repeat(batch_size, axis=0)
#     # [batch_size, max_len] = [batch_size, max_len] >= [batch_size, 1]
#     return idxes_batch < len_batch.reshape(-1, 1)


class PrpDataset(tud.Dataset):
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.genre_to_id = {genre: id_ for id_, genre in enumerate(self.config['id_to_genre'])}
        self.examples = [json.loads(line) for line in open(self.config[f'{name}_path'])]
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=self.config['bert_cache_dir'])

    def __len__(self):
        return len(self.examples)

    def get_gold_clusters(self, example_idx):
        return self.examples[example_idx]['clusters']

    def get_doc_key(self, example_idx):
        return self.examples[example_idx]['doc_key']

    def get_pronoun_info(self, example_idx):
        return self.examples[example_idx]['pronoun_info']

    def get_sentences(self, example_idx):
        return self.examples[example_idx]['sentences']

    def get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for s in speakers:
            if s not in speaker_dict and len(speaker_dict) < self.config['max_num_speakers']:
                speaker_dict[s] = len(speaker_dict)
        return speaker_dict

    @staticmethod
    def truncate_example(example, max_sent_num):
        sents = example['sentences']
        sent_num = len(sents)
        sent_lens = [len(sent) for sent in sents]
        start_sent_idx = random.randint(0, sent_num - max_sent_num)
        end_sent_idx = start_sent_idx + max_sent_num
        start_word_idx = sum(sent_lens[:start_sent_idx])
        end_word_idx = sum(sent_lens[:end_sent_idx])

        clusters = [
            [
                (l - start_word_idx, r - start_word_idx)
                for l, r in cluster
                if start_word_idx <= l <= r < end_word_idx
            ] for cluster in example['clusters']
        ]
        clusters = [cluster for cluster in clusters if cluster]

        return {
            'sentences': example['sentences'][start_sent_idx:end_sent_idx],
            'clusters': clusters,
            'speakers': example['speakers'][start_sent_idx:end_sent_idx],
            'doc_key': example['doc_key'],
            'pos': example['pos'][start_word_idx:end_word_idx],
            'start_sent_idx': start_sent_idx
        }


    def __getitem__(self, example_idx):
        start_time = time.time()

        example = self.examples[example_idx]

        if self.name == 'train' and len(example['sentences']) > self.config['max_sent_num']:
            raise ValueError(f'example {example_idx} needs truncation')
            example = Dataset.truncate_example(example, self.config['max_sent_num'])

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = example["speakers"]
        speaker_dict = self.get_speaker_dict(util.flatten(speakers) + ['caption'])

        text_len = np.array([len(s) for s in sentences])
        max_sentence_length = max(text_len)

        input_ids, input_mask, speaker_ids = [], [], []
        for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
            # sent_input_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
            sentence = [word.lower() for word in sentence]
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
            sent_len = len(sent_input_ids)
            sent_input_mask = [1] * sent_len
            sent_speaker_ids = [speaker_dict.get(s, 3) for s in speaker]
            sent_input_ids.extend([0] * (max_sentence_length - sent_len))
            sent_input_mask.extend([0] * (max_sentence_length - sent_len))
            sent_speaker_ids.extend([0] * (max_sentence_length - sent_len))
            input_ids.append(sent_input_ids)
            speaker_ids.append(sent_speaker_ids)
            input_mask.append(sent_input_mask)
        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        speaker_ids = torch.tensor(speaker_ids)
        assert num_words == torch.sum(input_mask).item(), (num_words, torch.sum(input_mask))

        doc_key = example['doc_key'].replace('/', ':')

        genre_id = torch.as_tensor([self.genre_to_id[doc_key[:2]]])

        clusters = example['clusters']

        gold_mentions = sorted(
            tuple(mention) for cluster in clusters
            for mention in cluster
        )

        gold_mention_to_id = {
            mention: id_
            for id_, mention in enumerate(gold_mentions)
        }

        gold_starts, gold_ends = map(
            # np.array,
            torch.as_tensor,
            zip(*gold_mentions) if gold_mentions else ([], [])
        )

        gold_cluster_ids = torch.zeros(len(gold_mentions), dtype=torch.long)

        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                # leave cluster_id of 0 for dummy
                gold_cluster_ids[gold_mention_to_id[tuple(mention)]] = cluster_id + 1

        # [num_words, max_span_width]
        candidate_starts = torch.arange(num_words).view(-1, 1).repeat(1, self.config['max_span_width'])
        # [num_words, max_span_width]
        cand_cluster_ids = torch.zeros_like(candidate_starts)

        if gold_mentions:
            gold_end_offsets = gold_ends - gold_starts
            gold_mention_mask = gold_end_offsets < self.config['max_span_width']
            filtered_gold_starts = gold_starts[gold_mention_mask]
            filtered_gold_end_offsets = gold_end_offsets[gold_mention_mask]
            filtered_gold_cluster_ids = gold_cluster_ids[gold_mention_mask]
            cand_cluster_ids[filtered_gold_starts, filtered_gold_end_offsets] = filtered_gold_cluster_ids

        # [num_words * max_span_width]
        candidate_ends = (candidate_starts + torch.arange(self.config['max_span_width']).view(1, -1)).view(-1)

        sentence_indices = torch.tensor(example['sentence_map'])
        # remove cands with cand_ends >= num_words
        # [num_words * max_span_width]
        candidate_starts = candidate_starts.view(-1)
        # [num_words * max_span_width]
        cand_cluster_ids = cand_cluster_ids.view(-1)
        # [num_words * max_span_width]
        cand_mask = candidate_ends < num_words
        # [cand_num]
        candidate_starts = candidate_starts[cand_mask]
        # [cand_num]
        candidate_ends = candidate_ends[cand_mask]
        # [cand_num]
        cand_cluster_ids = cand_cluster_ids[cand_mask]

        # [cand_num]
        cand_start_sent_idxes = sentence_indices[candidate_starts]
        # [cand_num]
        cand_end_sent_idxes = sentence_indices[candidate_ends]
        # [cand_num]
        cand_mask = (cand_start_sent_idxes == cand_end_sent_idxes)

        # remove cands whose start and end not in the same sentences
        # [cand_num]
        candidate_starts = candidate_starts[cand_mask]
        # [cand_num]
        candidate_ends = candidate_ends[cand_mask]
        # [cand_num]
        cand_cluster_ids = cand_cluster_ids[cand_mask]

        # [cand_num]
        cand_mention_labels = cand_cluster_ids > 0

        return (
            example_idx,
            (input_ids, input_mask,
            speaker_ids, genre_id,
            gold_starts, gold_ends, gold_cluster_ids,
            candidate_starts, candidate_ends, cand_cluster_ids),
            cand_mention_labels
        )

    @staticmethod
    def collate_fn(batch):
        # batch_size = 1
        # batch, = batch
        # breakpoint()

        # return batch
        (example_idx, *tensors, cand_mention_labels), = batch
        # breakpoint()

        # assert tensors[2].dtype == np.float32

        # print(torch.as_tensor(tensors[2]).cuda().type())

        return (
            example_idx,
            tensors,
            cand_mention_labels
            # tuple(
            #     # map(
            #     #     lambda tensor: torch.as_tensor(tensor).cuda(),
            #     #     # lambda tensor: tensor.cuda(),
            #     #     tensors
            #     # )
            # )
        )



def save_predictions(name, predictions):
    # if name == 'valid':
    #     results = []
    #
    #     for prediction, word_ids, label in zip(predictions, datasets['valid'].texts, datasets['valid'].labels):
    #         results.append(
    #             {
    #                 'text': vocab.textify(word_ids),
    #                 'correct': bool(int(prediction) == label),
    #                 'label': id_to_class[label],
    #                 'prediction':  id_to_class[prediction]
    #             }
    #         )
    #
    #     json.dump(results, open('results.json', 'w'), indent=4)
    # else:
    #     np.save('predictions.npy', predictions)
    with open(f'predictions.{name}', 'w') as predictions_file:
        predictions_file.writelines(
            '\n'.join(map(lambda i: str(i.cpu().item()), predictions))
        )


def get_doc_stats(datasets, names):
    for name in names:
        max_num_words = 0
        max_sent_len = 0

        for example in datasets[name].examples:
            if name == 'train' and len(example['sentences']) > datasets[name].config['max_sent_num']:
                example = PrpDataset.truncate_example(example, datasets[name].config['max_sent_num'])

            max_num_words = max(max_num_words, sum(len(sent) for sent in example['sentences']))
            max_sent_len = max(max_sent_len, max(len(sent) for sent in example['sentences']))

        print(f'{name}: max_num_words = {max_num_words}, max_sent_len = {max_sent_len}')


if __name__ == '__main__':
    # TODO: check data_utils code by importing data
    config = util.initialize_from_env()
    names = ('val',)
    datasets = {
        name: PrpDataset(name, config)
        for name in names
    }
    get_doc_stats(datasets, names)