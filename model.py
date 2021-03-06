from model_utils import init_params, build_len_mask_batch
from modules import Squeezer
import math
from functools import cmp_to_key, partial
import time

import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_dist_buckets = 10

        from transformers import AutoModel
        if self.config['bert_cased']:
            self.encoder = AutoModel.from_pretrained('bert-base-cased', cache_dir=self.config['bert_cache_dir'])
        else:
            self.encoder = AutoModel.from_pretrained('bert-base-uncased', cache_dir=self.config['bert_cache_dir'])

        self.span_width_embedder = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.config['max_span_width'],
                embedding_dim=self.config['feature_size']
            ),
            nn.Dropout(self.config['dropout_prob'])
        )
        self.head_scorer = nn.Sequential(
            nn.Linear(self.config['span_embedding_dim'], 1),
            Squeezer(dim=-1)
        )

        span_embedding_dim = self.config['span_embedding_dim'] * 3 + self.config['feature_size']
        mention_scorer = [
                          nn.Linear(span_embedding_dim, self.config['ffnn_size']),
                          nn.ReLU(),
                          nn.Dropout(self.config['dropout_prob']),
                          ]

        for _ in range(1, self.config['mention_scorer_depth']):
            mention_scorer.extend([
                                  nn.Linear(self.config['ffnn_size'], self.config['ffnn_size']),
                                  nn.ReLU(),
                                  nn.Dropout(self.config['dropout_prob']),
                                  ])
        mention_scorer.extend([
                              nn.Linear(self.config['ffnn_size'], 1),
                              Squeezer(dim=-1)
                              ])
        self.mention_scorer = nn.Sequential(*mention_scorer)
        # self.mention_scorer = nn.Sequential(
        #     nn.Linear(span_embedding_dim, self.config['ffnn_size']),
        #     nn.ReLU(),
        #     nn.Dropout(self.config['dropout_prob']),
        #     # nn.Linear(self.config['ffnn_size'], self.config['ffnn_size']),
        #     # nn.ReLU(),
        #     # nn.Dropout(self.config['dropout_prob']),
        #     nn.Linear(self.config['ffnn_size'], 1),
        #     Squeezer(dim=-1)
        # )

        span_width_scorer = [
                             nn.Embedding(
                                 num_embeddings=self.config['max_span_width'],
                                 embedding_dim=self.config['feature_size']
                             ),
                             nn.Linear(self.config['feature_size'], self.config['ffnn_size']),
                             nn.ReLU(),
                             nn.Dropout(self.config['dropout_prob']),
                             ]
        for _ in range(1, self.config['span_width_scorer_depth']):
            span_width_scorer.extend([
                                     nn.Linear(self.config['ffnn_size'], self.config['ffnn_size']),
                                     nn.ReLU(),
                                     nn.Dropout(self.config['dropout_prob']),
                                     ])
        span_width_scorer.extend([
                                 nn.Linear(self.config['ffnn_size'], 1),
                                 Squeezer(dim=-1)
                                 ])
        self.span_width_scorer = nn.Sequential(*span_width_scorer)
        # self.span_width_scorer = nn.Sequential(
        #     nn.Embedding(
        #         num_embeddings=self.config['max_span_width'],
        #         embedding_dim=self.config['feature_size']
        #     ),
        #     nn.Linear(self.config['feature_size'], self.config['ffnn_size']),
        #     nn.ReLU(),
        #     nn.Dropout(self.config['dropout_prob']),
        #     # nn.Linear(self.config['ffnn_size'], self.config['ffnn_size']),
        #     # nn.ReLU(),
        #     # nn.Dropout(self.config['dropout_prob']),
        #     nn.Linear(self.config['ffnn_size'], 1),
        #     Squeezer(dim=-1)
        # )

        self.ant_distance_scorer = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.num_dist_buckets,
                embedding_dim=self.config['feature_size']
            ),
            nn.Dropout(self.config['dropout_prob']),
            nn.Linear(self.config['feature_size'], 1),
            Squeezer(dim=-1)
        )        

        self.src_span_projector = nn.Linear(
            span_embedding_dim, span_embedding_dim
        )

        self.genre_embedder = nn.Embedding(
            num_embeddings=len(self.config['id_to_genre']),
            embedding_dim=self.config['feature_size']
        )

        self.speaker_pair_embedder = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.config['feature_size']
        )

        self.ant_offset_embedder = nn.Embedding(
            num_embeddings=self.num_dist_buckets,
            embedding_dim=self.config['feature_size']
        )

        pair_embedding_dim = (span_embedding_dim + self.config['feature_size']) * 3
        slow_ant_scorer = [
                           nn.Linear(pair_embedding_dim, self.config['ffnn_size']),
                           nn.ReLU(),
                           nn.Dropout(self.config['dropout_prob']),
                           ]
        for _ in range(1, self.config['slow_ant_scorer_depth']):
            slow_ant_scorer.extend([
                                   nn.Linear(self.config['ffnn_size'], self.config['ffnn_size']),
                                   nn.ReLU(),
                                   nn.Dropout(self.config['dropout_prob']),
                                   ])
        slow_ant_scorer.extend([
                               nn.Linear(self.config['ffnn_size'], 1),
                               Squeezer(dim=-1)
                               ])
        self.slow_ant_scorer = nn.Sequential(*slow_ant_scorer)
        # self.slow_ant_scorer = nn.Sequential(
        #     nn.Linear(pair_embedding_dim, self.config['ffnn_size']),
        #     nn.ReLU(),
        #     nn.Dropout(self.config['dropout_prob']),
        #     # nn.Linear(self.config['ffnn_size'], self.config['ffnn_size']),
        #     # nn.ReLU(),
        #     # nn.Dropout(self.config['dropout_prob']),
        #     nn.Linear(self.config['ffnn_size'], 1),
        #     Squeezer(dim=-1)
        # )

        self.attended_span_embedding_gate = nn.Sequential(
            nn.Linear(span_embedding_dim * 2, span_embedding_dim),
            nn.Sigmoid()
        )

        self.init_params()


    def init_params(self):
        for name, module in self.named_children():
            if 'encoder' not in name:
                module.apply(partial(init_params, initializer=self.config['initializer']))

    def get_trainable_params(self):
        yield from filter(lambda param: param.requires_grad, self.parameters())

    def get_span_emb(self, head_emb, mention_doc, start_idxes, end_idxes):
        num_words = mention_doc.size(0)

        start_embeddings, end_embeddings = mention_doc[start_idxes], mention_doc[end_idxes]

        # [span_num]
        span_widths = end_idxes - start_idxes + 1

        # [span_num]
        span_width_ids = span_widths - 1  # [k]

        # [max_span_width, span_width_embedding_dim]
        width_scores = self.span_width_embedder(torch.arange(self.config["max_span_width"], device=start_idxes.device))
        # [span_num, span_width_embedding_dim]
        span_width_embeddings = width_scores[span_width_ids]

        # [span_num, max_span_width]
        idxes_of_spans = torch.clamp(
            torch.arange(self.config['max_span_width'], dtype=start_idxes.dtype, device=start_idxes.device).view(1, -1)\
                + start_idxes.view(-1, 1),
            max=(num_words - 1)
        )

        # [span_num, max_span_width, span_width_embedding_dim]
        embeddings_of_spans = head_emb[idxes_of_spans]

        # [num_words]
        head_scores = self.head_scorer(mention_doc)

        # [span_num, max_span_width]
        head_scores_of_spans = head_scores[idxes_of_spans]

        # [span_num, max_span_width]
        span_masks = build_len_mask_batch(span_widths, self.config['max_span_width'])
        # [span_num, max_span_width]
        head_scores_of_spans.masked_fill_(~span_masks, -float('inf'))
        # [span_num, max_span_width, 1]
        attentions_of_spans = F.softmax(head_scores_of_spans, dim=1).unsqueeze(2)

        # [span_num, span_width_embedding_dim]
        attended_head_embeddings = (attentions_of_spans * embeddings_of_spans).sum(dim=1)

        # [span_num, span_embedding_dim]
        span_embeddings = torch.cat(
            (
                start_embeddings, end_embeddings, span_width_embeddings, attended_head_embeddings
            ), dim=-1
        )

        return span_embeddings

    @staticmethod
    def extract_top_spans(
        # [cand_num]
        span_scores,
        # [cand_num]
        candidate_starts,
        # [cand_num]
        candidate_ends,
        top_span_num,
    ):
        span_num = span_scores.size(0)

        sorted_span_idxes = torch.argsort(span_scores, descending=True).tolist()

        top_span_idxes = []
        end_idx_to_min_start_dix, start_idx_to_max_end_idx = {}, {}
        selected_span_num = 0

        # while selected_span_num < top_span_num and curr_span_idx < span_num:
        #     i = sorted_span_idxes[curr_span_idx]

        for span_idx in sorted_span_idxes:
            crossed = False
            start_idx = candidate_starts[span_idx]
            end_idx = candidate_ends[span_idx]

            if end_idx == start_idx_to_max_end_idx.get(start_idx, -1):
                continue

            for j in range(start_idx, end_idx + 1):
                if j in start_idx_to_max_end_idx and j > start_idx and start_idx_to_max_end_idx[j] > end_idx:
                    crossed = True
                    break

                if j in end_idx_to_min_start_dix and j < end_idx and end_idx_to_min_start_dix[j] < start_idx:
                    crossed = True
                    break

            if not crossed:
                top_span_idxes.append(span_idx)
                selected_span_num += 1

                if start_idx not in start_idx_to_max_end_idx or end_idx > start_idx_to_max_end_idx[start_idx]:
                    start_idx_to_max_end_idx[start_idx] = end_idx

                if end_idx not in end_idx_to_min_start_dix or start_idx < end_idx_to_min_start_dix[end_idx]:
                    end_idx_to_min_start_dix[end_idx] = start_idx

            if selected_span_num == top_span_num:
                break

        def compare_span_idxes(i1, i2):
            if candidate_starts[i1] < candidate_starts[i2]:
                return -1
            elif candidate_starts[i1] > candidate_starts[i2]:
                return 1
            elif candidate_ends[i1] < candidate_ends[i2]:
                return -1
            elif candidate_ends[i1] > candidate_ends[i2]:
                return 1
            # elif i1 < i2:
            #     return -1
            # elif i1 > i2:
            #     return 1
            else:
                return 0

        top_span_idxes.sort(key=cmp_to_key(compare_span_idxes))

        # for span_idx in range(1, len(top_span_idxes)):
        #     assert compare_span_idxes(span_idx - 1, span_idx) == -1

        # last_end_idx = -1
        #
        # for i in range(len(top_span_idxes)):
        #     span_idx = top_span_idxes[i]
        #     start_idx, end_idx = candidate_starts[span_idx], candidate_ends[span_idx]
        #
        #     assert start_idx <= end_idx
        #
        #     if i:
        #         assert start_idx > last_end_idx
        #
        #     last_end_idx = end_idx

        return torch.as_tensor(
            top_span_idxes
        )

    def forward(
        self,
        # [num_seg, num_words]
        input_ids,
        # [num_seg, num_words]
        input_mask,
        # [num_seg, num_words]
        speaker_ids,
        # [1]
        genre_id,
        # [gold_num]
        gold_starts,
        # [gold_num]
        gold_ends,
        # [gold_num]
        gold_cluster_ids,
        # [cand_num]
        candidate_starts,
        # [cand_num]
        candidate_ends,
        # [cand_num]
        cand_cluster_ids,
    ):
        start_time = time.time()

        # [num_seg, num_words, hidden_size]
        mention_doc = self.encoder(input_ids, attention_mask=input_mask)[0] # [num_seg, num_words, emb]
        mention_doc = self.flatten_emb_by_sentence(mention_doc, input_mask) # [num_words, emb]

        num_words = mention_doc.size(0)

        # [cand_num, span_embedding_dim]
        candidate_span_emb = self.get_span_emb(
            mention_doc, mention_doc,
            candidate_starts, candidate_ends
        )

        # [cand_num]
        cand_mention_scores = self.get_mention_scores(candidate_span_emb, candidate_starts, candidate_ends)

        top_cand_num = min(3900, int(num_words * self.config['top_span_ratio']))

        # debug
        # print('extracting top spans')

        # [top_cand_num]
        if top_cand_num < num_words:
            top_span_idxes = self.extract_top_spans(
                # [cand_num]
                cand_mention_scores,
                # [cand_num]
                candidate_starts,
                # [cand_num]
                candidate_ends,
                top_cand_num,
            )
        else:
            top_span_idxes = torch.arange(num_words)
        top_span_idxes.to(candidate_starts.device)

        # debug
        # print('top spans extracted')


        top_cand_num = top_span_idxes.size(0)

        top_start_idxes = candidate_starts[top_span_idxes]
        top_end_idxes = candidate_ends[top_span_idxes]
        # [top_cand_num, span_embedding_dim]
        top_span_embeddings = candidate_span_emb[top_span_idxes]
        # [top_cand_num]
        top_span_cluster_ids = cand_cluster_ids[top_span_idxes]
        # [top_cand_num]
        top_span_mention_scores = cand_mention_scores[top_span_idxes]

        # top_span_sent_idxes = cand_sent_idxes[top_span_idxes]

        # [top_cand_num]
        speaker_ids = self.flatten_emb_by_sentence(speaker_ids, input_mask)
        top_span_speaker_ids = speaker_ids[top_start_idxes]

        pruned_ant_num = min(self.config['max_top_antecedents'], top_cand_num)

        # debug
        # print('pruning ants')

        (
            # [top_span_num, pruned_ant_num], [top_span_num, pruned_ant_num]
            top_ant_idxes_of_spans, top_ant_mask_of_spans,
            # [top_span_num, pruned_ant_num], [top_span_num, pruned_ant_num]
            top_fast_ant_scores_of_spans, top_ant_offsets_of_spans,
            # [top_span_num, top_span_num], [top_span_num, top_span_num]
            full_fast_ant_scores_of_spans, full_ant_mask_of_spans
        ) = self.prune(
            # [top_cand_num, span_embedding_dim]
            top_span_embeddings,
            # [top_cand_num]
            top_span_mention_scores,
            pruned_ant_num
        )

        # leave out segment distance here

        # top_fast_ant_scores_of_spans = top_fast_ant_scores_of_spans.to(torch.device(1))

        # [top_cand_num, 1]
        dummy_scores = torch.zeros(top_cand_num, 1, device=top_fast_ant_scores_of_spans.device)
        # top_span_embeddings = top_span_embeddings.to(torch.device(1))

        top_ant_scores_of_spans = None

        # [genre_embedding_dim]
        genre_embedding = self.genre_embedder(genre_id.view(1, 1)).view(-1)

        for i in range(self.config['coref_depth']):

            # [top_span_num, pruned_ant_num, span_embedding_dim]
            top_ant_embeddings_of_spans = top_span_embeddings[top_ant_idxes_of_spans]
            top_slow_ant_scores_of_spans = self.get_slow_ant_scores_of_spans(
                # [top_cand_num, span_embedding_dim]
                top_span_embeddings,
                # [top_span_num, pruned_ant_num]
                top_ant_idxes_of_spans,
                # [top_span_num, pruned_ant_num, span_embedding_dim]
                top_ant_embeddings_of_spans,
                # [top_span_num, pruned_ant_num]
                top_ant_offsets_of_spans,
                # [top_cand_num]
                top_span_speaker_ids,
                # [genre_embedding_dim]
                genre_embedding
            )

            # [top_cand_num, pruned_ant_num]
            top_ant_scores_of_spans = top_fast_ant_scores_of_spans + top_slow_ant_scores_of_spans

            # [top_cand_num, 1 + pruned_ant_num]
            top_ant_scores_of_spans = torch.cat(
                (
                    # [top_cand_num, 1]
                    dummy_scores,
                    # [top_cand_num, pruned_ant_num]
                    top_ant_scores_of_spans
                ), dim=1
            )

            if i == self.config['coref_depth'] - 1:
                break

            # [top_cand_num, 1 + pruned_ant_num]
            top_ant_attentions_of_spans = F.softmax(
                # [top_cand_num, 1 + pruned_ant_num]
                top_ant_scores_of_spans, dim=-1
            )
            # [top_cand_num, 1 + pruned_ant_num, span_embedding_dim]
            top_ant_embeddings_of_spans = torch.cat(
                (top_span_embeddings.view(top_cand_num, 1, -1), top_ant_embeddings_of_spans), dim=1
            )
            # [top_cand_num, span_embedding_dim]
            attended_top_span_embeddings = \
                (
                    # [top_cand_num, 1 + pruned_ant_num, 1]
                    top_ant_attentions_of_spans.view(top_cand_num, -1, 1)
                    # [top_cand_num, 1 + pruned_ant_num, span_embedding_dim]
                    * top_ant_embeddings_of_spans
                ).sum(1)

            # [top_cand_num, span_embedding_dim]
            g = self.attended_span_embedding_gate(
                # [top_cand_num, span_embedding_dim + span_embedding_dim]
                torch.cat(
                    (top_span_embeddings, attended_top_span_embeddings), dim=1
                )
            )

            top_span_embeddings = g * attended_top_span_embeddings + (1. - g) * top_span_embeddings
            # top_span_embeddings = attended_top_span_embeddings

        # [top_cand_num, pruned_ant_num]
        top_ant_cluster_ids_of_spans = top_span_cluster_ids[top_ant_idxes_of_spans]

        # [top_span_num, 1 + top_span_num]
        full_fast_ant_scores_of_spans = torch.cat(
            (
                # [top_span_num, 1]
                dummy_scores,
                # [top_span_num, top_span_num]
                full_fast_ant_scores_of_spans
            ), dim=1
        )

        return (
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

    def get_slow_ant_scores_of_spans(
        self,
        # [top_cand_num, span_embedding_dim]
        top_span_embeddings,
        # [top_span_num, pruned_ant_num]
        top_ant_idxes_of_spans,
        # [top_span_num, pruned_ant_num, span_embedding_dim]
        top_ant_embeddings_of_spans,
        # [top_span_num, pruned_ant_num]
        top_ant_offsets_of_spans,
        # [top_cand_num]
        top_span_speaker_ids,
        # [genre_embedding_dim]
        genre_embedding
    ):
        top_span_num, pruned_ant_num = top_ant_idxes_of_spans.size()
        # [top_span_num, pruned_ant_num]
        top_ant_speaker_ids_of_spans = top_span_speaker_ids[top_ant_idxes_of_spans]
        # [top_span_num, pruned_ant_num, speaker_pair_embedding_dim]
        speaker_pair_embeddings_of_spans = self.speaker_pair_embedder(
            # [top_span_num, pruned_ant_num]
            (top_span_speaker_ids.view(-1, 1) == top_ant_speaker_ids_of_spans).long().to(top_span_embeddings.device)
        )
        # [top_span_num, pruned_ant_num, ant_offset_embedding_dim]
        ant_offset_embeddings_of_spans = self.ant_offset_embedder(
            self.get_offset_bucket_idxes_batch(top_ant_offsets_of_spans).to(top_span_embeddings.device)
        )
        feature_embeddings_of_spans = torch.cat(
            (
                speaker_pair_embeddings_of_spans,
                # [top_span_num, pruned_ant_num, feature_size]
                genre_embedding.view(1, 1, -1).repeat(top_span_num, pruned_ant_num, 1),
                ant_offset_embeddings_of_spans
            ), dim=-1
        )
        # [top_span_num, pruned_ant_num, feature_size * 3]
        feature_embeddings_of_spans = F.dropout(
            feature_embeddings_of_spans, p=self.config['dropout_prob'], training=self.training
        )
        # [top_span_num, pruned_ant_num, span_embedding_dim] * [top_cand_num, 1, span_embedding_dim]
        similarity_embeddings_of_spans = top_ant_embeddings_of_spans \
                                         * top_span_embeddings.view(top_span_num, 1, -1)

        pair_embeddings_of_spans = torch.cat(
            (
                # [top_span_num, pruned_ant_num, span_embedding_dim]
                top_span_embeddings.view(top_span_num, 1, -1).repeat(1, pruned_ant_num, 1),
                # [top_span_num, pruned_ant_num, span_embedding_dim]
                top_ant_embeddings_of_spans,
                # [top_span_num, pruned_ant_num, span_embedding_dim]
                similarity_embeddings_of_spans,
                # [top_span_num, pruned_ant_num, feature_size * 3]
                feature_embeddings_of_spans
            ), dim=-1
        )

        # [top_span_num, pruned_ant_num]
        slow_ant_scores_of_spans = self.slow_ant_scorer(pair_embeddings_of_spans)
        # [top_span_num, pruned_ant_num]
        return slow_ant_scores_of_spans

    def prune(
        self,
        # [top_cand_num, span_embedding_dim]
        top_span_embeddings,
        # [top_cand_num]
        top_span_mention_scores,
        pruned_ant_num
    ):
        top_span_num = top_span_embeddings.size(0)

        span_idxes = torch.arange(top_span_num, device=top_span_embeddings.device)
        # [top_span_num, top_span_num]
        antecedent_offsets = span_idxes.view(-1, 1) - span_idxes.view(1, -1)
        # [top_span_num, top_span_num]
        full_ant_mask_of_spans = antecedent_offsets >= 1
        # [top_span_num, top_span_num]
        full_fast_ant_scores_of_spans = top_span_mention_scores.view(-1, 1) + top_span_mention_scores.view(1, -1)
        full_fast_ant_scores_of_spans.masked_fill_(~full_ant_mask_of_spans, -float('inf'))
        full_fast_ant_scores_of_spans += self.get_fast_ant_scores_of_spans(top_span_embeddings)

        # [top_span_num, top_span_num]
        antecedent_distance_buckets = self.get_offset_bucket_idxes_batch(antecedent_offsets).to(top_span_embeddings.device)
        distance_scores = self.ant_distance_scorer(torch.arange(self.num_dist_buckets, device=top_span_embeddings.device))
        antecedent_distance_scores = distance_scores[antecedent_distance_buckets]
        full_fast_ant_scores_of_spans += antecedent_distance_scores

        # [top_span_num, pruned_ant_num] on cuda
        _, top_ant_idxes_of_spans = torch.topk(
            # [top_span_num, top_span_num]
            full_fast_ant_scores_of_spans, k=pruned_ant_num, dim=-1, sorted=False
        )
        # top_ant_idxes_of_spans = top_ant_idxes_of_spans.cpu()
        # [top_span_num, 1]
        span_idxes = span_idxes.view(-1, 1)
        # [top_span_num, pruned_ant_num]
        top_ant_mask_of_spans = full_ant_mask_of_spans[span_idxes, top_ant_idxes_of_spans]
        # [top_span_num, pruned_ant_num]
        top_fast_ant_scores_of_spans = full_fast_ant_scores_of_spans[span_idxes, top_ant_idxes_of_spans]
        # [top_span_num, pruned_ant_num]
        top_ant_offsets_of_spans = antecedent_offsets[span_idxes, top_ant_idxes_of_spans]

        return (
            # [top_span_num, pruned_ant_num], [top_span_num, pruned_ant_num]
            top_ant_idxes_of_spans, top_ant_mask_of_spans,
            # [top_span_num, pruned_ant_num], [top_span_num, pruned_ant_num]
            top_fast_ant_scores_of_spans, top_ant_offsets_of_spans,
            # [top_span_num, top_span_num], [top_span_num, top_span_num]
            full_fast_ant_scores_of_spans, full_ant_mask_of_spans
        )

    def get_fast_ant_scores_of_spans(
        self,
        # [top_cand_num, span_embedding_dim]
        top_span_embeddings
    ):
        # [top_cand_num, span_embedding_dim]
        top_src_span_embeddings = F.dropout(
            self.src_span_projector(top_span_embeddings),
            p=self.config['fast_ant_score_dropout_prob'], training=self.training
        )
        # [top_cand_num, span_embedding_dim]
        top_tgt_span_embeddings = F.dropout(top_span_embeddings, p=self.config['fast_ant_score_dropout_prob'], training=self.training)
        # [top_span_num, top_span_num] = # [top_cand_num, span_embedding_dim] @ [span_embedding_dim, top_cand_num]
        return top_src_span_embeddings @ top_tgt_span_embeddings.t()


    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = emb.size(0)
        max_sentence_length = emb.size(1)

        emb_rank = len(emb.size())
        if emb_rank == 2:
            flattened_emb = emb.reshape(num_sentences * max_sentence_length)
        elif emb_rank == 3:
            flattened_emb = emb.reshape(num_sentences * max_sentence_length, emb.size(2))
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return flattened_emb[text_len_mask.bool().reshape(num_sentences * max_sentence_length)]


    def get_mention_scores(self, span_emb, span_starts, span_ends):
        span_scores = self.mention_scorer(span_emb)
        width_scores = self.span_width_scorer(torch.arange(self.config["max_span_width"], device=span_starts.device))
        span_width_index = span_ends - span_starts # [NC]
        width_scores = width_scores[span_width_index]
        span_scores += width_scores
        return span_scores


    def get_offset_bucket_idxes_batch(self, offsets_batch):
        """
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        offsets_batch_for_log = offsets_batch.masked_fill(offsets_batch <= 1, 1).float()
        log_space_idxes_batch = (torch.log(offsets_batch_for_log) / math.log(2)).floor().long() + 3

        identity_mask_batch = (offsets_batch <= 4).long()

        return torch.clamp(
            identity_mask_batch * offsets_batch + (1 - identity_mask_batch) * log_space_idxes_batch,
            min=0, max=9
        )
