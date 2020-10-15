import os
import os.path as osp
import argparse
import json
import copy
import glob
import re
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(description='turn jsonlines into bert format')
parser.add_argument('--dataset', type=str, default='vispro',
                    help='dataset to transform: vispro, mscoco, neg, cap, conll, medical, nn, vispro.pool, vispro.1.0, visprp')
parser.add_argument('--max_seg_len', type=int, default=512, 
                    help='max segment len')
parser.add_argument('--model', type=str, default='bert',
                    help='model to use: bert or roberta')
parser.add_argument('--cased', action='store_true',
                    help='save cased letters')

def get_sentence_map(segments, sentence_end):
  current = 0
  sent_map = []
  sent_end_idx = 0
  assert len(sentence_end) == sum([len(s) -2 for s in segments])
  for segment in segments:
    sent_map.append(current)
    for i in range(len(segment) - 2):
      sent_map.append(current)
      current += int(sentence_end[sent_end_idx])
      sent_end_idx += 1
    sent_map.append(current)
  return sent_map


def flatten(l):
  return [item for sublist in l for item in sublist]


if __name__ == '__main__':
  args = parser.parse_args()
  cache_dir = '/home/yuxintong/tools/transformers'

  if args.model == 'bert':
    from transformers import AutoTokenizer
    if args.cased:
      tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", 
                                                cache_dir=cache_dir)
    else:
      tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", 
                                                cache_dir=cache_dir)
  else:
    raise ValueError(f'undefined model {args.model}')
  if args.dataset == 'vispro':
    datasets = [f'{split}.vispro.1.1.jsonlines' for split in ['train', 'val', 'test']]
    cap_np = [json.loads(line) for line in open(f'data/cap_np.vispro.1.1.{args.model}.jsonlines')]
    cap_np = {c['doc_key']: c for c in cap_np}
  elif args.dataset == 'conll':
    datasets = [f'{split}.english.middle.pronoun.jsonlines' for split in ['train', 'dev', 'test']]
  elif args.dataset == 'vispro.1.0':
    datasets = [f'{split}.vispro.1.0.jsonlines' for split in ['train', 'val', 'test']]
  elif args.dataset == 'visprp':
    datasets = [f'{split}.visprp.1.0.jsonlines' for split in ['train', 'val', 'test']]
  elif args.dataset == 'medical':
    datasets = [f'{split}.medical.pronoun.jsonlines' for split in ['train', 'test']]
  elif args.dataset == 'mscoco':
    datasets = ['mscoco_label.jsonlines']
  elif args.dataset == 'neg':
    datasets = ['neg_np.vispro.1.1.jsonlines']
  elif args.dataset == 'cap':
    datasets = ['cap_np.vispro.1.1.jsonlines']
  elif args.dataset == 'nn':
    datasets = ['cap_ant.nn.vispro.1.1.jsonlines']
  elif args.dataset == 'vispro.pool':
    datasets = [f'{split}.vispro.1.1.jsonlines' for split in ['train', 'val', 'test']]
    # load np2nn and nn2id
    np2nn_lines = [json.loads(line) for line in open('data/cap_ant.np2nn.vispro.1.1.jsonlines')]
    np2nn = dict()
    for np2nn_line in np2nn_lines:
      NP = list(np2nn_line.keys())[0]
      np2nn[NP] = np2nn_line[NP]
    nn2id_lines = [json.loads(line) for line in open('data/cap_ant.nn.vispro.1.1.jsonlines')]
    nn2id = dict()
    for nn2id_line in nn2id_lines:
      nn2id[' '.join(nn2id_line['sentences'][0])] = int(nn2id_line['doc_key'])
  else:
    raise ValueError(f'Unknown dataset type: {args.dataset}')


  # load data
  for dataset in datasets:
    max_tokenized_seg_len = 0
    input_filename = osp.join('data', dataset)
    data = [json.loads(line) for line in open(input_filename)]
    if args.dataset in ['vispro', 'conll', 'medical', 'vispro.1.0', 'visprp']:
      output_filename = input_filename.replace('.jsonlines', f'.{args.model}.{args.max_seg_len}.jsonlines')
    elif args.dataset == 'vispro.pool':
      output_filename = input_filename.replace('.jsonlines', f'.{args.model}.{args.max_seg_len}.jsonlines').replace('vispro', 'vispro.pool')
    else:
      output_filename = input_filename.replace('.jsonlines', f'.{args.model}.jsonlines')
    if args.cased:
      output_filename = output_filename.replace('.jsonlines', '.cased.jsonlines')
    output_file = open(output_filename, 'w')

    for dialog in tqdm(data):
      sents = dialog['sentences']
      if args.dataset in ['vispro', 'vispro.pool']:
        # exclude caption
        caption_len = len(sents[0])
        caption = sents[0]
        sents = sents[1:]
        speakers = dialog['speakers'][1:]
      elif args.dataset in ['conll', 'medical', 'vispro.1.0', 'visprp']:
        speakers = dialog['speakers']
      else:
        speakers = [['caption'] * len(sents[0])]

      word_idx = -1
      subtoken_idx = 0
      sentence_end = []
      token_end = []
      speakers_subtoken = []
      subtokens_sents = []
      subtoken_map = []
      word_to_subtoken = {}
      # tokenization, get segments, sentence_map, subtoken_map, speakers
      for sent_id, sent in enumerate(sents):
        for word_id, word in enumerate(sent):
          word_idx += 1
          subtokens = tokenizer.tokenize(word)
          num_subtokens = len(subtokens)
          subtokens_sents.extend(subtokens)
          sentence_end += [False] * num_subtokens
          subtoken_map += [word_idx] * num_subtokens
          word_to_subtoken[word_idx] = [subtoken_idx, subtoken_idx + num_subtokens - 1]
          subtoken_idx += num_subtokens
          token_end += ([False] * (num_subtokens - 1)) + [True]
          speakers_subtoken += [speakers[sent_id][word_id]] * num_subtokens
        if len(sentence_end) > 0:
          sentence_end[-1] = True

      # split into segment
      segments = []
      segment_subtoken_map = []
      segment_speakers = []
      current = 0
      previous_tokens = 0
      added_word_to_subtoken = {}
      added_subtoken_count = 1
      while current < len(subtokens_sents):
        end = min(current + args.max_seg_len - 1 - 2, len(subtokens_sents) - 1)
        while end >= current and not sentence_end[end]:
          end -= 1
        if end < current:
          end = min(current + args.max_seg_len - 1 - 2, len(subtokens_sents) - 1)
          while end >= current and not token_end[end]:
            end -= 1
          if end < current:
            raise Exception("Can't find valid segment")
        segments.append(['[CLS]'] + subtokens_sents[current:end + 1] + ['[SEP]'])
        segment_subtoken_map.append([previous_tokens] + subtoken_map[current:end + 1] + [subtoken_map[end]])
        segment_speakers.append(['[SPL]'] + speakers_subtoken[current:end + 1] + ['[SPL]'])
        added_word_to_subtoken[added_subtoken_count] = range(subtoken_map[current], subtoken_map[end] + 1)
        added_subtoken_count += 2
        previous_tokens = subtoken_map[end]
        current = end + 1
        # if current != len(subtokens_sents):
        #   print(f'{dialog["doc_key"]} has separated segments.')
      max_tokenized_seg_len = max(max_tokenized_seg_len, max([len(s) for s in segments]))
      prev_subtoken_count = sum([len(s) for s in segments])

      if args.dataset in ['vispro', 'conll', 'medical', 'vispro.pool', 'vispro.1.0'] or (args.dataset == 'visprp' and 'clusters' in dialog):
        # convert old mention index to new one
        caption_NPs = {'doc_key':[], 'sentences':[]}
        mentions_old = set()
        old_index_to_new = {}
        for cluster in dialog['clusters']:
          for mention in cluster:
            mentions_old.add(tuple(mention))
        for pronoun_info in dialog["pronoun_info"]:
          mentions_old.add(tuple(pronoun_info['current_pronoun']))
          for mention in pronoun_info['candidate_NPs']:
            mentions_old.add(tuple(mention))
        for mention in mentions_old:
          if args.dataset == 'vispro':
            if mention[0] >= caption_len:
              added_subtoken = 0
              for added, word_idx in added_word_to_subtoken.items():
                if mention[0] - caption_len in word_idx:
                  added_subtoken = added
              new_index = [word_to_subtoken[mention[0] - caption_len][0] + added_subtoken, word_to_subtoken[mention[1] - caption_len][1] + added_subtoken]
            else:
              # find np in cap_np.jsonlines
              np = ' '.join(caption[mention[0]:mention[1] + 1]).lower()
              if np not in cap_np:
                raise ValueError(f"{np} of {dialog['doc_key']} not in caption list")
              len_np_seg = len(cap_np[np]["sentences"][0])
              new_index = [prev_subtoken_count + 1, prev_subtoken_count + len_np_seg - 2]
              prev_subtoken_count += len_np_seg
              caption_NPs['doc_key'].append(np)
              caption_NPs['sentences'].append(cap_np[np]["sentences"][0])
          elif args.dataset == 'vispro.pool':
            if mention[0] >= caption_len:
              added_subtoken = 0
              for added, word_idx in added_word_to_subtoken.items():
                if mention[0] - caption_len in word_idx:
                  added_subtoken = added
              new_index = [word_to_subtoken[mention[0] - caption_len][0] + added_subtoken, word_to_subtoken[mention[1] - caption_len][1] + added_subtoken]
            else:
              # replace np with nn ids
              np = ' '.join(caption[mention[0]:mention[1] + 1]).lower()
              if np not in np2nn:
                # np in caption but not in clusters containing pronouns
                continue
              np2nn_cur = np2nn[np]
              new_index = {'nn': nn2id[np2nn_cur['nn']], 'synonym':[], 'hypernym':[], 'hyponym':[]}
              for key in ['synonym', 'hypernym', 'hyponym']:
                for nn in np2nn_cur[key]:
                  new_index[key].append(nn2id[nn])
          elif args.dataset in ['conll', 'medical', 'vispro.1.0', 'visprp']:
            added_subtoken = 0
            for added, word_idx in added_word_to_subtoken.items():
              if mention[0] in word_idx:
                added_subtoken = added
            new_index = [word_to_subtoken[mention[0]][0] + added_subtoken, word_to_subtoken[mention[1]][1] + added_subtoken]

          old_index_to_new[tuple(mention)] = new_index

        # deal with clusters
        # only include clusters with size larger than 1
        clusters_segments = []
        for cluster in dialog["clusters"]:
          cluster_subtokens = []
          for mention in cluster:
            if tuple(mention) in old_index_to_new:
              cluster_subtokens.append(old_index_to_new[tuple(mention)])
            else:
              if args.dataset == 'vispro.pool' and mention[0] < caption_len:
                # check if the cluster do not contain any pronoun
                prp_list = [p['current_pronoun'] for p in dialog['pronoun_info']]
                find_prp = False
                for m in cluster:
                  if m  in prp_list:
                    find_prp = True
                    break
                if not find_prp:
                  continue
              raise ValueError(f'doc_key: {dialog["doc_key"]} mention: {mention}')
          if len(cluster_subtokens) > 1:
            clusters_segments.append(cluster_subtokens)

        # deal with pronoun_info
        pronoun_info_new = []
        for pronoun_info in dialog["pronoun_info"]:
          prp_new_cur = {}
          prp_new_cur['current_pronoun'] = old_index_to_new[tuple(pronoun_info['current_pronoun'])]
          if args.dataset == 'vispro.pool':
            prp_new_cur['candidate_NPs'] = [old_index_to_new[tuple(p)] for p in pronoun_info['candidate_NPs'] if p[0] >= caption_len]
          else:
            prp_new_cur['candidate_NPs'] = [old_index_to_new[tuple(p)] for p in pronoun_info['candidate_NPs']]
          prp_new_cur['correct_NPs'] = [old_index_to_new[tuple(p)] for p in pronoun_info['correct_NPs']]
          if args.dataset in ['vispro', 'vispro.pool', 'vispro.1.0', 'visprp']:
            prp_new_cur['reference_type'] = pronoun_info['reference_type']
            prp_new_cur['coreference_in_cap_only'] = pronoun_info['coreference_in_cap_only']
          pronoun_info_new.append(prp_new_cur)

      # get sentence map
      sentence_map = get_sentence_map(segments, sentence_end)

      dialog_output = {
        "doc_key": dialog["doc_key"],
        "sentences": segments,
        "speakers": segment_speakers,
        "sentence_map": sentence_map,
        "subtoken_map": flatten(segment_subtoken_map),
      }

      if args.dataset in ['vispro', 'conll', 'medical', 'vispro.pool', 'vispro.1.0'] or (args.dataset == 'visprp' and 'clusters' in dialog):
        dialog_output["clusters"] = clusters_segments
        dialog_output["original_sentences"] = dialog["sentences"]
        dialog_output["original_pronoun_info"] = dialog["pronoun_info"]
        dialog_output["pronoun_info"] = pronoun_info_new
        if args.dataset == 'vispro':
          dialog_output["caption_NPs"] = caption_NPs
      if args.dataset in ['vispro', 'vispro.pool', 'vispro.1.0', 'visprp']:
        dialog_output["image_file"] = dialog["image_file"]
        dialog_output["original_sentences"] = dialog["sentences"]


      output_file.write(json.dumps(dialog_output) + '\n')

    output_file.close()
    print(f'Output saved to {output_filename}. Max segment length is {max_tokenized_seg_len}.')
