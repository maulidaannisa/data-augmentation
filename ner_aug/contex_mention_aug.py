# this python code to combined the context and mention augmented to generate new sentences variation
import numpy as np
import pickle
from scipy import spatial
from niacin.augment import RandAugment
import nlpaug.augmenter.char as nac
from niacin.text import en
from chv_aug import ChvAug
from drug_aug import DrugAug
from numpy import random
import string
import random
import re
from collections import Counter
from difflib import SequenceMatcher

import argparse,codecs
ap = argparse.ArgumentParser()
ap.add_argument("--input1", required=True, type=str, help="input file of augmented mention data")
ap.add_argument("--input2", required=True, type=str, help="output folder of augmented context data")
ap.add_argument("--output", required=False, type=str, help="number of augmentation")

args = ap.parse_args()

# read conll file by sentence    
def delimited(file, delimiter = '\n', bufsize = 4096):
	buf = ''
	while True:
		newbuf= file.read(bufsize)
		if not newbuf:
			yield buf
			return
		buf +=newbuf
		lines =buf.split(delimiter)
		for line in lines[:-1]:
			yield line
		buf = lines[-1]

def get_category2mentions(sentences, labels):
	mentions = []
	for sent_index, sentence in enumerate(sentences):
		mention = []
		for token_index, token in enumerate(sentence):
			label = labels[sent_index][token_index].strip()
			if label == "O" or label[0] == "B":
				if len(mention) > 0:
					mentions.append(mention)
				mention = []
			if label[0] == "B": mention.append(label[2:])
			if label != "O": mention.append(token)
		if len(mention) > 0:
			mentions.append(mention)

	category2mentions = {}
	for mention in mentions:
		if mention[0] not in category2mentions: category2mentions[mention[0]] = {}
		category2mentions[mention[0]][" ".join(mention[1:])] = 1

	for category in category2mentions.keys():
		mentions = list(category2mentions[category].keys())
		category2mentions[category] = mentions
	return category2mentions

def generate_mentions(sentence, labels):
	mentions = []
	mention = []
	for token_index, token in enumerate(sentence):
		# print(labels[token_index])
		label = labels[token_index].strip()
		if label == "O" or label[0] == "B":
			if len(mention) > 0:
				mentions.append(mention)
			mention = []

		if label[0] == "B": mention.append((label[2:], token_index))
		if label != "O": mention.append((token, token_index))

	if len(mention) > 0:
		mentions.append(mention)

	return mentions


def generate_context_mention_aug_sent(sentence_mention, label_mention, sentence_context, label_context):
    mentions_sent_mention = generate_mentions(sentence_mention, label_mention)
    mentions_sent_context = generate_mentions(sentence_context, label_context)

    if not mentions_sent_mention or not mentions_sent_context:
        print("Mismatch in the number of mentions between sentences.")
        return [(token, label) for token, label in zip(sentence_context, label_context)]

    new_label_context = label_context.copy()  # Create a copy of label_context
    new_sentence_context = sentence_context.copy()  # Create a copy of sentence_context

    adjustment = 0  # To track the index adjustment due to length differences

    for idx, mention in enumerate(mentions_sent_mention):
        if idx >= len(mentions_sent_context):
            print("Mismatch in the number of mentions between sentences.")
            break

        ctx_mention = mentions_sent_context[idx][1:]
        mention_to_replace = [ctx[0] for ctx in ctx_mention]
        mention_to_insert = [m[0] for m in mention[1:]]
        original_index_to_replace = mentions_sent_context[idx][0][1]

        # Adjust start index based on cumulative adjustments
        start_index = original_index_to_replace + adjustment
        print(start_index)

        # Calculate the length difference
        length_diff = len(mention_to_insert) - len(mention_to_replace)

        # Perform the replacement in new_sentence_context
        new_sentence_context[start_index:start_index + len(mention_to_replace)] = mention_to_insert
        new_label_context[start_index:start_index + len(mention_to_replace)] = label_mention[mention[0][1]:mention[0][1] + len(mention_to_insert)]

        if length_diff > 0:
            for i in range(length_diff):
                new_sentence_context.insert(start_index + len(mention_to_replace) + i, mention_to_insert[len(mention_to_replace) + i])
                new_label_context.insert(start_index + len(mention_to_replace) + i, label_mention[mention[0][1] + len(mention_to_replace) + i])
        elif length_diff < 0:
            for i in range(-length_diff):
                del new_sentence_context[start_index + len(mention_to_insert)]
                del new_label_context[start_index + len(mention_to_insert)]

        # Update adjustment based on the length difference
        adjustment += length_diff

    augmented_sentence = [(token, label) for token, label in zip(new_sentence_context, new_label_context)]
    return augmented_sentence




def save_augmentation(aug_data, output_file, insert_original=True):
	writer = open(output_file, 'w', encoding='utf-8')

	# insert augmented data then
	for aug_sentence in aug_data:
		for s in aug_sentence:
			word = s[0]
			label = s[1]

			writer.write(word+" "+label+"\n")
		writer.write("\n")

	writer.close()


def gen_ner_aug(train_mention_aug,train_context_aug, output_folder):
	ann_infile_mention = codecs.open(train_mention_aug,'r')
	aug_result = {"char":[],"word":[],"wordchar":[],"mr":[],"mr_sem":[],"lm":[]}
	sentences_mention, labels_mention = [], []
   
	print("building category2mention")
	lines = delimited(ann_infile_mention,"\n\n",bufsize = 1)
	for i, line in enumerate(lines):
		info = line.rstrip().split("\n")
		if line.rstrip() =="":
			continue

		sent = [a.split()[0] for a in info if len(a.split()) > 1]
		label = [a.split()[1] for a in info if len(a.split()) > 1]

		sentences_mention.append(sent)
		labels_mention.append(label)

	category2mentions_mention_aug = get_category2mentions(sentences_mention, labels_mention)


	sentences_context, labels_context = [], []
	ann_infile_context = codecs.open(train_context_aug,'r')
	print("building category2mention")
	lines = delimited(ann_infile_context,"\n\n",bufsize = 1)
	for i, line in enumerate(lines):
		info = line.rstrip().split("\n")
		if line.rstrip() =="":
			continue

		sent_ctx= [a.split()[0] for a in info if len(a.split()) > 1]
		label_ctx = [a.split()[1] for a in info if len(a.split()) > 1]

		sentences_context.append(sent_ctx)
		labels_context.append(label_ctx)
   
	category2mentions_context_aug = get_category2mentions(sentences_context, labels_context)

	generated_aug_sentences = []
	print(len(sentences_mention))
	print(len(sentences_context))
	for i, sent_ment in enumerate(sentences_mention):
		label_ment = labels_mention[i]
		sent_con = sentences_context[i]
		labels_con = labels_context[i]

		augmented_sentence = generate_context_mention_aug_sent(sent_ment, label_ment, sent_con, labels_con)
		generated_aug_sentences.append(augmented_sentence)
	save_augmentation(generated_aug_sentences[0], output_folder + "train_20_mention_context_char.txt", insert_original=False)


gen_ner_aug(args.input1, args.input2, args.output)

