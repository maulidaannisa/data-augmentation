import numpy as np
import pickle
from scipy import spatial
from niacin.augment import RandAugment
import nlpaug.augmenter.char as nac
from niacin.text import en

from chv_aug import ChvAug
chv_aug = ChvAug()

from drug_aug import DrugAug
drug_aug = DrugAug()

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

def load_pickle(category):
    pickle_folder = "PATH_TO_PICKLE/"

    if category == 'disease' or category=='dis':
        cadec_disease = pickle.load( open( pickle_folder + "cadec_disease.pickle", "rb" ) )
        medred_disease = pickle.load( open( pickle_folder + "medred_disease.pickle", "rb" ) )

        cadec_disease.update(medred_disease)

        return cadec_disease
    elif category== 'drug':
        cadec_drug = pickle.load( open( pickle_folder + "cadec_drug.pickle", "rb" ) )
        medred_drug = pickle.load( open( pickle_folder + "medred_drug.pickle", "rb" ) )

        cadec_drug.update(medred_drug)

        return cadec_drug
    else:
        print(category)
        return {}
        # raise ValueError("category is not defined...")

def select_similar_mention(category, phrase, num_aug):
    dict_mentions = load_pickle(category)
    if not dict_mentions:
        return [phrase for i in range(num_aug)]

    sim_dict = {}
    for mention, val in dict_mentions.items():
        if mention.lower() == phrase.lower():
            continue
            
        # Note that spatial.distance.cosine computes the distance, and not the similarity. So, you must subtract the value from 1 to get the similarity.
        sim = 1 - spatial.distance.cosine(val, dict_mentions[phrase])
        sim_dict[mention] = sim
        
    sort_orders = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    mentions = []
    for i, m in enumerate(sort_orders):
        if i == num_aug:
            break
        mentions.append(m[0])

    return mentions

def generate_mentions(sentence, labels):
    mentions = []
    mention = []

    for token_index, token in enumerate(sentence):
        label = labels[token_index].strip()
        if label == "O" or label[0] == "B":
            if len(mention) > 0:
                mentions.append(mention)
            mention = []

        if label[0] == "B": mention.append(label[2:])
        if label != "O": mention.append(token)

    if len(mention) > 0:
        mentions.append(mention)

    return mentions

def generate_aug_sent(sentence, labels, aug_list, num_aug):
  # print(aug_list)
  generated_aug_sentences = []
  for i in range(num_aug):
        generated_sentence = []
        idx_aug = 0        

        for token_index, token in enumerate(sentence):
            label = labels[token_index].strip()
            if label == "O":
                generated_sentence.append((token, 'O'))
            elif label[0] == "B":
                if i >= len(aug_list[idx_aug]):
                    continue

                category = label[2:]

                replaced_mention = aug_list[idx_aug][i].split()
                if len(replaced_mention) == 0:
                    continue
                    
                generated_sentence.append((replaced_mention[0], "B-%s" % category))

                idx_aug += 1

                for t in replaced_mention[1:]:
                    generated_sentence.append((t, "I-%s" % category))

            elif label[0] == "I":
                continue
            else:
                print(token, label)
                raise ValueError("unreachable line...")

        generated_aug_sentences.append(generated_sentence)
        # add this to only insert one augmented text when augmented the surrounding augmented text
        break

  return generated_aug_sentences

def mention_replacement(sentence, labels, category2mentions, num_aug=3):
    generated_aug_sentences = []

    for i in range(num_aug):
        generated_sentence = []

        for token_index, token in enumerate(sentence):
            label = labels[token_index].strip()
            if label == "O":
                generated_sentence.append((token, 'O'))
            elif label[0] == "B":
                category = label[2:]
                candidates = category2mentions[category]
                random_idx = np.random.choice(len(candidates), 1)[0]
                replaced_mention = candidates[random_idx].split()

                generated_sentence.append((replaced_mention[0], "B-%s" % category))

                for t in replaced_mention[1:]:
                    generated_sentence.append((t, "I-%s" % category))

            elif label[0] == "I":
                continue
            else:
                print(token, label)
                raise ValueError("unreachable line...")

        generated_aug_sentences.append(generated_sentence)

    return generated_aug_sentences

def mention_replacement_semantic(sentence, labels, num_aug=3):
    mentions = generate_mentions(sentence, labels)

    aug_list = []

    for m in mentions:
        label = m[0]
        phrase = ' '.join(m[1:])
        mentions_aug = select_similar_mention(label.lower(), phrase, num_aug)

        aug_list.append(mentions_aug)

    generated_aug_sentences = generate_aug_sent(sentence, labels, aug_list, num_aug)

    return generated_aug_sentences

def chv_replacement(sentence, labels):
    mentions = generate_mentions(sentence, labels)

    aug_list = []

    for m in mentions:
        label = m[0]
        phrase = ' '.join(m[1:])
        text = ""
        if label.lower() == 'drug':
            text = drug_aug.augment(phrase.lower())
        else:
            text = chv_aug.augment(phrase.lower())

        aug_list.append(text)

    generated_sentence = []
    idx_aug = 0
    for token_index, token in enumerate(sentence):
        label = labels[token_index].strip()
        if label == "O":
            generated_sentence.append((token, 'O'))
        elif label[0] == "B":
            category = label[2:]
            replaced_mention = aug_list[idx_aug].split()
            generated_sentence.append((replaced_mention[0], "B-%s" % category))

            idx_aug += 1

            for t in replaced_mention[1:]:
                generated_sentence.append((t, "I-%s" % category))

        elif label[0] == "I":
            continue
        else:
            print(token, label)
            raise ValueError("unreachable line...")

    return [generated_sentence]

def char_aug(sentence, labels):
  mentions = generate_mentions(sentence, labels)
  aug_list = []

  # we have 3 different character augmentation
  num_aug = 3
  ocr_aug = nac.OcrAug()
  augmentor = RandAugment([
      en.add_misspelling,
      en.add_fat_thumbs,
  ], n=2, m=10, shuffle=False)

  for m in mentions:
    augmented_texts = []

    label = m[0]
    phrase = ' '.join(m[1:]).lower()

    augmented_text = ocr_aug.augment(phrase)
    if augmented_text:
        augmented_texts.append(augmented_text[0])
    else:
        augmented_texts.append('')

    for tx in augmentor:
        augmented_text = tx(phrase)
        augmented_texts.append(augmented_text)
    print(augmented_texts)
    aug_list.append(augmented_texts)
  
  generated_aug_sentences = generate_aug_sent(sentence, labels, aug_list, num_aug)

  return generated_aug_sentences

def word_aug(sentence, labels):
  mentions = generate_mentions(sentence, labels)
  aug_list = []

  # we have 4 different word augmentation
  num_aug = 3
  augmentor = RandAugment([
      en.add_synonyms,
      en.add_hyponyms,
      # en.swap_words,
      en.add_hypernyms,
  ], n=3, m=30, shuffle=False)

  for m in mentions:
    augmented_texts = []

    label = m[0]
    phrase = ' '.join(m[1:]).lower()
    
    augmented_text = ""
    if label.lower() == 'drug':
        augmented_text = drug_aug.augment(phrase)
    else:
        augmented_text = chv_aug.augment(phrase)

    augmented_texts.append(augmented_text)

    for tx in augmentor:
        augmented_text = tx(phrase)
        augmented_texts.append(augmented_text)

    aug_list.append(augmented_texts)
    # print(aug_list)
  
  generated_aug_sentences = generate_aug_sent(sentence, labels, aug_list, num_aug)

  return generated_aug_sentences

def word_char_aug(sentence, labels):
    generated_aug_sentences = char_aug(sentence, labels)
    generated_aug_sentences.extend(word_aug(sentence, labels))

    return generated_aug_sentences

def lm_aug(sentence, labels, lm_dict, num_aug=3):
  mentions = generate_mentions(sentence, labels)
  aug_list = []

  for m in mentions:

    label = m[0]
    phrase = ' '.join(m[1:]).lower()
    augmented_texts = lm_dict[phrase.strip()]

    aug_list.append(augmented_texts)

  # our lm generates maximum 5 sentence
  generated_aug_sentences = generate_aug_sent(sentence, labels, aug_list, num_aug)

  return generated_aug_sentences




