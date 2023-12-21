from niacin.augment import RandAugment
import nlpaug.augmenter.char as nac
from niacin.text import en
import random
from aug_chv import ChvAug
from aug_drug import DrugAug

chv_aug = ChvAug()
drug_aug = DrugAug()

def get_label_text(line):
  label = line.split("\t")[0]
  original_text = line.split("\t")[1].strip()

  return label, original_text

def get_label_text_fasttext(line):
  label = line.split()[0].split("__label__")[-1]
  original_text = ' '.join(line.split()[1:])

  return label, original_text

def export_result(input_original, output_path, all_lines):
    with open(input_original, 'r') as train_file:
      for line in train_file:
        # label, original_text = get_label_text(line)
        label, original_text = get_label_text_fasttext(line)

        all_lines.append("__label__{} {}".format(label, original_text))

    # random.shuffle(all_lines)

    with open(output_path, 'w', encoding='utf-8') as combine_file:
      for line in all_lines:
        combine_file.write(line + "\n")

def char_aug(input_path):
  all_lines = []

  ocr_aug = nac.OcrAug()

  augmentor = RandAugment([
      en.add_misspelling,
      en.add_fat_thumbs,
  ], n=2, m=30, shuffle=False)

  with open(input_path, 'r') as txt_file:
    for line in txt_file:
      # label, original_text = get_label_text(line)
      label, original_text = get_label_text_fasttext(line)

      # ditambah dengan ocr aug
      augmented_text = ocr_aug.augment(original_text)
      all_lines.append("__label__{} {}".format(label, augmented_text))
      

      for tx in augmentor:
        augmented_text = tx(original_text)
        all_lines.append("__label__{} {}".format(label, augmented_text))

  return all_lines



def word_aug(input_path):
  all_lines = []

  augmentor = RandAugment([
      en.add_synonyms,
      en.add_hyponyms,
      en.swap_words,
      en.add_hypernyms,
  ], n=4, m=30, shuffle=False)


  with open(input_path, 'r') as txt_file:
    for line in txt_file:
      # label, original_text = get_label_text(line)
      label, original_text = get_label_text_fasttext(line)
      current_aug = []

      for tx in augmentor:
        augmented_text = tx(original_text)
        all_lines.append("__label__{} {}".format(label, augmented_text))
        current_aug.append(augmented_text)

      # ditambah dengan chv drug aug
      augmented_text = drug_aug.augment(original_text)
      augmented_text = chv_aug.augment(augmented_text)
      current_aug.append(augmented_text)

      unique_aug = list(set(current_aug))



      all_lines.append("__label__{} {}".format(label, augmented_text))


  return all_lines

data_folder = 'dataset/psytar/'

# fasttext format
input_path = data_folder + 'train_data.txt'

combine_aug_data = []

word_aug_data = word_aug(input_path)
combine_aug_data.extend(word_aug_data.copy())
export_result(input_path, data_folder + 'train_data_word.txt', word_aug_data)

char_aug_data = char_aug(input_path)
combine_aug_data.extend(char_aug_data.copy())
export_result(input_path, data_folder + 'train_data_char.txt', char_aug_data)

export_result(input_path, data_folder + 'train_data_wordchar.txt', combine_aug_data)


