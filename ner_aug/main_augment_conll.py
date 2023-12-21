from augment_data_conll import *
import json
import argparse,codecs
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=True, type=str, help="output folder of augmented data")
ap.add_argument("--num_aug", required=False, type=str, help="number of augmentation")

args = ap.parse_args()


#number of augmented sentences to generate per original sentence
num_aug = 3 #default
if args.num_aug:
    num_aug = args.num_aug

# how much to change each sentence
# alpha = 0.3 #default
# if args.alpha:
#    alpha = args.alpha

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
    

def save_augmentation(original_data, aug_data, output_file, insert_original=True):
    writer = open(output_file, 'w', encoding='utf-8')

    # insert original data first
    if insert_original:
        sentences = original_data[0]
        labels = original_data[1]
        for i, sent in enumerate(sentences):
            for token_index, token in enumerate(sent):
                label = labels[i][token_index]

                writer.write(token+" "+label+"\n")
            writer.write("\n")

    # insert augmented data then
    for aug_sentences in aug_data:
        for aug_sentence in aug_sentences: 
            for s in aug_sentence:
                word = s[0]
                label = s[1]

                writer.write(word+" "+label+"\n")
            writer.write("\n")

    writer.close()

def gen_ner_aug(train_orig, output_folder):
    ann_infile = codecs.open(train_orig,'r')
    aug_result = {"char":[],"word":[],"wordchar":[],"mr":[],"mr_sem":[],"lm":[]}
    sentences, labels = [], []

    print("building category2mention")
    lines = delimited(ann_infile,"\n\n",bufsize = 1)
    for i, line in enumerate(lines):
        info = line.rstrip().split("\n")
        if line.rstrip() =="":
            continue

        sent = [a.split()[0] for a in info if len(a.split()) > 1]
        label = [a.split()[1] for a in info if len(a.split()) > 1]

        sentences.append(sent)
        labels.append(label)

    category2mentions = get_category2mentions(sentences, labels)

    for i, sent in enumerate(sentences):
        if i% 50 == 0:
            print (i,"lines finished...")

        label = labels[i]

        # for lm, prefetch from google colab, due to limited resource in my computer
        # lm_dict = get_lm_dict("dataset/medred/MedRed_mentions.json")
        lm_dict = get_lm_dict("dataset/cadec/data_for_auglm_dict.json")

        aug_result["char"].append(char_aug(sent, label))
        aug_result["word"].append(word_aug(sent, label))
        aug_result["wordchar"].append(word_char_aug(sent, label))
        # aug_result["mr"].append(mention_replacement(sent, label, category2mentions))
        aug_result["mr_sem"].append(mention_replacement_semantic(sent, label))
        aug_result["lm"].append(lm_aug(sent, label, lm_dict))
        
    # all_result = aug_result["char"] + aug_result["word"] + aug_result["wordchar"] + aug_result["mr"] + aug_result["mr_sem"] + aug_result["lm"]
    all_result = aug_result["char"] + aug_result["word"] + aug_result["mr_sem"] + aug_result["lm"]

    original_data = (sentences, labels)
    save_augmentation(original_data, aug_result["char"], output_folder + "train_50_char.txt", insert_original=False)
    save_augmentation(original_data, aug_result["word"], output_folder + "train_50_word.txt", insert_original=False)
    save_augmentation(original_data, aug_result["wordchar"], output_folder + "train_50_wordchar.txt", insert_original=False)
    # save_augmentation(original_data, aug_result["mr"], output_folder + "train_mr.txt", insert_original=False)
    save_augmentation(original_data, aug_result["mr_sem"], output_folder + "train_50_mr_sem.txt", insert_original=False)
    save_augmentation(original_data, aug_result["lm"], output_folder + "train_50_lm.txt", insert_original=False)
    save_augmentation(original_data, all_result, output_folder + "train_50_allaug.txt", insert_original=False)



def get_lm_dict(DICT_PATH):
    with open(DICT_PATH) as f:
        return json.load(f)

#main function
if __name__ == "__main__":
    import time 
    before = time.time()

    gen_ner_aug(args.input, args.output)

    cost = time.time()-before
    print (cost)
