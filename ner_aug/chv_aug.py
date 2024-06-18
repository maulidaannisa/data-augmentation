import csv
import random
from nltk import ngrams
import re

class ChvAug():
    def __init__(self):
        self.chv_map = {}
        self.__build_chv_map()

    def __build_chv_map(self):
        res_folder = "/Users/annisaningtyas/Documents/GitHub/data-augmentation/resource/"

        # format 1
        filepath = res_folder + "CHV-amia14-data.tsv"
        with open(filepath, encoding='utf-8') as csvfile:
            chv = csv.reader(csvfile, delimiter='\t')
            for c in chv:
                consumer = c[-2].strip()
                professional = c[-3].strip()
                
                # provide two way mapping
                self.__map_chv(professional, consumer)
                self.__map_chv(consumer, professional)

        # format 2
        filepath = res_folder + "CHV_concepts_terms_flatfile_20110204.tsv"
        with open(filepath, encoding='utf-8') as csvfile:
            chv = csv.reader(csvfile, delimiter='\t')
            for c in chv:
                consumer = c[1].strip()
                chv_preferred = c[2].strip()
                
                # provide two way mapping
                self.__map_chv(chv_preferred, consumer)
                self.__map_chv(consumer, chv_preferred)
        

    def __map_chv(self, key, value):
        if key not in self.chv_map:
            self.chv_map[key] = [value]
        else:
            self.chv_map[key].append(value)

    def clean_bracket(self, text):
        regex = '\(.*?\)'
        text = re.sub(regex,'',text)
        text = text.replace('(','')
        text = text.replace(')','')

        return text

    def last_clean(self, text):
        single_space = ' '.join(text.split())
        return single_space

    def get_synonyms(self, professional):
        if professional in self.chv_map:
            return self.chv_map[professional]
        else:
            return []

    def augment(self, text, only_once=True):
        augment_map = {}
        tokens = text.split()
        found_syn = False

        # checking will start from 6-grams down to uni(1-)gram
        for n in range(6,0,-1):
            grams = ngrams(tokens, n)

            for gram in grams:
                phrase = " ".join(gram)
                syn = self.get_synonyms(phrase)
                if len(syn) > 0:
                    augment_map[phrase] = self.clean_bracket(random.choice(syn))

                    if only_once:
                        found_syn=True
                        break

                if only_once and found_syn:
                    break

        for phrase, syn in augment_map.items():
            # text = text.replace(phrase, syn)
            text = re.sub(r'\b{}\b'.format(phrase), syn, text)

        text = self.last_clean(text)
        
        return text