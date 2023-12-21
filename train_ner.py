from flair.embeddings import *
from typing import List
import os



glove_roberta : List[TokenEmbeddings] = [
          WordEmbeddings('glove'),
          TransformerWordEmbeddings('roberta-base')
        ]


from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

data_folder = 'PATH_TO_FILE' # initializing the corpus

def load_corpus(train_filename, dev_filename, test_filename):
  # define columns
  columns = {0 : 'text', 1 : 'ner'} # directory where the data resides
  
  # encoding can also be cp1252
  corpus: Corpus = ColumnCorpus(data_folder, 
                                columns,
                                train_file = train_filename,
                                test_file = test_filename,
                                dev_file = dev_filename,
                                encoding = 'utf-8')
  
  return corpus


# corpus_aug.obtain_statistics(label_type=tag_type, pretty_print=False)
def train_ner(prefix_name, corpus):
  # tag to predict
  tag_type = 'ner'

  # make tag dictionary from the corpus
  tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

  folder_embedding = {
    prefix_name+'-glove_roberta' : glove_roberta,
  }

  for folder, embedding_types in folder_embedding.items():
    print(folder)

    if os.path.exists(data_folder+folder + '/final-model.pt'):
      continue
    
    embeddings : StackedEmbeddings = StackedEmbeddings(
                                   embeddings=embedding_types)


    tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                          embeddings=embeddings,
                                          tag_dictionary=tag_dictionary,
                                          tag_type=tag_type,
                                          use_crf=True)

    trainer = None
    if os.path.exists(data_folder + folder + '/checkpoint.pt'):
      checkpoint = data_folder + folder + '/checkpoint.pt'
      trainer = ModelTrainer.load_checkpoint(checkpoint, corpus)
    else:
      trainer = ModelTrainer(tagger, corpus)


    trainer.train(data_folder+folder,
                  learning_rate=0.1,
                  # train_with_dev=True,
                  mini_batch_size=32,
                  checkpoint=True,
                  max_epochs=200)

# train base
corpus = load_corpus('train_data.txt', 'dev_data.txt', 'test_data.txt')
train_ner('base', corpus)

# train aug
for i in range(5):
  amounts = ['20','50','100']
  augs = ['char','word','lm','wordchar','allaug','mr_sem']
  for amount in amounts:
    for aug in augs:
      corpus = load_corpus('train_{}_{}.txt'.format(amount, aug), 'dev_data.txt', 'test_data.txt')
      train_ner('aug-{}-{}-{}'.format(amount, aug, i), corpus)

print("We reach the end of the script")
