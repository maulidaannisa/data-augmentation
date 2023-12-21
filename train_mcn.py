from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.embeddings import *
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load corpus for base model (without augmentation)
def load_corpus_base(fold):
  data_folder = 'PATH_TO_DATA'
  corpus: Corpus = ClassificationCorpus(data_folder,
                                        test_file='test_data.txt',
                                        dev_file='dev_data.txt',
                                        train_file='train_data.txt')

  return corpus

# load corpus for augmentation model
# receive one parameter coresponding to augmentation method
def load_corpus(aug):
  data_folder = 'PATH_TO_DATA'
  corpus: Corpus = ClassificationCorpus(data_folder,
                                        test_file='test_data.txt',
                                        train_file='train_data_{}.txt'.format(aug))

  return corpus

# load corpus for augmentation model
# receive one parameter coresponding to augmentation method
def train_data(folder, corpus):
  data_folder = 'PATH_TO_DATA/' + folder + '/'

  # 2. create the label dictionary
  label_dict = corpus.make_label_dictionary()

  # 3. make a list of word embeddings
  word_embeddings = {
      # 'flair' : [WordEmbeddings('glove'),FlairEmbeddings('news-forward'),FlairEmbeddings('news-backward')],
      # 'roberta' : [TransformerWordEmbeddings('roberta-base')],
      'glove-roberta' : [WordEmbeddings('glove'),TransformerWordEmbeddings('roberta-base')],
      # 'bert' : [TransformerWordEmbeddings('bert-base-uncased')],
      # 'hunflair' : [WordEmbeddings("pubmed"), FlairEmbeddings("pubmed-forward"), FlairEmbeddings("pubmed-backward")],
  }
  
  for folder_name, word_embedding in word_embeddings.items():
    if os.path.exists(data_folder+folder + '/final-model.pt'):
      continue

    print(folder, folder_name)

    if os.path.exists(data_folder + folder_name + '/final-model.pt'):
      continue

    document_embeddings = DocumentRNNEmbeddings(word_embedding, hidden_size=256)

    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

    trainer = None
    if os.path.exists(data_folder + folder_name + '/checkpoint.pt'):
      checkpoint = data_folder + folder_name + '/checkpoint.pt'
      trainer = ModelTrainer.load_checkpoint(checkpoint, corpus)
    else:
      trainer = ModelTrainer(classifier, corpus)
    
    trainer.train(data_folder + folder_name,
                  learning_rate=0.1,
                  mini_batch_size=32,
                  anneal_factor=0.5,
                  checkpoint=True,
                  patience=5,
                  max_epochs=150)

# run base model
for i in range(5):
   train_data("base-run-{}".format(i), load_corpus_base(i))

# run augmentation model
augs = ['char','word','lm','wordchar','allaug']
for aug in augs:
  for i in range(5):
    train_data("aug-{}-run-{}".format(aug, i), load_corpus(aug))

