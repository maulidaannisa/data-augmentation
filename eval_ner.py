from flair.models import SequenceTagger
from flair.data import Sentence
import codecs
import re
from nervaluate import Evaluator
import os
import json

mode = "medred"

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

def conll_to_sentence(input_file):
  ann_infile = codecs.open(input_file,'r')
  lines = delimited(ann_infile,"\n\n",bufsize = 1)

  sentences = []
  labels = []
  for i, line in enumerate(lines):
    #parts = line[:-1].split('\t')
    #label = parts[0]
    if i% 50 == 0:
      print (i,"lines finished...")
    info = line.rstrip().split("\n")

    if line.rstrip() =="":
      continue

    # print(info)

    # cadec
    if mode == "cadec":
      sent_labels = " ".join(info)
      words_tags = re.split(r'[ ]', sent_labels)
    elif mode == "medred":
      words_tags = info


    words = []
    tags = []
    for w in words_tags:
      if w.strip() == "":
        continue

      # print(w)
      # cadec
      if mode == "cadec":
        words.append(w.split('\t')[0])
        tags.append(w.split('\t')[1])
      elif mode == "medred":
        words.append(w.split()[0])
        tags.append(w.split()[1])

    sentence = ''
    for w in words:
      # if w=='.':
      #   sentence = sentence+w
      #   continue
      sentence = sentence +' '+ w
    label  =[' '.join(tags)]
    sentences.append(sentence.lstrip())
    labels.append(label)

  return sentences, labels

def ner_evaluation(model_file, test_file):

  model = SequenceTagger.load(model_file)
  # model.max_subtokens_sequence_length = 512
  # model.stride = 512 // 2 #or

  sentences, labels= conll_to_sentence(test_file)

  true = []
  pred = []

  for i, sentence in enumerate (sentences):
    # print(sentence)
    true_label = labels[i][0].split()
    # print(true_label)
    true.append(true_label)
    sent_spans = [(ele.start(), ele.end() - 1) for ele in re.finditer(r'\S+', sentence)]
    
    sent = Sentence(sentence)
    model.predict(sent)
    prediction =sent.to_dict(tag_type='ner')
    ent_predic = prediction.get('entities')
    # print(ent_predic)
    
    predict_label = []

    for i, span in enumerate(sent_spans):
      predict_label.append("O")
      for ent in ent_predic:
        if ent['start_pos'] == span[0]:
           predict_label[i] = "B-" +  ent['labels'][0].value
           break
        elif ent['start_pos'] <= span[0] <= ent['end_pos']:
           predict_label[i] = "I-" +  ent['labels'][0].value
           break
    # print(predict_label)
    
    pred.append(predict_label)

    
  # cadec
  if mode == "cadec":
    evaluator = Evaluator(true, pred, tags=['Disease', 'Drug'], loader="list")
  elif mode == "medred":
    evaluator = Evaluator(true, pred, tags=['DIS', 'DRUG'], loader="list")
  

  results, results_by_tag = evaluator.evaluate()
  
  return results, results_by_tag

if mode == "cadec":
  model_folder = "data_ner_medical/cadec/"
  test_file = "data_ner_medical/cadec/test_data.txt"
elif mode == "medred":
  model_folder = "data_ner_medical/medred/"
  test_file = "data_ner_medical/medred/MedRed_test.txt"

for x in os.walk(model_folder):
  if 'test.tsv' in x[2]:
    print(x[0])
    model_file = x[2]
    name = x[0].split('/')[-1]

    results, results_by_tag = None, None

    model_pt = None

    if os.path.isfile(x[0] + "/best-model.pt"):
      model_pt = x[0] + "/best-model.pt"
    elif os.path.isfile(x[0] + "/final-model.pt"):
      model_pt = x[0] + "/final-model.pt"

    if model_pt:
      #try:
      results, results_by_tag = ner_evaluation(model_pt,test_file)
      #except:
      #print(x[0],"Error")
      #continue

      out_file_1 = open(name + ".json", "w")
      out_file_2 = open(name + "_bytag.json", "w")

      json.dump(results, out_file_1, indent = 4)
      json.dump(results_by_tag, out_file_2, indent = 4)
      
      out_file_1.close()
      out_file_2.close()

