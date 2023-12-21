import codecs
from augment_data_conll import *


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

def gen_mentions_for_lm(train_orig, output_file):
    ann_infile = codecs.open(train_orig,'r')
    lines = delimited(ann_infile,"\n\n",bufsize = 1)
    
    # building all data
    sentences = []
    labels = []
    phrases = []
    
    writer = open(output_file, 'w', encoding='utf-8')

    for i, line in enumerate(lines):
        if i% 50 == 0:
            print (i,"lines finished...")

        info = line.rstrip().split("\n")
        
        if line.rstrip() =="":
            continue

        sent = [a.split()[0] for a in info if len(a.split()) > 1]
        label = [a.split()[1] for a in info if len(a.split()) > 1]

        mentions = generate_mentions(sent, label)
        for m in mentions:
            label = m[0]
            phrase = ' '.join(m[1:]).lower().strip()
            if phrase not in phrases:
                writer.write(phrase+"\n")
            phrases.append(phrase)

    writer.close()

gen_mentions_for_lm("dataset/medred/MedRed_train.txt","dataset/medred/MedRed_mentions.txt",)