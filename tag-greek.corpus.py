# Documentation
from typing import List

#Export
from collections import Counter

# Dealing with our CSV
import pandas
import re

# Dealing with lemmatization
import flair
import tqdm
import torch
flair.device = torch.device('cuda:0')  # switch to "cpu" if you need

from flair.models import SequenceTagger
from flair.data import Sentence


#Load CSV
import sys

SAMPLE_SIZE = 3000 
SENTENCE_SPLITTER = re.compile(r"(?<=[!?\.])")
print(sys.argv[1])
texts = pandas.read_csv(sys.argv[1])
texts = texts[~texts.title.str.contains("Dub\.|Sp\.|Fragm|Excerpt|(e cod\.)|Suda|recensio")]
#texts.drop(labels=["Auteur réel si différent", "Source attrib (lien, doi, bibl)", "Auteur", "Source"], axis=1,
#           inplace=True)

#texts = texts[texts.Mots > 1000]


#tokeniser = AutoTokenizer.from_pretrained("pranaydeeps/Ancient-Greek-BERT")
#model = AutoModel.from_pretrained("pranaydeeps/Ancient-Greek-BERT")  
tagger = SequenceTagger.load('final-model.pt')


def get_poses(sentence) -> List[str]:
    tagger.predict(sentence)
    return [e.value[0] for e in sentence.get_labels("pos")]


def get_text_poses(text, last=False) -> List[str]:
    sentences = [Sentence(s.strip()) for s in SENTENCE_SPLITTER.split(text) if s.strip()]
    #if last:
    #    sentences = sentences[::-1]
    out = []
    for sentence in sentences:
        out.extend(get_poses(sentence))
        #if len(filter_punct(out)) > SAMPLE_SIZE:
        #    break 
    #if last:
    #    sentences[::-1]
    return out


def get_3grams(poses, N=3) -> Counter:
    """ Transform a list of POS into a list of 3-grams
    """
    return Counter(["-".join(poses[i:i+N]) for i in range(len(poses)-N+1)])


def filter_punct(poses):
    return [p for p in poses if p[0] != "u"]


total = Counter()
NGRAMS_DF = []
POS_DF = []


for idx, text in tqdm.tqdm(texts.iterrows()):
    pos_text = list(filter_punct(get_text_poses(text["full-text-raw"])))[text["start"]:text["end"]]
    #     if text["first-3k"]:
    #         pos_text = pos_text[:SAMPLE_SIZE]
    #     else:
    #         pos_text = pos_text[-SAMPLE_SIZE:]
    ngrams = get_3grams(pos_text)
    total += ngrams
    NGRAMS_DF.append({**text.to_dict(), **ngrams, "full-pos-text": "".join(pos_text)})
    #POS_DF.append({**text.to_dict(), "POSes": "".join(pos_text)})


NGRAMS_DF = pandas.DataFrame(NGRAMS_DF)
#NGRAMS_DF.drop(labels="text", axis=1, inplace=True)


#POS_DF = pandas.DataFrame(POS_DF)
#POS_DF.drop(labels="text", axis=1, inplace=True)


NGRAMS_DF.to_csv(sys.argv[1]+".POS-3grams.csv", index=False)
