# Documentation
from typing import List

#Export
from collections import Counter
import json

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
texts = texts.sort_values("tokens")
#texts.drop(labels=["Auteur réel si différent", "Source attrib (lien, doi, bibl)", "Auteur", "Source"], axis=1,
#           inplace=True)

#texts = texts[texts.Mots > 1000]


#tokeniser = AutoTokenizer.from_pretrained("pranaydeeps/Ancient-Greek-BERT")
#model = AutoModel.from_pretrained("pranaydeeps/Ancient-Greek-BERT")  
tagger = SequenceTagger.load('final-model.pt')


def get_poses(sentence) -> List[str]:
    tagger.predict(sentence)
    return [(e.text, e.get_label("pos", zero_tag_value="-").value) for e in sentence.tokens]


def get_text_poses(text, last=False) -> List[str]:
    sentences = [Sentence(s.strip()) for s in SENTENCE_SPLITTER.split(text) if s.strip()]
    out = []
    for sentence in sentences:
        out.extend(get_poses(sentence))
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
    pos_text = get_text_poses(text["text"])
    #print(text["full-text-raw"].count(" "), print(text["tokens"]))
    with open(f"./tagged/{text['file']}-tagged.txt", "w") as f:
        f.write("\n".join([
            "\t".join(tok) for tok in pos_text
        ]))
    total.update(Counter(tok[0] for tok in pos_text if tok[1][0] != "u"))
