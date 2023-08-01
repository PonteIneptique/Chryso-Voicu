# Documentation
from typing import List

#Export
from collections import Counter
import json
import os

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

SENTENCE_SPLITTER = re.compile(r"(?<=[!?\.])")
texts = pandas.read_csv(sys.argv[1])
texts = texts[~texts.title.str.contains("Dub\.|Sp\.|Fragm|Excerpt|(e cod\.)|Suda|recensio|fragm|sp\.|dub\.|(fort\. auctore)|Scholia")]
texts = texts.sort_values("tokens")

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

total = Counter()
NGRAMS_DF = []
POS_DF = []



for idx, text in tqdm.tqdm(texts.iterrows()):
    if os.path.exists(f"./tagged/{text['file']}-tagged.txt"):
        print(f"Passing {text['file']}")
        continue
    pos_text = get_text_poses(text["full-text-raw"])
    #print(text["full-text-raw"].count(" "), print(text["tokens"]))
    with open(f"./tagged/{text['file']}-tagged.txt", "w") as f:
        f.write("\n".join([
            "\t".join(tok) for tok in pos_text
        ]))
    total.update(Counter(tok[0] for tok in pos_text if tok[1][0] != "u"))