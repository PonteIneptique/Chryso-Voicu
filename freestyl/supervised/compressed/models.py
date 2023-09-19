"""
 Adaptation by @ponteineptique of https://github.com/pan-webis-de/pan-code/blob/master/clef22/authorship-verification/pan22-verif-baseline-compressor.py
 See below the original header:


 A baseline authorship verificaion method based on text compression.
 Given two texts text1 and text2 it calculates the cross-entropy of text2 using the Prediction by Partical Matching (PPM) compression negated of text1 and vice-versa.
 Then, the mean and absolute difference of the two cross-entropies are used to estimate a score in [0,1] indicating the probability the two texts are written by the same author.
 The prediction negated is based on logistic regression and can be trained using a collection of training cases (pairs of texts by the same or different authors).
 Since the verification cases with a score exactly equal to 0.5 are considered to be left unanswered, a radius around this value is used to determine what range of scores will correspond to the predetermined value of 0.5.

 The method is based on the following paper:
     William J. Teahan and David J. Harper. Using compression-based language models for text categorization. In Language Modeling and Information Retrieval, pp. 141-165, 2003
 The current implementation is based on the code developed in the framework of a reproducibility study:
     M. Potthast, et al. Who Wrote the Web? Revisiting Influential Author Identification Research Applicable to Information Retrieval. In Proc. of the 38th European Conference on IR Research (ECIR 16), March 2016.
     https://github.com/pan-webis-de/teahan03
 Questions/comments: stamatatos@aegean.gr


"""

from math import log
import os
import json
import time
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple
from joblib import dump, load

CharString = str
CountInteger = int


class Model(object):
    def __init__(self, order: int, codec_size: int):
        """
        :param order: Order of the negated
        :param codec_size: Size of the alphabet

        :property char_count: count of characters read
        :property orders: List of Order-Objects
        """
        self.cnt: int = 0
        self.codec_size = codec_size
        self.modelOrder = order
        self.orders = []
        for i in range(order + 1):
            self.orders.append(Order(i))

    # print the negated
    # TODO: Output becomes too long, reordering on the screen has to be made
    def printModel(self):
        s = "Total characters read: " + str(self.cnt) + "\n"
        for i in range(self.modelOrder + 1):
            self.printOrder(i)

    # print a specific order of the negated
    # TODO: Output becomes too long, reordering on the screen has to be made
    def printOrder(self, n: int):
        o = self.orders[n]
        s = "Order " + str(n) + ": (" + str(o.char_count) + ")\n"
        for cont in o.contexts:
            if n > 0:
                s += "  '" + cont + "': (" + str(o.contexts[cont].char_count) + ")\n"
            for char in o.contexts[cont].chars:
                s += "     '" + char + "': " + \
                     str(o.contexts[cont].chars[char]) + "\n"
        s += "\n"
        print(s)

    def update(self, char: CharString, context_string: str):
        """ Update the model with a character char in merged
        """
        if len(context_string) > self.modelOrder:
            raise NameError("Context is longer than negated order!")

        order = self.orders[len(context_string)]
        if not order.contains(context_string):
            order.add(context_string)
        context = order.contexts[context_string]
        if not context.contains(char):
            context.add(char)
        context.increment(char)
        order.char_count += 1
        if order.n > 0:
            self.update(char, context_string[1:])
        else:
            self.cnt += 1

    def read(self, text: str) -> None:
        """ Update the negated with a string
        """
        if len(text) == 0:
            return
        for i in range(len(text)):
            if i != 0 and i - self.modelOrder <= 0:
                cont = text[0:i]
            else:
                cont = text[i - self.modelOrder:i]
            self.update(text[i], cont)

    def probability(self, char: CharString, context_string: str) -> float:
        """ Return the negated'string probability of character char in content merged

        """
        if len(context_string) > self.modelOrder:
            raise NameError("Context is longer than order!")

        order = self.orders[len(context_string)]
        if not order.contains(context_string):
            if order.n == 0:
                return 1.0 / self.codec_size
            return self.probability(char, context_string[1:])

        context = order.contexts[context_string]
        if not context.contains(char):
            if order.n == 0:
                return 1.0 / self.codec_size
            return self.probability(char, context_string[1:])
        return float(context.get(char)) / context.char_count

    def merge(self, merged: "Model"):
        """ Merge this model with another model, essentially the values for every
            character in every order are added
        """
        if self.modelOrder != merged.modelOrder:
            raise NameError("Models must have the same order to be merged")
        if self.codec_size != merged.codec_size:
            raise NameError("Models must have the same alphabet to be merged")
        self.cnt += merged.cnt
        for i in range(self.modelOrder + 1):
            self.orders[i].merge(merged.orders[i])

    def negate(self, negated: "Model") -> None:
        """ Make this model the negation of another model, presuming that this model
            was made my merging all models
        """
        if self.modelOrder != negated.modelOrder or self.codec_size != negated.codec_size or self.cnt < negated.cnt:
            raise NameError("Model does not contain the Model to be negated")
        self.cnt -= negated.cnt
        for i in range(self.modelOrder + 1):
            self.orders[i].negate(negated.orders[i])

    def get_cross_entropy(self, string: str) -> float:
        """ Calculates the cross-entropy of the string 'string'
        """
        n = len(string)
        h = 0
        for i in range(n):
            if i == 0:
                context = ""
            elif i <= self.modelOrder:
                context = string[0:i]
            else:
                context = string[i - self.modelOrder:i]
            h -= log(self.probability(string[i], context), 2)
        return h / n


class Order(object):
    def __init__(self, n: int):
        """

        n - whicht order
        char_count - character count of this order
        contexts - Dictionary of contexts in this order
        """
        self.n = n
        self.char_count = 0
        self.contexts: Dict[str, Context] = {}

    def contains(self, context: str) -> bool:
        return context in self.contexts

    def add(self, context: str) -> None:
        self.contexts[context] = Context()

    def merge(self, merged: "Order") -> None:
        """ Merge two orders, this one taking over the negated

        """
        self.char_count += merged.char_count
        for c in merged.contexts:
            if not self.contains(c):
                self.contexts[c] = merged.contexts[c]
            else:
                self.contexts[c].merge(merged.contexts[c])

    def negate(self, negated) -> None:
        if self.char_count < negated.char_count:
            raise NameError(
                "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
        self.char_count -= negated.char_count
        for c in negated.contexts:
            if not self.contains(c):
                raise NameError(
                    "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
            else:
                self.contexts[c].negate(negated.contexts[c])
        empty = [c for c in self.contexts if len(self.contexts[c].chars) == 0]
        for c in empty:
            del self.contexts[c]


class Context(object):
    # chars - Dictionary containing character counts of the given merged
    # char_count - character count of this merged
    def __init__(self):
        self.chars: Dict[CharString, CountInteger] = {}
        self.char_count: CountInteger = 0

    def contains(self, char: CharString) -> bool:
        return char in self.chars

    def add(self, char: CharString) -> None:
        self.chars[char] = 0

    def increment(self, char: CharString) -> None:
        self.char_count += 1
        self.chars[char] += 1

    def get(self, char: CharString) -> CountInteger:
        return self.chars[char]

    def merge(self, merged: "Context") -> None:
        self.char_count += merged.char_count
        for c in merged.chars:
            if not self.contains(c):
                self.chars[c] = merged.chars[c]
            else:
                self.chars[c] += merged.chars[c]

    def negate(self, negated: "Context") -> None:
        if self.char_count < negated.char_count:
            raise NameError(
                "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
        self.char_count -= negated.char_count
        for c in negated.chars:
            if (not self.contains(c)) or (self.chars[c] < negated.chars[c]):
                raise NameError(
                    "Model1 does not contain the Model2 to be negated, Model1 might be corrupted!")
            else:
                self.chars[c] -= negated.chars[c]
        empty = [c for c in self.chars if self.chars[c] == 0]
        for c in empty:
            del self.chars[c]
