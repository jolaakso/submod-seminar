# can be removed once cleanup done
import numpy as np

class Corpus:
    def __init__(self, docs):
        self.document_list = docs
        self.__gen_bag_of_words()

    def __getitem__(self, key):
        if not key in self.word_bag:
            return 0
        return self.word_bag[key]

    def bag_of_words(self):
        return self.word_bag

    def term_count(self):
        return sum(self.word_bag.values())

    def term_frequency(self, ngram):
        return self[ngram] / self.word_count()

    def sentence_count(self):
        return sum([doc.sentence_count() for doc in self.documents()])

    def document_count(self):
        return len(self.documents())

    def documents(self):
        return self.document_list

    def contains_term(self, ngram):
        return self[ngram] > 0

    def term_document_occurrences(self, ngram):
        return sum([doc.contains_term(ngram) for doc in self.documents()])

    def __gen_bag_of_words(self):
        bag_of_words = {}
        for doc in self.document_list:
            for ngram in doc.keys():
                if ngram in bag_of_words:
                    bag_of_words[ngram] += doc[ngram]
                else:
                    bag_of_words[ngram] = doc[ngram]
        self.word_bag = bag_of_words
