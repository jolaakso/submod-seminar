import nltk as nl
from . import Sentence

class Document:
    def __init__(self, doc, name=''):
        # doc
        # Is either the whole document as string, a list of sentence strings
        # or a list of Sentence objects
        if isinstance(doc, str):
            self.__init__(nl.tokenize.sent_tokenize(doc), name)
            return

        if isinstance(doc, list) and len(doc) > 0:
            doc = [Sentence(sentence) if isinstance(sentence, str) else sentence for sentence in doc]
        self.sentence_list = doc
        self.name = name
        self.__gen_bag_of_words()

    def __getitem__(self, key):
        if not key in self.word_bag:
            return 0
        return self.word_bag[key]

    def bag_of_words(self):
        return self.word_bag

    def sentence_vector(self, sentence):
        return np.array([sentence[ngram] for ngram in self.keys()])

    def keys(self):
        return self.word_bag.keys()

    def term_frequency(self, ngram):
        return self[ngram] / self.word_count()

    def term_count(self):
        return sum(self.word_bag.values())

    def sentence_count(self):
        return len(self.sentence_list)

    def sentences(self):
        return self.sentence_list

    def contains_term(self, ngram):
        return self[ngram] > 0

    def term_sentence_occurrences(self, ngram):
        return sum([sentence.contains_term(ngram) for sentence in self.sentences()])

    def __gen_bag_of_words(self):
        bag_of_words = {}
        for sentence in self.sentence_list:
            for ngram in sentence.keys():
                if ngram in bag_of_words:
                    bag_of_words[ngram] += sentence[ngram]
                else:
                    bag_of_words[ngram] = sentence[ngram]
        self.word_bag = bag_of_words
