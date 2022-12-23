import nltk as nl
import re

class Sentence:
    def __init__(self, words):
        self.__stemmer = nl.stem.SnowballStemmer('english')
        self.original_sent = words
        self.words = words
        if isinstance(words, str):
            # This is a sentence string, tokenize it first,
            # otherwise assume it is a list of already tokenized words
            self.words = nl.tokenize.word_tokenize(self.words)
        self.__preprocess_words()
        self.__gen_bag_of_words()

    def __getitem__(self, key):
        if not key in self.word_bag:
            return 0
        return self.word_bag[key]

    def word_list(self):
        return self.words

    def bag_of_words(self):
        return self.word_bag

    def original_sentence(self):
        return self.original_sent

    def keys(self):
        return self.word_bag.keys()

    def terms(self):
        return self.keys()

    def raw_word_count(self):
        return len(self.words)

    def term_count(self):
        return sum(self.word_bag.values())

    def term_frequency(self, ngram):
        return self[ngram] / self.word_count()

    def contains_term(self, ngram):
        return self[ngram] > 0

    # Used in initialization
    # make sure to call __gen_bag_of_words afterwards
    def __preprocess_words(self):
        regex = r"[^a-z,']"
        self.words = [re.sub(regex, '', word.lower()) for word in self.words]
        self.words = [self.__stemmer.stem(word) for word in self.words if len(word) > 0]

    # Uses both words and bigrams
    def __gen_bag_of_words(self):
        bag_of_words = {}

        for i in range(len(self.words)):
            word = self.words[i]

            if word in bag_of_words:
                bag_of_words[word] += 1
            else:
                bag_of_words[word] = 1

            if i+1 < len(self.words):
                bigram = word + ' ' + self.words[i+1]
                if bigram in bag_of_words:
                    bag_of_words[bigram] += 1
                else:
                    bag_of_words[bigram] = 1
        self.word_bag = bag_of_words
