import numpy as np
from scipy.sparse import lil_array

class Vectorizer:
    # TODO: make a way to pass global info as param and have partial vects
    def __init__(self, corpus):
        self.terms = list(corpus.bag_of_words().keys())
        # Used to O(1) index into the term vector
        term_reverse_index = {}
        for i in range(len(self.terms)):
            term_reverse_index[self.terms[i]] = i
        documents = corpus.documents()
        # Offset of i+1
        self.term_doc_occs = np.zeros(len(self.terms))
        self.doc_ends = []
        self.word_count = []
        self.frequency_matrix = lil_array((corpus.sentence_count(), len(self.terms)), dtype='float')
        for i in range(len(documents)):
            document = documents[i]
            sentences = document.sentences()
            sentence_corpus_ix = 0
            if len(self.doc_ends) == 0:
                self.doc_ends.append(len(sentences))
            else:
                sentence_corpus_ix = self.doc_ends[-1]
                self.doc_ends.append(len(sentences) + sentence_corpus_ix)
            for j in range(len(sentences)):
                sentence = sentences[j]
                for t in sentence.terms():
                    self.frequency_matrix[sentence_corpus_ix+j, term_reverse_index[t]] = sentence[t]
                self.word_count.append(sentence.raw_word_count())
            doc_occs = np.array([1.0 if document[term] > 0 else 0.0 for term in self.terms])
            self.term_doc_occs += doc_occs
        self.word_count = np.array(self.word_count)
        self.idf = self.__inverse_document_frequencies()

    def __inverse_document_frequencies(self):
        document_count = len(self.doc_ends)
        return np.log(document_count * np.reciprocal(self.term_doc_occs))

    def ngram_mask(self, n):
        return np.array([term.count(' ') == n-1 for term in self.terms])

    def document_sentences(self, doc_ix):
        if doc_ix == 0:
            return self.frequency_matrix[0:self.doc_ends[0], :].toarray()
        return self.frequency_matrix[self.doc_ends[doc_ix-1]:self.doc_ends[doc_ix], :].toarray()

    def document_word_counts(self, doc_ix):
        if doc_ix == 0:
            return self.word_count[0:self.doc_ends[0]]
        return self.word_count[self.doc_ends[doc_ix-1]:self.doc_ends[doc_ix]]

    # Input: sentence matrix of a document
    # Output: similarities between the sentences
    def similarity_matrix(self, sentence_matrix):
        sentence_count, _ = sentence_matrix.shape
        similarities = np.zeros((sentence_count, sentence_count), dtype='float')
        for i in range(sentence_count):
            for j in range(sentence_count):
                sentence1 = sentence_matrix[i]
                sentence2 = sentence_matrix[j]
                distance_product = np.sqrt(np.dot(sentence1, sentence1)) * np.sqrt(np.dot(sentence2, sentence2))
                if distance_product != 0:
                    similarities[i, j] = np.dot(sentence1, sentence2) / distance_product
        return similarities
