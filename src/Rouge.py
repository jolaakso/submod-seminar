import numpy as np
from . import Corpus
from . import Vectorizer

class Rouge:
    # TODO: support multiple reference summaries
    def __init__(self, corpus, reference_summaries):
        corpus_docs = corpus.documents()
        self.docs_count = len(corpus_docs)
        assert(self.docs_count == len(reference_summaries))
        evaluation_corpus = Corpus(corpus_docs + reference_summaries)
        self.vectorizer = Vectorizer(evaluation_corpus)

    def ngram_recall(self, n, doc_ix, summary_ixs):
        ngram_mask = self.vectorizer.ngram_mask(n)
        doc_matrix = self.vectorizer.document_sentences(doc_ix)[:, ngram_mask]
        summary_matrix = doc_matrix[summary_ixs, :]
        ref_matrix = self.vectorizer.document_sentences(self.docs_count + doc_ix)[:, ngram_mask]
        summary_ngrams = np.sum(summary_matrix, axis=0)
        ref_ngrams = np.sum(ref_matrix, axis=0)
        true_positives = np.sum(np.minimum(summary_ngrams, ref_ngrams))
        ref_ngram_count = np.sum(ref_ngrams)

        return float(true_positives) / float(ref_ngram_count)

    def ngram_precision(self, n, doc_ix, summary_ixs):
        ngram_mask = self.vectorizer.ngram_mask(n)
        doc_matrix = self.vectorizer.document_sentences(doc_ix)[:, ngram_mask]
        summary_matrix = doc_matrix[summary_ixs, :]
        ref_matrix = self.vectorizer.document_sentences(self.docs_count + doc_ix)[:, ngram_mask]
        summary_ngrams = np.sum(summary_matrix, axis=0)
        ref_ngrams = np.sum(ref_matrix, axis=0)
        true_positives = np.sum(np.minimum(summary_ngrams, ref_ngrams))
        candidate_ngram_count = np.sum(summary_ngrams)

        return float(true_positives) / float(candidate_ngram_count)
