from . import Vectorizer
import numpy as np
import scipy.cluster.vq

class Summarizer:
    def __init__(self, corpus, diversity_factor=6, normalize_scores=False, random=False):
        self.corpus = corpus
        self.diversity_factor = diversity_factor
        self.vectorizer = Vectorizer(self.corpus)
        self.normalize = normalize_scores
        self.random = random

    def summarize_by_indices(self, document_ix, budget, saturation_threshold=0.8, cost_scaling_factor=0.9):
        summary_ixs = np.array([], dtype=np.int32)

        doc_matrix = self.vectorizer.document_sentences(document_ix)
        idf_weighted = doc_matrix * self.vectorizer.idf
        nonzero_columns = np.sum(idf_weighted, axis=0) > 0
        idf_weighted = idf_weighted[:, nonzero_columns]
        similarity_matrix = self.vectorizer.similarity_matrix(idf_weighted)
        sentence_count, _ = similarity_matrix.shape
        upper_bounds = np.sum(similarity_matrix, axis=0)
        saturation_thresholds = upper_bounds * saturation_threshold
        expected_similarities = upper_bounds / float(sentence_count)

        if sentence_count == 0 or len(idf_weighted) == 0:
            self.__print_warning(document_ix, 'empty sentence count')
            # Empty
            return summary_ixs

        cluster_count = sentence_count // 5
        if cluster_count == 0:
            cluster_count = sentence_count
        _, cluster_labels = scipy.cluster.vq.kmeans2(idf_weighted, cluster_count, minit='++', seed=0xBEEFBEEF)
        #print(cluster_labels)
        sentence_costs = self.vectorizer.document_word_counts(document_ix)

        remaining_ixs = list(range(sentence_count))
        current_score = 0
        current_cost = 0

        while len(remaining_ixs) > 0:
            max_base_score = current_score
            max_score_gain = None
            max_score_arg = None

            scores = np.zeros((len(remaining_ixs), 2))

            for j in range(len(remaining_ixs)):
                i = remaining_ixs[j]
                candidate_set = np.append(summary_ixs, i)
                scores[j, 0] = self.coverage(candidate_set, similarity_matrix, saturation_thresholds)
                scores[j, 1] = self.diversity(candidate_set, cluster_count, cluster_labels, expected_similarities)

            # normalize
            if self.normalize:
                max_coverage = np.amax(scores[:, 0])
                max_diversity = np.amax(scores[:, 1])
                if not np.isclose(max_coverage, 0):
                    scores[:, 0] /= max_coverage
                if not np.isclose(max_diversity, 0):
                    scores[:, 1] /= max_diversity
                print(scores)

            # argmax
            for j in range(len(remaining_ixs)):
                i = remaining_ixs[j]
                candidate_set = np.append(summary_ixs, i)
                coverage_score = scores[j, 0]
                diversity_score = scores[j, 1]
                base_score = coverage_score + self.diversity_factor * diversity_score
                if self.random:
                    base_score = current_score + np.random.rand() * 10
                # NOTE: not the score to be added to current score, just for evaluating which to pick now
                #
                # Some sentences are degenerated and 'dont contain words', just assign 0 gain to those
                score_gain = 0
                if sentence_costs[i] > 0:
                    score_gain = (base_score - current_score) / float(sentence_costs[i]**cost_scaling_factor)
                # print('gain: ' + str(score_gain))
                if max_score_gain == None or score_gain > max_score_gain:
                    max_score_gain = score_gain
                    max_base_score = base_score
                    max_score_arg = i
                # print('cc: ' + str(sentence_costs[max_score_arg]))
            if sentence_costs[max_score_arg] + current_cost <= budget and max_base_score - current_score > 0:
                summary_ixs = np.append(summary_ixs, max_score_arg)
                current_cost += sentence_costs[max_score_arg]
                current_score = max_base_score
                # print('cost: ' + str(current_cost))
                # print('score: ' + str(current_score))
            # Always remove from the remaining list
            remaining_ixs = [ix for ix in remaining_ixs if ix != max_score_arg]
        return summary_ixs


    # Similarity upper bounds are the upper bounds for the summand
    def coverage(self, summary_row_ixs, similarity_matrix, similarity_upper_bounds):
        # Include only the columns that correspond to sentences in the summary
        summary_similarities = similarity_matrix[summary_row_ixs]
        summary_similarity_sums = np.sum(summary_similarities, axis=0)
        bounded = np.minimum(summary_similarity_sums, similarity_upper_bounds)

        return np.sum(bounded)

    # clusters: list of lists of row indices of each cluster set
    # expected_similarities: list of size len(doc_sentences) containing the expected
    # similarity score of a sentence within the document
    # TODO: selected clusters could be straight in matrix by masking
    def diversity(self, summary_row_ixs, k, cluster_labels, expected_similarities):
        result = 0
        for cluster in range(k):
            cluster_mask = np.equal(cluster_labels, cluster)
            summary_cluster_mask = cluster_mask[summary_row_ixs]
            intersection = summary_row_ixs[summary_cluster_mask]
            result += np.sqrt(np.sum(expected_similarities[intersection]))

        return result

    def __print_warning(self, doc_ix, msg, sentence_ix=-1):
        problem_doc = self.corpus.documents()[doc_ix]
        print("[WARNING] " + msg + ": " + problem_doc.name)
        if sentence_ix > -1:
            if sentence_ix > 0:
                print(problem_doc.sentences()[sentence_ix-1].original_sentence())
            print(problem_doc.sentences()[sentence_ix].original_sentence())
