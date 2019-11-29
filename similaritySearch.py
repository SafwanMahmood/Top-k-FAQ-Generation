import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class SimilaritySearch:

    question_tokens = []
    term_document_matrix = []
    mapWordToId = {}
    word_list = {}

    def __init__(self, question_tokens):
        self.question_tokens = question_tokens
        self.term_document_matrix, self.mapWordToId, self.word_list = self.get_tf_idf_weights(question_tokens)
        self.inverted_index = self.initialise_inverted_index(self.word_list)
        return

    '''
        Initialises  inverted index
    '''
    def initialise_inverted_index(self, word_list):
        inverted_index = {}

        for word in word_list.keys():
            inverted_index[word] = []

        return inverted_index

    '''
        get TF-IDF weights
    '''

    def get_tf_idf_weights(self, tokens):
        word_list = {}
        for document in tokens:
            for word in document:
                word_list[word] = True

        mapWordToId = {}
        count = 0
        for word in word_list.keys():
            mapWordToId[word] = count
            count = count + 1

        termDocumentMatrix = np.zeros((len(word_list.keys()),len(tokens)))

        document_id = 0

        for document in tokens:
            for word in document:
                termDocumentMatrix[mapWordToId[word]][document_id] = termDocumentMatrix[mapWordToId[word]][document_id] + 1
            document_id += 1

        non_zero_values_in_each_row = np.log10(len(tokens)*1.0/((termDocumentMatrix != 0).sum(1)))

        termDocumentMatrix = (termDocumentMatrix.T*non_zero_values_in_each_row).T

        return termDocumentMatrix.T, mapWordToId , word_list