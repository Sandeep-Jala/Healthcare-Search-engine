"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from indexing import InvertedIndex
from collections import Counter
import math
from collections import defaultdict
import numpy as np
from sentence_transformers import CrossEncoder
class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]=None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        # 1. Tokenize query
        query_tokens = self.tokenize(query)
        query_tokens_without_sw = [token for token in query_tokens if token not in self.stopwords]
        query_word_counts = Counter(query_tokens_without_sw)

        # 2. Fetch a list of possible documents from the index
        Relevant_document = set()
        # for term in query_word_counts:
        #     Relevant_document.update(doc_id for doc_id, count in self.index.get_postings(term))
        # 2. Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)
        # score = []

        # doc_word_counts = {docid: {q_term1: count, }, ...}
        # for term in query_word_counts:
        #     postings = self.index.get_postings(term)
        #     for posting_doc_id, freq in postings:
        #         if posting_doc_id == doc_id:
        #             doc_word_counts[term] = freq
        #             break
        doc_word_counts = {}
        for query_word in query_word_counts:
            if query_word in self.index.index:
                postings = self.index.get_postings(query_word)
                for doc_id, freq in postings:
                    Relevant_document.add(doc_id)
                    if doc_id not in doc_word_counts:
                        doc_word_counts[doc_id] = {}
                    doc_word_counts[doc_id][query_word] = freq
        # 3. Return **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        scored_documents = []
        for doc_id in Relevant_document:
            score = self.scorer.score(doc_id, doc_word_counts[doc_id], query_word_counts)
            scored_documents.append((doc_id, score))

        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return scored_documents
        #raise NotImplementedError


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        
        raise NotImplementedError


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        return 10


# TODO Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        dot_product = sum(doc_word_counts[word] * query_word_counts[word] for word in query_word_counts if word in doc_word_counts)
        #magnitude_doc = math.sqrt(sum(doc_word_counts[word] ** 2 for word in query_word_counts if word in doc_word_counts))
        #magnitude_query = math.sqrt(sum(query_word_counts[word] ** 2 for word in query_word_counts if word in doc_word_counts))
        #if magnitude_doc == 0 or magnitude_query == 0:
        #    score = 0
        #else:
        #    score = dot_product / (magnitude_doc * magnitude_query)
        # 2. Return the score
        return dot_product

# TODO Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        # Article_length = self.index.document_metadata[docid]['length']
        # C = self.index.get_statistics()['total_token_count']
        # Score = 0
        # for word in query_word_counts:
        #     if word in doc_word_counts:
        #         CF = self.index.statistics['vocab'][word]
        #         cd_wi = doc_word_counts.get(word, 0)
        #         query_wi = query_word_counts.get(word, 0)
        #         if CF > 0:
        #             term_prob_collection = CF/ C
        #             Score += query_wi * (math.log(1+(cd_wi/(self.mu * term_prob_collection))))

        # Score += len(query_word_counts) * math.log(self.mu/ (Article_length + self.mu))
        doc_len = self.index.document_metadata[docid]['length']
        mu = self.parameters['mu']

        score = 0
        for word in query_word_counts:
            if word in self.index.index:
                postings = self.index.get_postings(word)
                doc_tf = doc_word_counts.get(word, 0)

                if doc_tf > 0:
                    query_tf = query_word_counts[word]
                    p_wc = self.index.get_term_metadata(word)['term_count'] / (self.index.get_statistics()['total_token_count'])
                    tfidf = np.log(1 + (doc_tf / (mu * p_wc)))
                    score += (query_tf * tfidf)
        score += len(query_word_counts) * np.log(mu / (doc_len + mu))
        return score

        # 2. Compute additional terms to use in algorithm

        # 3. For all query_parts, compute score

        # 4. Return the score
        return NotImplementedError


# TODO Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
    
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        savg_doc_length = self.index.get_statistics()['mean_document_length']
        D = self.index.get_statistics()['number_of_documents']

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query
        Score = 0
        for word in query_word_counts:
            if word in doc_word_counts:
                cq_wi = query_word_counts[word]
                cd_wi = doc_word_counts[word]

                df_wi = self.index.get_term_metadata(word)['doc_frequency']
                Article_length = self.index.document_metadata[docid]['length']
        # 3. For all query parts, compute the TF and IDF to get a score
                TF = (cd_wi * (self.k1 + 1)) /  (self.k1 * (1 - self.b + self.b * (Article_length / savg_doc_length)) + cd_wi)
                IDF = math.log((D - df_wi + 0.5) / (df_wi + 0.5))
                QTF = (cq_wi * (self.k3 + 1)) / (self.k3 + cq_wi)

        # 4. Return score
                Score += (TF * IDF * QTF)
        return Score
        return NotImplementedError


# TODO Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        Score = 0
        avdl = self.index.get_statistics()['mean_document_length']
        D = self.index.get_statistics()['number_of_documents']

        # 2. Compute additional terms to use in algorithm
        for word in query_word_counts:
            if word in doc_word_counts:
                cq_wi = query_word_counts[word]
                cd_wi = doc_word_counts[word]

                df_wi = self.index.get_term_metadata(word)['doc_frequency']
                Article_length = self.index.document_metadata[docid]['length']


        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
                TF = (1 + math.log(1 + math.log(cd_wi)))/ (1 - self.b + self.b * (Article_length / avdl))
                IDF = math.log((D + 1) / df_wi) if df_wi > 0 else 0
                QTF = cq_wi 
                Score += (TF * IDF * QTF)
        # 4. Return the score
        return Score
        return NotImplementedError


# TODO Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        score = 0
        for word in query_word_counts:
            if word in doc_word_counts:
                cd_wi = doc_word_counts[word]
                D = self.index.get_statistics()['number_of_documents']
                df_wi = self.index.get_term_metadata(word)['doc_frequency']
                idf = math.log(D/df_wi)+1
                tf = math.log(cd_wi +1)
                score += tf * idf
        # 2. Compute additional terms to use in algorithm


        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return the score
        return score

class TF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        score = 0
        for word in query_word_counts:
            if word in doc_word_counts:
                cd_wi = doc_word_counts[word]
                tf = math.log(cd_wi +1)
                score += tf 
        # 2. Compute additional terms to use in algorithm


        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return the score
        return score
        


# TODO Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8, 'n': 2, 'm': 3}) -> None:
        self.index = index
        self.parameters = parameters
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.n = parameters['n']
        self.m = parameters['m']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        savg_doc_length = self.index.get_statistics()['mean_document_length']
        D = self.index.get_statistics()['number_of_documents']

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query
        Score = 0
        for word in query_word_counts:
            if word in doc_word_counts:
                cq_wi = query_word_counts[word]
                cd_wi = doc_word_counts[word]

                df_wi = self.index.get_term_metadata(word)['doc_frequency']
                Article_length = self.index.document_metadata[docid]['length']
        # 3. For all query parts, compute the TF and IDF to get a score
                TF = (cd_wi * (self.k1 + 1)) /  (self.k1 * (1 - self.b + self.b * (Article_length / savg_doc_length)) + cd_wi)**(1/self.n)
                IDF = ((D - df_wi) / (df_wi))**(1/self.m)
                QTF = (cq_wi * (self.k3 + 1)) / (self.k3 + cq_wi)

        # 4. Return score
                Score += (TF * IDF * QTF)
        return Score


# TODO (HW3): The CrossEncoderScorer class uses a pre-trained cross-encoder model from the Sentence Transformers package
#             to score a given query-document pair; check README for details
#
# NOTE: This is not a RelevanceScorer object because the method signature for score() does not match, but it
# has the same intent, in practice
class CrossEncoderScorer:
    '''
    A scoring object that uses cross-encoder to compute the relevance of a document for a query.
    '''
    def __init__(self, raw_text_dict: dict[int, str], 
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.model = CrossEncoder(cross_encoder_model_name, device ="cpu")
        self.raw_text_dict = raw_text_dict
        

    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # NOTE: Do not forget to handle an edge case
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)
        
        
        if docid not in self.raw_text_dict or not query:
            return 0.0
        # NOTE: unlike the other scorers like BM25, this method takes in the query string itself,
        # not the tokens!
        doc_text = self.raw_text_dict.get(docid, '')

        # TODO (HW3): Get a score from the cross-encoder model
        #             Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed
        score = self.model.predict([(query, doc_text)])
        return score[0]

