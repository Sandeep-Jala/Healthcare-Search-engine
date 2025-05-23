import lightgbm
import math
from document_preprocessor import Tokenizer
from indexing import InvertedIndex, BasicInvertedIndex
from ranker import *
from collections import Counter
import csv
import pandas as pd


class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker model.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The relevance scorer
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any arguments that are needed as fields of this class
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.feature_extractor = feature_extractor

        # TODO: Initialize the LambdaMART model (but don't train it yet)
        self.model = LambdaMART()

    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores (dict): A dictionary of queries mapped to a list of 
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            tuple: A tuple containing the training data in the form of three lists: x, y, and qgroups
                X (list): A list of feature vectors for each query-document pair
                y (list): A list of relevance scores for each query-document pair
                qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y.
        # This is for LightGBM to know how many relevance scores we have per query.
        X = []
        y = []
        qgroups = []

        for query, doc_relevance_scores in query_to_document_relevance_scores.items():
            query_parts = self.document_preprocessor.tokenize(query)
            query_parts = [token for token in query_parts if token not in self.stopwords]


            doc_term_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)
            title_term_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)

            query_docs_features = []
            query_docs_relevance = []

            for docid, relevance in doc_relevance_scores:
                features = self.feature_extractor.generate_features(
                    docid, doc_term_counts.get(docid, {}), title_term_counts.get(docid, {}), query_parts
                )
                query_docs_features.append(features)
                query_docs_relevance.append(relevance)

            X.extend(query_docs_features)

            y.extend(query_docs_relevance)
            qgroups.append(len(query_docs_relevance))
        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # TODO: Retrieve the set of documents that have each query word (i.e., the postings) and
        # create a dictionary that keeps track of their counts for the query word
        doc_term_counts = {}
        for term in query_parts:
            postings = index.get_postings(term)
            for docid, freq in postings:
                if docid not in doc_term_counts:
                    doc_term_counts[docid] = {}
                doc_term_counts[docid][term] = freq
        return doc_term_counts

    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """
        # TODO: Convert the relevance data into the right format for training data preparation

        # TODO: prepare the training data by featurizing the query-doc pairs and
        # getting the necessary datastructures

        # TODO: Train the model
        training = {}
        with open(training_data_filename, 'r', encoding='utf-8', errors='replace') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                query = row["query"]  # Replace "query" with the actual column name
                docid = row["docid"]  # Replace "docid" with the actual column name
                relevance = row["rel"]  # Replace "relevance" with the actual column name

                if query not in training:
                    training[query] = []
                training[query].append((int(docid), round(float(relevance))))
        X, y, qgroups = self.prepare_training_data(training)
        self.model.fit(X, y, qgroups)

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # TODO: Return a prediction made using the LambdaMART model
        return self.model.predict(X)
        pass

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # TODO: Retrieve potentially-relevant documents
        query_parts = self.document_preprocessor.tokenize(query)
        query_parts = [term for term in query_parts if term not in self.stopwords]


        # TODO: Fetch a list of possible documents from the index and create a mapping from
        # a document ID to a dictionary of the counts of the query terms in that document.
        # You will pass the dictionary to the RelevanceScorer as input.
        #
        # NOTE: we collect these here (rather than calling a Ranker instance) because we'll
        # pass these doc-term-counts to functions later, so we need the accumulated representations

        # TODO: Accumulate the documents word frequencies for the title and the main body
        doc_term_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)
        title_term_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)

        # TODO: Score and sort the documents by the provided scrorer for just the document's main text (not the title)
        # This ordering determines which documents we will try to *re-rank* using our L2R model
        # document_scoreced = [(docid, self.scorer.score(docid, counts, Counter(query_parts))) for docid, counts in doc_term_counts.items()]
        # document_scoreced.sort(key=lambda x: x[1], reverse=True)
        document_scoreced = self.ranker.query(query)
        #document_scoreced.sort(key=lambda x: x[1], reverse=True)

        # TODO: Filter to just the top 100 documents for the L2R part for re-ranking
        top_100 = document_scoreced[:100]

        # TODO: Construct the feature vectors for each query-document pair in the top 100
        X = [self.feature_extractor.generate_features(docid, doc_term_counts.get(docid, {}), title_term_counts.get(docid, {}), query_parts) for docid, i in top_100]


        # TODO: Use your L2R model to rank these top 100 documents
        ranked_documents = self.predict(X)

        # TODO: Sort posting_lists based on scores
        # posting_lists = [(docid, score) for docid, score in zip([docid for docid, _ in top_100], ranked_documents)]

        # # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked
        # for docid, score in document_scoreced[100:]:
        #     posting_lists.append((docid, score))

        # # TODO: Return the ranked documents
        # return posting_lists
        reranked_docs = list(zip([docid for docid, _ in top_100], ranked_documents))
        reranked_docs.sort(key=lambda x: x[1], reverse=True)

        # Combine re-ranked documents with the rest
        final_ranking = reranked_docs + document_scoreced[100:]

        return final_ranking
        pass


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str]):
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
        """
        # TODO: Set the initial state using the arguments
        self.document_index = document_index
        self.title_index = title_index
    
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
  

        # TODO: For the recognized categories (i.e,. those that are going to be features), considering
        # how you want to store them here for faster featurizing





        # TODO (HW2): Initialize any RelevanceScorer objects you need to support the methods below.
        #             Be sure to use the right InvertedIndex object when scoring.
        self.Doc_tf_idf_scorer = TF_IDF(self.document_index)
        self.Title_tf_idf_scorer = TF_IDF(self.title_index)
        self.Doc_tf_scorer = TF(self.document_index)
        self.Title_tf_scorer = TF(self.title_index)
        self.Bm25_scorer = BM25(self.document_index)
        self.Pivoted_normalization_scorer = PivotedNormalization(self.document_index)


    # TODO: Article Length
    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.document_index.get_doc_metadata(docid)['length']
        

    # TODO: Title Length
    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.get_doc_metadata(docid)['length']
        

    # TODO: TF
    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """

        if index == self.document_index:
            return self.Doc_tf_scorer.score(docid, word_counts, Counter(query_parts))
        elif index == self.title_index:
            return self.Title_tf_scorer.score(docid, word_counts, Counter(query_parts))
        else:
            raise ValueError("Invalid index")

    # TODO: TF-IDF
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        if index == self.document_index:
            score = self.Doc_tf_idf_scorer.score(docid, word_counts, Counter(query_parts))
            return score
        elif index == self.title_index:
            score = self.Title_tf_idf_scorer.score(docid, word_counts, Counter(query_parts))
            return score
        else:
            raise ValueError("Invalid index")

        pass

    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        # TODO: Calculate the BM25 score and return it
        score = self.Bm25_scorer.score(docid, doc_word_counts, Counter(query_parts))
        return score

    # TODO: Pivoted Normalization
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        # TODO: Calculate the pivoted normalization score and return it
        score = self.Pivoted_normalization_scorer.score(docid, doc_word_counts, Counter(query_parts))
        return score


    # TODO 11: Add at least one new feature to be used with your L2R model.
    #def euclidean_distance(self, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        # query_counts = Counter(query_parts)
        # doc_word_counts = {term: sum(value.values()) if isinstance(value, dict) else value for term, value in doc_word_counts.items()}
        # common_terms = set(doc_word_counts.keys()).union(query_counts.keys())
        # distance = 0.0

        # for term in common_terms:
        #     doc_count = int(doc_word_counts.get(term, 0))
        #     query_count = int(query_counts.get(term, 0))
        #     distance += (doc_count - query_count) ** 2

        # return math.sqrt(distance)
    def get_query_coverage_ratio_title(self, title_word_counts: dict[str, int], query_parts: list[str]) -> float:
        if not query_parts:
            return 0.0

        query_terms_in_title = sum(1 for term in query_parts if term in title_word_counts)
        return query_terms_in_title / len(query_parts)
        

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str]) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """

        feature_vector = []
        query_text = " ".join(query_parts)

        # TODO: Document Length
        feature_vector.append(self.get_article_length(docid))

        # TODO: Title Length
        feature_vector.append(self.get_title_length(docid))

        # TODO Query Length
        feature_vector.append(len(query_parts))

        # TODO: TF (document)
        feature_vector.append(self.get_tf(self.document_index, docid, doc_word_counts, query_parts))

        # TODO: TF-IDF (document)
        feature_vector.append(self.get_tf_idf(self.document_index, docid, doc_word_counts, query_parts))

        # TODO: TF (title)
        feature_vector.append(self.get_tf(self.title_index, docid, title_word_counts, query_parts))

        # TODO: TF-IDF (title)
        feature_vector.append(self.get_tf_idf(self.title_index, docid, title_word_counts, query_parts))

        # TODO: BM25
        feature_vector.append(self.get_BM25_score(docid, doc_word_counts, query_parts))

        # TODO: Pivoted Normalization
        feature_vector.append(self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts))

        # TODO: Add at least one new feature to be used with your L2R model.
        feature_vector.append(self.get_query_coverage_ratio_title(title_word_counts, query_parts))

        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 10,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.04,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": 1,
        }

        if params:
            default_params.update(params)

        # TODO: initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.model = lightgbm.LGBMRanker(**default_params)

    def fit(self,  X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples.
            y_train (array-like): Target values.
            qgroups_train (array-like): Query group sizes for training data.

        Returns:
            self: Returns the instance itself.
        """

        # TODO: fit the LGBMRanker's parameters using the provided features and labels
        self.model.fit(X_train, y_train, group=qgroups_train)

        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length.

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """

        # TODO: Generating the predicted values using the LGBMRanker
        return self.model.predict(featurized_docs)
