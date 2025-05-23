'''
Here you will be implementing the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''

from enum import Enum
from document_preprocessor import Tokenizer
from collections import Counter, defaultdict


class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex, PositionalIndex
    PositionalIndex = 'PositionalIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.statistics = {}   # the central statistics of the index
        self.statistics['vocab'] = Counter()  # token count
        self.vocabulary = set()  # the vocabulary of the collection
        # metadata like length, number of unique tokens of the documents
        self.document_metadata = {}

        self.index = defaultdict(list)  # the index

    # NOTE: The following functions have to be implemented in the two inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str], original_length: int = None) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document after filtering
            original_length: The original number of tokens before filtering
                If None, it will be set to len(tokens)
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filtered tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filtered tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        if docid in self.document_metadata:
            del self.document_metadata[docid]

        for token in list(self.index.keys()):
            self.index[token] = [(doc, freq) for doc, freq in self.index[token] if doc != docid]
            if not self.index[token]:
                del self.index[token]
                self.vocabulary.remove(token)

    def add_doc(self, docid: int, tokens: list[str], original_length: int = None) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document after filtering
            original_length: The original number of tokens before filtering
                If None, it will be set to len(tokens)
        """
        if original_length is None:
            original_length = len(tokens)

        token_counts = Counter(tokens)
        self.document_metadata[docid] = {
            'unique_tokens': len(token_counts),
            "length": original_length,       # Total tokens including filtered tokens
            "stored_length": len(tokens)     # Tokens after filtering
        }
        for token, count in token_counts.items():
            self.index[token].append((docid, count))
            self.vocabulary.add(token)
            self.statistics['vocab'][token] += count

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        return self.index.get(term, [])

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        return self.document_metadata.get(doc_id, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        postings = self.index.get(term, [])
        term_count = sum(freq for _, freq in postings)
        doc_frequency = len(postings)
        return {
            "term_count": term_count,
            "doc_frequency": doc_frequency
        }

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filtered tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filtered tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        total_token_count = sum(metadata['length'] for metadata in self.document_metadata.values())
        stored_total_token_count = sum(metadata['stored_length'] for metadata in self.document_metadata.values())
        number_of_documents = len(self.document_metadata)
        mean_document_length = total_token_count / number_of_documents if number_of_documents > 0 else 0

        return {
            "unique_token_count": len(self.index),
            "total_token_count": total_token_count,
            "stored_total_token_count": stored_total_token_count,
            "number_of_documents": number_of_documents,
            "mean_document_length": mean_document_length
        }

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        import json
        import os
        os.makedirs(index_directory_name, exist_ok=True)
        data_to_save = {
            'index': dict(self.index),  # Convert defaultdict to a regular dict for JSON compatibility
            'statistics': self.statistics,
            'vocabulary': list(self.vocabulary),  # Convert set to list for JSON compatibility
            'document_metadata': self.document_metadata  # Already a dict, so no conversion needed
        }
        with open(os.path.join(index_directory_name, 'index.json'), 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f)
        print(f'Index saved to {index_directory_name}')

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        import json
        import os
        with open(os.path.join(index_directory_name, 'index.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.index = defaultdict(list, {k: v for k, v in data['index'].items()})
        self.statistics = data['statistics']
        self.vocabulary = set(data['vocabulary'])
        self.document_metadata = {int(k): v for k, v in data['document_metadata'].items()}

        print(f'Index loaded from {index_directory_name}')


class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()
        self.statistics['index_type'] = 'PositionalIndex'

    def add_doc(self, docid: int, tokens: list[str], original_length: int = None) -> None:
        if original_length is None:
            original_length = len(tokens)

        token_positions = defaultdict(list)
        for position, token in enumerate(tokens):
            token_positions[token].append(position)

        self.document_metadata[docid] = {
            "unique_tokens": len(token_positions),
            "length": original_length,
            "stored_length": len(tokens)
        }

        for token, positions in token_positions.items():
            self.index[token].append((docid, len(positions), positions))
            self.vocabulary.add(token)
            self.statistics['vocab'][token] += len(positions)

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        postings = self.index.get(term, [])
        term_count = sum(freq for _, freq, _ in postings)
        doc_frequency = len(postings)
        return {
            "term_count": term_count,
            "doc_frequency": doc_frequency
        }

    def get_postings(self, term: str) -> list:
        return self.index.get(term, [])


class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the entire corpus at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            An inverted index

        '''
        import os
        import json
        import gzip
        import csv
        from tqdm import tqdm

        # Determine which index type to create
        if index_type == IndexType.BasicInvertedIndex:
            index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            index = PositionalInvertedIndex()
        else:
            raise ValueError("Unsupported index type")

        # Load the doc2query.csv into a dictionary
        if doc_augment_dict is None:
            doc_augment_dict = {}
            if os.path.exists('doc2query.csv'):
                with open('doc2query.csv', 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        docid = int(row['docID'])
                        query = row['query']
                        if docid in doc_augment_dict:
                            doc_augment_dict[docid].append(query)
                        else:
                            doc_augment_dict[docid] = [query]

        # First pass to calculate term frequencies if minimum_word_frequency > 0
        term_frequencies = Counter()
        if minimum_word_frequency > 0:
            doc_count = 0
            if dataset_path.endswith('.jsonl.gz'):
                open_func = gzip.open
                mode = 'rt'
            elif dataset_path.endswith('.jsonl'):
                open_func = open
                mode = 'r'
            else:
                raise ValueError("Unsupported file format. Only .jsonl and .jsonl.gz are supported.")

            with open_func(dataset_path, mode, encoding='utf-8') as f:
                for line in f:
                    if max_docs != -1 and doc_count >= max_docs:
                        break
                    document = json.loads(line.strip())
                    doc_id = int(document.get('ID', doc_count))  # Get the docid from the JSON object

                    # Load the document's text
                    text = document.get(text_key, '')

                    # Append pre-generated queries to the text
                    if doc_id in doc_augment_dict:
                        queries = ' '.join(doc_augment_dict[doc_id])
                        text += ' ' + queries

                    tokens = document_preprocessor.tokenize(text)
                    term_frequencies.update(tokens)
                    doc_count += 1

        # Read the collection and process/index each document
        doc_count = 0
        if dataset_path.endswith('.jsonl.gz'):
            open_func = gzip.open
            mode = 'rt'
        elif dataset_path.endswith('.jsonl'):
            open_func = open
            mode = 'r'
        else:
            raise ValueError("Unsupported file format. Only .jsonl and .jsonl.gz are supported.")

        with open_func(dataset_path, mode, encoding='utf-8') as f:
            for line in tqdm(f, desc="Indexing documents"):
                if max_docs != -1 and doc_count >= max_docs:
                    break
                document = json.loads(line.strip())
                #doc_id = int(document.get('ID', doc_count))  # Get the docid from the JSON object
                doc_id = int(document['ID'])

                # Load the document's text
                text = document.get(text_key, '')

                # Append pre-generated queries to the text
                if doc_id in doc_augment_dict:
                    queries = ' '.join(doc_augment_dict[doc_id])
                    text += ' ' + queries

                # Proceed as normal
                tokens = document_preprocessor.tokenize(text)
                original_length = len(tokens)

                # Remove stopwords
                if stopwords:
                    tokens = [token for token in tokens if token not in stopwords]

                # Remove tokens with frequency less than minimum_word_frequency
                if minimum_word_frequency > 0:
                    tokens = [token for token in tokens if term_frequencies[token] >= minimum_word_frequency]

                index.add_doc(doc_id, tokens, original_length)
                doc_count += 1

        # Update statistics
        extra_stats = index.get_statistics()
        index.statistics['unique_token_count'] = extra_stats['unique_token_count']
        index.statistics['total_token_count'] = extra_stats['total_token_count']
        index.statistics['stored_total_token_count'] = extra_stats['stored_total_token_count']
        index.statistics['number_of_documents'] = extra_stats['number_of_documents']
        index.statistics['mean_document_length'] = extra_stats['mean_document_length']

        return index


'''
The following class is a stub class with none of the essential methods implemented. It is merely here as an example.
'''


class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''

    def add_doc(self, docid, tokens, original_length=None):
        """Tokenize a document and add term ID """
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1

    def save(self):
        print('Index saved!')



