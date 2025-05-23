"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques and build your own tokenizer.
"""
import nltk
from nltk.tokenize import RegexpTokenizer
import time
# import matplotlib.pyplot as plt # since autograder is giving error for this import, I have commented it out
import json
import gzip
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        # TODO: Save arguments that are needed as fields of this class
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions if multiword_expressions else []

    
    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """
        # TODO: Add support for lower-casing and multi-word expressions
        if self.lowercase:
            input_tokens = [input_token.lower() for input_token in input_tokens]


        if self.multiword_expressions:
            if not self.lowercase:
                self.multiword_expressions = [expression.split() for expression in self.multiword_expressions]
            if self.lowercase:
                for expression in self.multiword_expressions:
                    expression = [word.lower().split() for word in expression]
            self.multiword_expressions = sorted(self.multiword_expressions, key=len, reverse=True)
            
            for i in range(len(input_tokens)):
                for expression in self.multiword_expressions:
                    if input_tokens[i:i+len(expression)] == expression:
                        input_tokens[i] = ' '.join(expression)
                        input_tokens = input_tokens[:i+1] + input_tokens[i+len(expression):]
                        break
        return input_tokens
    
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # You should implement this in a subclass, not here
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        text = text.split()
        return self.postprocess(text)


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = '\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        self.token_regex = token_regex
        self.tokenizer = RegexpTokenizer(self.token_regex)
        # TODO: Save a new argument that is needed as a field of this class
        # TODO: Initialize the NLTK's RegexpTokenizer 


    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens
        #       using the postprocess function
        tokens = self.tokenizer.tokenize(text)
        return self.postprocess(tokens)

    
def time_tokenization(tokenizer, documents):
    total_time = 0
    for doc in documents:
        text = doc['text'] 
        start_time = time.time()
        tokenizer.tokenize(text)
        end_time = time.time()
        total_time += (end_time - start_time)
    return total_time

def load_documents(file_path, limit=None):
    documents = []
    if limit:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i >= limit:
                    break
                documents.append(json.loads(line))
    else:
        with open(file_path, 'r') as file:
            for line in file:
                documents.append(json.loads(line))
    return documents

# TODO (HW3): Take in a doc2query model and generate queries from a piece of text
# Note: This is just to check you can use the models;
#       for downstream tasks such as index augmentation with the queries, use doc2query.csv
class Doc2QueryAugmenter:
    """
    This class is responsible for generating queries for a document.
    These queries can augment the document before indexing.

    MUST READ: https://huggingface.co/doc2query/msmarco-t5-base-v1

    OPTIONAL reading
        1. Document Expansion by Query Prediction (Nogueira et al.): https://arxiv.org/pdf/1904.08375.pdf
    """
    
    def __init__(self, doc2query_model_name: str = 'doc2query/msmarco-t5-base-v1') -> None:
        """
        Creates the T5 model object and the corresponding dense tokenizer.
        
        Args:
            doc2query_model_name: The name of the T5 model architecture used for generating queries
        """
        self.device = torch.device('cpu')  # Do not change this unless you know what you are doing
        self.tokenizer = T5Tokenizer.from_pretrained(doc2query_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(doc2query_model_name).to(self.device)

        # TODO (HW3): Create the dense tokenizer and query generation model using HuggingFace transformers

    def get_queries(self, document: str, n_queries: int = 5, prefix_prompt: str = '') -> list[str]:
        """
        Steps
            1. Use the dense tokenizer/encoder to create the dense document vector.
            2. Use the T5 model to generate the dense query vectors (you should have a list of vectors).
            3. Decode the query vector using the tokenizer/decode to get the appropriate queries.
            4. Return the queries.
         
            Ensure you take care of edge cases.
         
        OPTIONAL (DO NOT DO THIS before you finish the assignment):
            Neural models are best performing when batched to the GPU.
            Try writing a separate function which can deal with batches of documents.
        
        Args:
            document: The text from which queries are to be generated
            n_queries: The total number of queries to be generated
            prefix_prompt: An optional parameter that gets added before the text.
                Some models like flan-t5 are not fine-tuned to generate queries.
                So we need to add a prompt to instruct the model to generate queries.
                This string enables us to create a prefixed prompt to generate queries for the models.
                See the PDF for what you need to do for this part.
                Prompt-engineering: https://en.wikipedia.org/wiki/Prompt_engineering
        
        Returns:
            A list of query strings generated from the text
        """
        # Note: Feel free to change these values to experiment
        if not document:
            return []
        document_max_token_length = 400  # as used in OPTIONAL Reading 1
        top_p = 0.85
        
        document = prefix_prompt + document

        input_ids = self.tokenizer.encode(document, max_length=document_max_token_length, truncation=True, return_tensors='pt')
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=100, #max token for query
            do_sample=True,
            top_p=top_p,
            num_return_sequences=n_queries
        )
        queries = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        return queries

        # NOTE: See https://huggingface.co/doc2query/msmarco-t5-base-v1 for details

        # TODO (HW3): For the given model, generate a list of queries that might reasonably be issued to search
        #       for that document
        # NOTE: Do not forget edge cases
        pass


# Don't forget that you can have a main function here to test anything in the file
if __name__ == '__main__':
    pass

    # mwe_filepath = 'multi_word_expressions.txt'
    # mwe_list = []
    # with open(mwe_filepath, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         mwe_list.append(line.strip())

    # file_path = 'data/wikipedia_200k_dataset.jsonl.gz'
    # documents = load_documents(file_path)
    # len_doc = len(documents)
    # print(f"Number of documents: {len_doc}")
    # limit = 1000
    # documents = load_documents(file_path, limit=limit)
    
    # split_tokenizer = SplitTokenizer()
    # regex_tokenizer = RegexTokenizer()
    # spacy_tokenizer = SpaCyTokenizer()
    # split_time = time_tokenization(split_tokenizer, documents)
    # regex_time = time_tokenization(regex_tokenizer, documents)
    # spacy_time = time_tokenization(spacy_tokenizer, documents)


    # split_avg_time = split_time/ limit
    # regex_avg_time = regex_time/ limit
    # spacy_avg_time = spacy_time/ limit
    # print(f"SplitTokenizer time: {split_avg_time}")
    # print(f"RegexTokenizer time: {regex_avg_time}")
    # print(f"SpaCyTokenizer time: {spacy_avg_time}")
    # tokenizer_names = ['SplitTokenizer', 'RegexTokenizer', 'SpaCyTokenizer']
    # tokenizer_times = [split_avg_time, regex_avg_time, spacy_avg_time]
    # plt.figure()
    # plt.bar(tokenizer_names, tokenizer_times)
    # plt.xlabel('Tokenizer')
    # plt.ylabel('Time (s)')
    # plt.title('Time taken by each tokenizer')
    # plt.savefig('Avg_time_taken_by_tokenizer_without_mwe.png')
    # plt.show()
    
    # split_tokenizer = SplitTokenizer(multiword_expressions=mwe_list)
    # regex_tokenizer = RegexTokenizer(multiword_expressions=mwe_list)
    # spacy_tokenizer = SpaCyTokenizer(multiword_expressions=mwe_list)
    # split_time = time_tokenization(split_tokenizer, documents)
    # regex_time = time_tokenization(regex_tokenizer, documents)
    # spacy_time = time_tokenization(spacy_tokenizer, documents)


    # split_avg_time = split_time/ limit
    # regex_avg_time = regex_time/ limit
    # spacy_avg_time = spacy_time/ limit
    # print(f"SplitTokenizer time: {split_avg_time}")
    # print(f"RegexTokenizer time: {regex_avg_time}")
    # print(f"SpaCyTokenizer time: {spacy_avg_time}")

    # tokenizer_names = ['SplitTokenizer', 'RegexTokenizer', 'SpaCyTokenizer']
    # tokenizer_times = [split_avg_time, regex_avg_time, spacy_avg_time]
    # plt.figure()
    # plt.bar(tokenizer_names, tokenizer_times)
    # plt.xlabel('Tokenizer')
    # plt.ylabel('Time (s)')
    # plt.title('Time taken by each tokenizer')
    # plt.savefig('Avg_time_taken_by_tokenizer.png')
    
    # split_time = split_avg_time * len_doc
    # regex_time = regex_avg_time * len_doc
    # spacy_time = spacy_avg_time * len_doc
    # print(f"SplitTokenizer time for the entire dataset: {split_time}")
    # print(f"RegexTokenizer time for the entire dataset: {regex_time}")
    # print(f"SpaCyTokenizer time for the entire dataset: {spacy_time}")
