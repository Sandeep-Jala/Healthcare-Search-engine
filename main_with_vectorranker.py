import os
import json
import csv
import gzip
from collections import Counter, defaultdict

# Third-party libraries
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Local imports (make sure these modules are in the same directory or installed as packages)
from indexing import IndexType, InvertedIndex, BasicInvertedIndex, Indexer
from document_preprocessor import RegexTokenizer, Doc2QueryAugmenter
from relevance import map_score, ndcg_score, run_relevance_tests
from ranker import BM25, Ranker
import l2r
from l2r import L2RRanker, L2RFeatureExtractor
from Document_retriver import L2RRetriever
from RAG import RAGSystem
from vector_ranker import *
from Spellingcorrection import *
import numpy as np

def main():
    load_dotenv()  # Load environment variables if needed

    stopwords = set()
    with open('stopwords.txt', 'r', encoding='utf-8') as file:
        for line in file:
            stopwords.add(line.strip().lower())

    print('Loaded %d stopwords' % len(stopwords))

    Tokenizer = RegexTokenizer()

    index_type = IndexType.BasicInvertedIndex  # Or PositionalIndex based on your needs
    dataset_path = 'cleaned_output.jsonl'  # Path to your JSONL file
    minimum_word_frequency = 0  # Set to control minimum token frequency, 0 to ignore
    text_key = "text"  # The JSON key containing the text content

    # Load documents
    doc_text = {}
    doc_dict = {}
    with open(dataset_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc = json.loads(line)
            doc_text[doc['ID']] = doc['text']
            doc_dict[doc['ID']] = doc['URL']

    models = 'google/flan-t5-base'
    augmenter = Doc2QueryAugmenter(models)
    prefix = "Generate a query for the following text: "

    # Example of generating queries for a subset of documents (optional)
    documents = list(doc_text.values())[:1]
    for doc in tqdm(documents):
        query = augmenter.get_queries(doc, n_queries=10, prefix_prompt=prefix)
    # query now holds generated queries for the first document

    DOC2QUERY_PATH = 'queries.csv'
    doc_augment_dict = defaultdict(lambda: [])
    with open(DOC2QUERY_PATH, 'r', encoding='utf-8') as file:
        dataset = csv.reader(file)
        for idx, row in tqdm(enumerate(dataset), total=123649):
            if idx == 0:
                continue
            doc_augment_dict[int(row[0])].append(row[2])

    index_directory_name = 'index'
    if os.path.exists(index_directory_name):
        print("Index found. Loading existing index...")
        index = BasicInvertedIndex()
        index.load(index_directory_name)
        
    else:
        print("Index not found. Creating and saving index...")
        index = Indexer.create_index(
            index_type=index_type,
            dataset_path=dataset_path,
            document_preprocessor=Tokenizer,
            stopwords=stopwords,
            minimum_word_frequency=minimum_word_frequency,
            text_key=text_key,
            doc_augment_dict=doc_augment_dict
        )
        index.save(index_directory_name)

    # Get some meta data about doc 1 (optional)
    doc_meta = index.get_doc_metadata(0)

    # Set up BM25 scorer
    bm25_scorer = BM25(
        index=index,
        parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}
    )

    # Initialize Ranker with BM25 scorer
    ranker = Ranker(
        index=index,
        document_preprocessor=Tokenizer,
        stopwords=stopwords,
        scorer=bm25_scorer,
        raw_text_dict=doc_dict
    )

    bi_encoder_model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # or any other sentence-transformers model
    biencoder_model = SentenceTransformer(bi_encoder_model_name, device='cpu')

    documents_list = list(doc_text.values())
    # Encode the documents once
    encoded_docs = biencoder_model.encode(documents_list, convert_to_tensor=False)
    encoded_docs = np.array(encoded_docs)  # Ensure it's a numpy array

    # Create a mapping from row number to doc ID
    row_to_docid = list(doc_text.keys())

    # Initialize VectorRanker
    vector_ranker = VectorRanker(
        bi_encoder_model_name=bi_encoder_model_name,
        encoded_docs=encoded_docs,
        row_to_docid=row_to_docid
    )


    # Create the index for titles
    text_key = "Title"  # Now using Title as the text key
    title_index_directory_name = 'Titleindex'
    if os.path.exists(title_index_directory_name):
        print("Title index found. Loading existing title index...")
        Title_index = BasicInvertedIndex()
        Title_index.load(title_index_directory_name)

    else:
        print("Title index not found. Creating and saving...")
        Title_index = Indexer.create_index(
            index_type=index_type,
            dataset_path=dataset_path,
            document_preprocessor=Tokenizer,
            stopwords=stopwords,
            minimum_word_frequency=minimum_word_frequency,
            text_key=text_key,
            doc_augment_dict=doc_augment_dict
        )
        Title_index.save(title_index_directory_name)

    # Set up L2R feature extractor and ranker
    feature_extractor = L2RFeatureExtractor(
        document_index=index,
        title_index=Title_index,
        document_preprocessor=Tokenizer,
        stopwords=stopwords
    )
    l2r_ranker = L2RRanker(
        document_index=index,
        title_index=Title_index,
        document_preprocessor=Tokenizer,
        stopwords=stopwords,
        ranker=vector_ranker,
        feature_extractor=feature_extractor
    )

    # Train the L2R ranker with relevance data (make sure 'ranked_documents.csv' exists)
    l2r_ranker.train('ranked_documents.csv')

    # Create L2R retriever
    l2r_retriever = L2RRetriever(l2r_ranker, doc_text)

    # Prepare RAG System
    documents_list = list(doc_text.values())  # Convert doc_text dict to a list
    rag = RAGSystem(documents_list, l2r_retriever)

    # Conversation loop
    print("Welcome to the RAG Chatbot! Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.strip().lower() == 'exit':
            print("Goodbye!")
            break
        answer, ranked_doc = rag.get_answer(query, top_k=2)
        print(f"Chatbot: {answer}\n")
        print("Top 10 ranked documents:\n")
        for doc_id, score in ranked_doc[:10]:
            print(f"Document ID: {doc_id}")
            print(doc_dict[doc_id])
        print(50*'-')
        print("\n")

if __name__ == "__main__":
    main()
