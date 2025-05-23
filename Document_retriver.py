from l2r import L2RRanker

class L2RRetriever:
    def __init__(self, l2r_ranker: L2RRanker, doc_text):
        if l2r_ranker is None:
            raise ValueError("l2r_ranker must be provided.")
        if doc_text is None:
            raise ValueError("doc_text must be provided.")
        
        self.l2r_ranker = l2r_ranker
        self.doc_text = doc_text
    
    def retrieve(self, query: str, top_k: int = 2) -> tuple[str, list]:
        """
        Retrieves the top_k documents relevant to the query.

        Args:
            query (str): The user's search query.
            top_k (int): Number of top documents to retrieve.

        Returns:
            (str, list): A tuple where the first element is the concatenated string
                        of retrieved document texts and the second element is the ranked_documents list.
        """
        try:
            ranked_documents = self.l2r_ranker.query(query)
            if not ranked_documents:
                return "", []

            top_documents = ranked_documents[:top_k]
            doc_ids = [doc_id for doc_id, _ in top_documents]
            retrieved_texts = []

            for doc_id in doc_ids:
                if doc_id in self.doc_text:
                    retrieved_texts.append(self.doc_text[doc_id])

            concatenated_text = ' '.join(retrieved_texts)
            return concatenated_text, ranked_documents

        except Exception as e:
            # Handle the exception and return empty results
            return "", []

