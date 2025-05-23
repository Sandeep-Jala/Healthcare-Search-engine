import openai
import os
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env file (if using)
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("")
client = OpenAI(
    api_key="MyAPIKey",
)
def generate_answer(query, retrieved_text, max_tokens=500, temperature=0.7):
    prompt = f"""You are an AI assistant. Use the following context to answer the question. DO not use any information outside the context.

    Context:
    {retrieved_text}

    Question:
    {query}

    Answer:"""


    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are a Helpful assistant that looks in the context and generate answer for the query, do not use word like context, answer, question, etc."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            }
        ]
        }
    ],
    response_format={
        "type": "text"
    },
    temperature=0.2,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    #answer = response['choices'][0]['message']['content'].strip()
    answer = response.choices[0].message.content.strip()
    return answer

class RAGSystem:
    def __init__(self, documents, l2r_retriever):
        """
        Initializes the RAGSystem with the document corpus and a retriever.

        Args:
            documents (list): List of document texts.
            l2r_retriever (L2RRetriever): An instance of the L2RRetriever.
        """
        self.documents = documents
        self.l2r_retriever= l2r_retriever

    def get_answer(self, query, top_k=2):
        """
        Retrieves relevant documents and generates an answer using ChatGPT.

        Args:
            query (str): The user's question.

        Returns:
            str: Generated answer.
        """
        retrieved_text, ranked_doc = self.l2r_retriever.retrieve(query, top_k=top_k)
        retrieved_text = retrieved_text[:30000]
        if not retrieved_text:
            return "I'm sorry, but I couldn't find any relevant information to answer your question.", []
        answer = generate_answer(query, retrieved_text)
        return answer , ranked_doc
