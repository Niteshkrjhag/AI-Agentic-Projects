import faiss
import numpy as np
import requests
from typing import List, Dict, Any

from openai import OpenAI

# client = OpenAI()


dimension = 768

#In the context of vector databases and libraries like FAISS (Facebook AI Similarity Search), an index is a 
#specialized data structure used to store and quickly search through large collections of high-dimensional vectors.
index = faiss.IndexFlatL2(dimension)
documents = []


def get_embedding(text):
    response = requests.post(

        "http://localhost:11434/api/embeddings",

        json={"model": "nomic-embed-text", "prompt": text}

    )
    return response.json()["embedding"]


def add_docs(texts):
    global documents
    embeddings = [get_embedding(text) for text in texts]
    vectors  = np.array(embeddings).astype("float32")
    index.add(vectors)
    documents.extend(texts)


def search(query, k = 3):
    query_vector = np.array([get_embedding(query)]).astype("float32")
    distances, indexes = index.search(query_vector, k)
    results = [(s, documents[i]) for s, i in zip(distances[0],indexes[0])]
    return results # arranged in lower the distance better closer the alignment of document to the query


print(len(get_embedding("test")))
add_docs(["AI is artificial Intelligence", "Python is a programming language", "Delhi is the capital of India"])
results = search("what is AI")
print(results)