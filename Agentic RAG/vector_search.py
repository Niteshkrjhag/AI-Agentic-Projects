import faiss
import numpy as np
import requests
from typing import List, Dict, Any


class VectorSearchTool:
    def __init__(self):

        self.dimension = 768

        #In the context of vector databases and libraries like FAISS (Facebook AI Similarity Search), an index is a 
        #specialized data structure used to store and quickly search through large collections of high-dimensional vectors.
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []


    def get_embedding(self, text: str):
        response = requests.post(

            "http://localhost:11434/api/embeddings",

            json={"model": "nomic-embed-text", "prompt": text}

            )
        return response.json()["embedding"]


    def add_docs(self, texts: List[str]):
        global documents
        embeddings = [self.get_embedding(text) for text in texts]
        vectors  = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.documents.extend(texts)


    def run(self, query: str, k: int = 3):
        query_vector = np.array([self.get_embedding(query)]).astype("float32")
        distances, indexes = self.index.search(query_vector, k)
        
        results = [
            {"score": float(dist), "document": self.documents[i]}
            for dist, i in zip(distances[0], indexes[0])
        ]
        return results # arranged in lower the distance better closer the alignment of document to the query

