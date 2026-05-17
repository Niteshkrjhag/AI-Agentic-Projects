import faiss
import numpy as np
import requests
from typing import List, Dict, Any
import ollama

class VectorDBSearchTool:
    def __init__(self):

        self.dimension = 768

        #In the context of vector databases and libraries like FAISS (Facebook AI Similarity Search), an index is a 
        #specialized data structure used to store and quickly search through large collections of high-dimensional vectors.
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []


    def get_embedding(self, text: str):
        response = ollama.embed(
                        model = 'nomic-embed-text',
                        input = text,
                    )
        
        return response.embeddings[0]


    def add_docs(self, texts: List[str]):
        embeddings = [self.get_embedding(text) for text in texts]
        vectors  = np.array(embeddings).astype("float32")
        print("shape of vector is : ", vectors.shape)
        self.index.add(vectors)
        self.documents.extend(texts)


    def __call__(self, query: str, k: int = 3):
        query_vector = np.array([self.get_embedding(query)]).astype("float32")
        distances, indexes = self.index.search(query_vector, k)
        
        results = [ ]
        for dist, i in zip(distances[0], indexes[0]):

            # if faiss don't finds relevent docs then it returns -1 so we need to handle this case
            if i == -1:
                continue

            results.append({"score": float(dist), "document": self.documents[i]})
        return results # arranged in lower the distance better closer the alignment of document to the query


if __name__ == '__main__':
    # 1. Initialize the vector database tool
    vdst = VectorDBSearchTool()
    
    # 2. Define a list containing multiple documents
    sample_documents = [
        "Dog is an animal and lives in Dubai",
        "Cats are independent pets that love chasing lasers.",
        "The capital of France is Paris, famous for the Eiffel Tower.",
        "Python is a versatile programming language used in AI.",
        "Dubai is known for its modern architecture and warm weather."
    ]
    
    # 3. Add all documents at once
    print(f"Adding {len(sample_documents)} documents to the FAISS index...")
    vdst.add_docs(sample_documents)
    print("Successfully indexed!\n")
    
    # 4. Test Query 1: Animal-focused question
    print("--- Testing Query 1 ---")
    query_1 = "Tell me about pets or animals"
    results_1 = vdst(query_1, k=2)
    for rank, res in enumerate(results_1, 1):
        print(f"Rank {rank} (Score: {res['score']:.4f}): {res['document']}")
        
    # 5. Test Query 2: Location-focused question
    print("\n--- Testing Query 2 ---")
    query_2 = "Where is the Eiffel Tower located?"
    results_2 = vdst(query_2, k=1)
    for rank, res in enumerate(results_2, 1):
        print(f"Rank {rank} (Score: {res['score']:.4f}): {res['document']}")
