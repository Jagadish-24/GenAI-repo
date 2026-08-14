import os
import json
import sys
import time
from pathlib import Path
from typing import List,Dict, Any
import re
#libraries for retreival
import chromadb
from chromadb.config import Settings
import ollama

#setting different paths
data_dir = Path('data')
db_dir = data_dir/'chroma_db'
collection_name = 'book_chunks'
retrieval_dir = data_dir/'retreival_results'

#used models
text_embedding_model = 'nomic-embed-text'
llm_model = 'phi3:mini'

#retrieval settings
top_k_retrievals = 5 #top 5 chunks will be retrieved and be used to generate the llm model response

#create a folder for retrieval if not exists
retrieval_dir.mkdir(parents=True,exist_ok=True)

#function for connecting to database
def get_chroma_client():
    try:
        client = chromadb.PersistentClient(path=str(db_dir),settings=Settings(anonymized_telemetry=False))
        return client
    except Exception as e:
        print("Could not connect to chormadb client")
#function to get the collection from the database
def get_collection(client):
    try:
        collection = client.get_collection(collection_name)
        print(f"Connected to {collection_name}")
        print(f"Collection contains {collection.count()} chunks/documents")
        return collection
    except Exception as e:
        print(f"Collection not found sorry")
#function to take the query from user and convert it into embedding for retrieval
def get_query_embedding(query_text:str) -> List[float]:
    try:
        response = ollama.embeddings(model=text_embedding_model,prompt=query_text)
        return response['embedding']
    except Exception as e:
        print(f"Failed to generate embedding for this query {query_text}")
#function to perform the chunk retrieval from the database
def retrieve_chunks(collection,query:str,top_k int = top_k_retrievals) -> List[Dict[str,Any]]:
    query_embedding = get_query_embedding(query)
    results = collection.query(query_embeddings = [query_embedding],n_results=top_k*2) #here we get twice the required amount of chunks because we can use to re-rank them as needed later
    processed_results = []
    if results and results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] if 'distances' in results else None
            relevance = 1 - distance if distance else None
            is_question = False #since the document is in the form of question and answer, there is a possiblity that the retrieval will retreive he question chunk rather than the answer so we have bool which updates when we see a question in the retrived chunk and then we demote it in rank
            has_answer = False
            #check if it is a question
            if doc.strip().stratswith(('Q','Q:','Qn','Question','Question:',QUESTION','QUESTION:))
                is_question = True
            if re.search(r'Answer:|A:|Solution:|Steps:|'):
                has_answer = True
            adjusted_score = relevance if relevance else 0
#here we are increasing the score of anything that looks like an answer and also does not look like a question
            if has_answer:
                adjusted_score += 0.15
            if not is_question:
                adjusted_score += 0.10
#again we are penalizing the score anything that is a question and not an answer
            if is_question and not has_answer:
                adjusted_score -= 0.10
            processed_results.append({'test':doc,'page_number':metadata.get("page_number")})
            
