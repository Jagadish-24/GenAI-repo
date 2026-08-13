import os
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

#libraris for chromadb
import chromadb
from chromadb.config import Settings
data_dir = Path("data")
input_file = data_dir/"embedding.json"
database_dir = data_dir/"chroma_db"

collection_name = 'book_chunks' #collections store the embeddings - vector representations of the text data, documents - the original text content, metadat - source,timestamp etc

test_queries = ["What is natural language processing?", "How do transformers work?", 'What is machine learning?']

#initialize chromadb
def get_chroma_db_client():
    '''1. this function will initialize a chromadb client
    2. the chromadb stores the vectors in persistent/permananent directory so we dont need to perform embedding every time we restart the system'''
    try:
        client = chromadb.PersistentClient(path=str(database_dir),settings = Settings(anonymized_telemetry=False)) #disabling telemetry (connection to cloud) for privacy
    except ImportError:
        raise ImportError("Chromadb is not installed")

def get_or_create_collection(client):
    '''this function is the one which creates/fetches a collection
    collection is an equivalent to the "table" ins SQL, it holds all the chunks and their embeddings for a dataset'''
    try:
        collection = client.get_collection(collection_name)
        print(f'collection found : {collection_name} ')
        print(f"collection has {collection.count()} documents")
        return collection
    except Exception:
        print(f"Creating new collections : {collection_name}")
        collection = client.create_collection(name = collection_name, metadata={'description':"NLP book's chunks"})
        print('new collection created')
        return collection
def prepare_chunk_data(chunks:List[Dict[str,Any]]) -> Dict[str,Any]:
    '''Chromadb needs following to be added to the collection
    1. ids - unique identifiers for a each chunk
    2. embeddings - the vector representation of each chunk
    3. metadata - added info like page numbers
    4. documents - the actual text content itself'''
    ids = []; embeddings = []; metadatas=[]; documents = []
    for chunk in chunks:
        ids.append(chunk['chunk_id'])
        embeddings.append(chunk['embedding'])
        metadata = {'page_number':chunk['page_number'],
                    'chunk_type':chunk.get('chunk_type','unknown'),
                    'char_count' : chunk['char_count'],
                    'embedding_dim' : chunk.get('embedding_dim',0),
                    'embedding_method' : chunk.get('embedding_method','unknown')
                    }
        metadatas.append(metadata)

        documents.append(chunk['text'])
    print(f"Prepared {len(ids)} chunks for insertion")
    return {
        'ids':ids,
        "embeddings" : embeddings,
        'metadatas' : metadata,
        'documents' : documents
    }
def add_to_collection(collection,data:Dict[str,Any]):
    '''Add the prepared data to the chromadb persistent client'''
    print("Adding the prepared chunks into the database")
    existing_count = collection.count()
    if existing_count > 0:
        try:
            existing_ids = collection.get()['ids']
            if existing_ids:
                collection.delete(ids=existing_ids) #deleting the existing data to avoid duplicates
                print("Existing ids deleted")
        except Exception as e:
            print(f"Could not delete existing data : {e}")
    batch_size = 100 # the data is added to the vector db in bathces to avoid memory issues
    total_batches = 
           