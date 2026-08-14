import datetime
import os
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

#libraris for chromadb
import chromadb
from chromadb.config import Settings
#libraries for ollama
import ollama
data_dir = Path("data")
input_file = data_dir/"embeddings.json"
database_dir = data_dir/"chroma_db"
retrieve_dir = data_dir/"retrieved_chunks"

collection_name = 'book_chunks' #collections store the embeddings - vector representations of the text data, documents - the original text content, metadat - source,timestamp etc

test_queries = ["What is Confidentiality?", "What are Cybersecurity requirements at the supplier level?", 'What is Positioning technology?']

#initialize chromadb
def get_chroma_db_client():
    '''1. this function will initialize a chromadb client
    2. the chromadb stores the vectors in persistent/permananent directory so we dont need to perform embedding every time we restart the system'''
    try:
        client = chromadb.PersistentClient(path=str(database_dir),settings = Settings(anonymized_telemetry=False)) #disabling telemetry (connection to cloud) for privacy
    except ImportError:
        raise ImportError("Chromadb is not installed")
    return client
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
        collection = client.create_collection(name = collection_name, metadata={'description':"NLP book's chunks","hnsw:space": "cosine"})
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
        'metadatas' : metadatas,
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
    total_batches = (len(data['ids']) + batch_size - 1) // batch_size
    print(f"Adding data in {total_batches} batches of size {batch_size}")
    for i in range(0,len(data['ids']),batch_size):
        #we move every "batch_size = 100" chunks and add them to the vectordb
        batch_end = min(i+batch_size, len(data['ids'])) #sets the limit when i = 0 it will be 100, then i = 100 (note the step = 100), then batch_end = 200 = i=100 + bathc_szie = 100
        batch_ids = data['ids'][i:batch_end]
        batch_embeddings = data['embeddings'][i:batch_end]
        batch_metadatas = data['metadatas'][i:batch_end]
        batch_documents = data['documents'][i:batch_end]
        try: #adding the information into the collection
            collection.add(
                ids = batch_ids,
                embeddings = batch_embeddings,
                metadatas = batch_metadatas,
                documents = batch_documents
            )
            if (i + batch_size) % 500 ==0 or batch_end == len(data['ids']):
                #print something when every 500 chunks are added
                print(f"Added {batch_end}/{len(data['ids'])} chunks")
        except Exception as e:
            print(f"Failded to add batch starting at {i} : {e}")
            print("Trying to add them individually")
            for j in range(i,batch_end):
                try:
                    collection.add(
                        ids = [data['ids'][j]],
                        embeddings=[data['embeddings'][j]],
                        metadatas=[data['metadatas'][j]],
                        documents = [data['documents'][j]]
                    )
                except Exception as e2:
                    print(f"Failed to add chunk {j} : {e2}")
    final_count = collection.count()
    print(f"Database now has {final_count} chunks")
#creating a simple function to test retrieval from the database
def get_query_embedding(query_text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query_text
    )
    return response["embedding"]
def test_retrieval(collection, query_text:str,n_results:int=3,save_retrieved:bool=True):
    '''This code will test the database by retrieving some chunks for a query'''
    print(f"Test Query = {query_text}")
    try:
        '''doing a similarity search
        there are three ways of doing this:
        1. Cosine Similarity = cos(theta) = DotProduct(vector_A, vector_B) / Mag(vector_A) * Mag(vector_b), from this the codine distance = 1 - cosine similarity is found
            1.1 is cosine similaroty = 1 ==> theta = 0 deg, the vectors point in the same exact direction, They are perfectly similar
            1.2 cosine similaroty = 0 ==> theta = 90 deg the vectors are perpendicular, they share no similarity hence are independent of each other
            1.3 cosine similaroty = -1 ==> theta = 180 deg the vectors are in opposite directions, they are perfectly dissimilar'''
        '''to add later about square l2 and iiner product distance'''
        query_embedding = get_query_embedding(query_text)
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        retrieval_results = {
            'query': query_text,
            # 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'n_results': n_results,
            'results': []
        }
        if results and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else None
                print(f"Result {i+1} Page {metadata.get('page_number',"?")} : ")
                print(f"Relevance score = {1 - distance if distance else 'NA'}")
                print(f"Distance = {distance:.2f}")

                preview = doc[:200].replace('\n',"")
                if len(doc) > 200:
                    preview += "..."
                print(f"Preview : {preview}")

                retrieval_results['results'].append({
                    'rank': i+1,
                    'page_number': metadata.get('page_number',"?"),
                    'chunk_type': metadata.get('chunk_type','unknown'),
                    'char_count': metadata.get('char_count',0),
                    'distance': distance,
                    'relevance_score': 1 - distance if distance else None,
                    'preview': preview,
                    'full_text': doc,
                    'metadata': metadata
                    
                })
            if save_retrieved:
                retrieve_dir.mkdir(parents=True, exist_ok=True)
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_query = query_text.replace(" ", "_").replace("?", "").replace("/", "_")
                output_file = retrieve_dir/f"retrieved_{safe_query}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(retrieval_results, f, indent=2)
                print(f"Saved retrieved results to: {output_file}")
        else:
            print("No results found for the query")
    except Exception as e:
        print(f"Failed to retrieve results for the query : {e}")

print(f"Loading the data from {input_file}")

if not input_file.exists():
    raise FileNotFoundError(f"Input file {input_file} does not exist. Please run the embedding script first.")
with open(input_file,'r',encoding='utf-8') as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks from the input file")
if chunks and 'embedding' not in chunks[0]:
    print("The loaded chunks do not contain embeddings. Please run the embedding script first.")
print("Initializing the chromadb client")
client = get_chroma_db_client()
collection = get_or_create_collection(client)
data = prepare_chunk_data(chunks)
add_to_collection(collection,data)
for query in test_queries:
    test_retrieval(collection,query,n_results=3)
print("\nDatabase Statistics : ")
print(f"Total chunks in database: {collection.count()}")
print(f"Collection name: {collection_name}")
print(f"Database location: {database_dir}")
sample = collection.get(limit=5)
if sample and sample['metadatas']:
    print(f"Sample page numbers = {[m.get('page_number','?') for m in sample['metadatas'][:5]]}")
    summary_file = data_dir/"db_summary.json"
    summary = {
            "collection_name": collection_name,
            "total_documents": collection.count(),
            "database_location": str(data_dir),
            "embedding_dimension": len(chunks[0]['embedding']) if chunks else 0,
            "test_queries": test_queries,
            "creation_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_file}")
