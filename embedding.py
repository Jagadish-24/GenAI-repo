import os
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

#libraries for ollama based embedding generation
import requests
#libraries for tf-idf based method
from sklearn.feature_extraction.text import TfidfVectorizer

data_dir = Path("data")
input_file = data_dir/"chunked.json"
output_file = data_dir/"embeddings.json"

embedding_method = 'ollama' #to use if ollama is available
#embedding_method = 'tfidf' #to use if ollama is not available

#ollama settings
ollama_model = 'nomic-embed-text' #small embedding model & faster
# ollama_model = 'mxbai-embed-large'#larger & slower model

#tf-idf settings
max_features = 5000 #maximum number of 'features' for TF-IDF - features are the unique words (in other words n-grams) extracted from the text

def get_ollama_embedding(text:str,model:str=ollama_model) -> List[float]:
    '''function to get the embedding from ollama server
    before running in a separate terminal run 
    1. ollama serve
    2. ollama pull nomi-embed-text'''
    try:
        url = "http://localhost:11434/api/embeddings"
        payload = {
            'model':model,
            "prompt":text,
        }
        response = requests.post(url,json=payload,timeout=30)
        response.raise_for_status() #raises a flag if there is any 4## or 5## error, if its good no flag is raised

        data = response.json()
        embedding = data.get('embedding',[])

        if not embedding:
            raise ValueError(f"The embedding returned for {model} is empty")

        return embedding
    except ImportError:
        raise ImportError("requests library not installed")
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Ollama server not running or is not installed")
    except Exception as e:
        raise Exception(f"Failed to get embedding")
def generate_ollama_embeddings(chunks:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    '''using above function to generate ollama_embedding for all chunks'''
    print(f"using ollama model {ollama_model}")
    embedded_chunks = []
    total_chunks = len(chunks)
    for idx,chunk in enumerate(chunks):
        try:
            embedding = get_ollama_embedding(chunk['text']) #get the embedding for the text contained in the chunk
            embedded_chunk = chunk.copy() #creating a copy of the chunk itself so we preserve the structure and add other imp things like embedding, dimensions added to it
            embedded_chunk['embedding'] = embedding
            embedded_chunk['embedding_dimesion'] = len(embedding)
            embedded_chunk['embedding_method'] = 'ollama'
            embedded_chunk['embedding_model'] = ollama_model
            embedded_chunks.append(embedded_chunk)
            #progress indiaction
            if (idx+1) % 10 == 0 or (idx+1) == total_chunks:
                print(f"Embedding completed for {idx+1}/{total_chunks} chunks")
        except Exception as e:
            print(f"Falied embedding {idx} page {chunk['page_number']} : {e}")
            continue
    print(f"embedding successful")
    return embedded_chunks
#generating embeddings using tf-idf
def generate_tfidf_embeddings(chunks: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    '''this function to generate embeddings using tf-idf method - pure keyword search'''
    texts = [chunk['text'] for chunk in chunks]
    print(f"Training TF-IDF on {len(texts)} documents")
    #initialize the vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words = 'english',
        lowercase = True,
        strip_accents = 'unicode'
    )
    tfidf_matrix = vectorizer.fit_transform(texts) #fit transform does do things 1. learns the vocabulary and the IDF weigths from the input texts and converts the text into tf-idf feature vectors based on its learning
    print(f"tf-idf matrix shape = {tfidf_matrix.shape}")
    print(f"Vocabulary size of the document is {len(vectorizer.get_feature_names_out())}")
    embedded_chunks = []
    for idx, chunk in enumerate(chunks):
        #get the tf-idf vector for this chunk
        embedding = tfidf_matrix[idx]
        #convert it to an array
        embedding = embedding.toarray()
        #convert the 2d array from previous step to a single dimensional array 
        embedding = embedding.flatten()
        #convert the flattened array to a list
        embedding = embedding.tolist()
        embedded_chunk = chunk.copy() #creating a copy of the chunk itself so we preserve the structure and add other imp things like embedding, dimensions added to it
        embedded_chunk['embedding'] = embedding
        embedded_chunk['embedding_dimesion'] = len(embedding)
        embedded_chunk['embedding_method'] = 'tfidf'
        embedded_chunk['tfidf_vocab_size'] = len(vectorizer.get_feature_names_out())
        embedded_chunks.append(embedded_chunk)

        if (idx+1) % 50 == 0:
            print(f"Embedding {len(embedded_chunks)} chunks using TF-IDF" )
    print("Chunk embedding done")
    return embedded_chunks
#embedding generator with the method as an input
def generate_embedding(chunks:List[Dict[str,Any]],method : str = 'ollama') -> List[Dict[str,Any]]:
    '''The actual function which will be generating the embedding based on the specified method'''
    if not chunks:
        print("No chunks !")
        return []
    print(f"Starting the embedding for {len(chunks)}")
    #getting total text size
    total_chars = sum(len(c['text']) for c in chunks)
    print(f"Total charecters embed = {total_chars}")
    start_time = time.time()
    if method == 'ollama':
        embedded = generate_ollama_embeddings(chunks)
    elif method == 'tfidf':
        embedded = generate_tfidf_embeddings(chunks)
    else:
        raise ValueError(f"Invalid Method {method} chosen")
    elapsed = time.time() - start_time
    print(f"elapsed time = {elapsed:.2f}")
    return embedded
def validate_embeddings(embedded_chunks: List[Dict[str,Any]]) -> Dict[str,Any]:
    '''fucntion to check if my embedding are generated correctly or not'''
    if not embedded_chunks:
        return{'valid':False,'error':"No embedded_chunks"}

    #checking the first chunk
    first_chunk = embedded_chunks[0]
    if 'embedding' not in first_chunk:
        return {'valid':False, "error" : "No embedding found"}
    if not first_chunk['embedding']:
        return {'valid':False,'error':'Empty embedding'}
    embedding_dim = len(first_chunk['embedding']) #getting the dimension of the fisrt embedding to check if all chunks have uniform embedding length despite different chunk length
    for chunk in embedded_chunks:
        if len(chunk['embedding']) != embedding_dim:
            return {
                'valid':False,
                'error': f"Inconsistent dimensions : {len(chunk['embedding'])} vs {embedding_dim}"
            }
    embeddings = [c['embedding'] for c in embedded_chunks]
    embed_Array = np.array(embeddings)

    return {
        'valid' : True,
        'number_of_chunks' : len(embedded_chunks),
        'embedding_dim' : embedding_dim,
        'embedding_method' : embedded_chunks[0].get('embedding_method','unknown'),
        'embedding_mean' : embed_Array.mean(),
        'embedding_std' : embed_Array.std(),
        'embedding_min' : embed_Array.min(),
        'embedding_max' : embed_Array.max(),
    }

#executing the embedding generation
try:
    print(f"Loading Chunks from : {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(f"File not found : {input_file}")
    with open(input_file,'r',encoding='utf-8') as file:
        chunks = json.load(file)
    print(f"Loaded {len(chunks)} chunks")
    embedded_chunks = generate_embedding(chunks,method=embedding_method)
    print("Validating the generated embeddings")
    validation = validate_embeddings(embedded_chunks)
    if not validation['valid']:
        print(f"Validation failed : {validation.get('error','unknown error')}")
    print("validation passed")
    with open(output_file,'w',encoding='utf-8') as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)
    print("EMBEDDING COMPLETE")
except FileNotFoundError as e:
    print("File Not Found")
except json.JSONDecodeError as e:
    print("Invalid JSON input")
