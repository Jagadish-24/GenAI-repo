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
            processed_results.append({'test':doc,
                                      'page_number':metadata.get("page_number","?"),
                                      'distance':distance,
                                      'relevance_score':relevance,
                                      'adjusted_score':adjusted_score,
                                      'is_question':is_question,
                                      'has_answer':has_answer,
                                      'metadata':metadata})
        processed_results.sort(key=lambda x:x['adjusted_score'], reverse=True) #sort the chunks according to the newly calculated relevance scores
        results = processed_results[:top_k] #retun top_k chunks after doing the re-ranking
        print(f"Retrieved {len(results)} relvant chunks")
        return results
#the function which will be used to build the prompt to the LLM
def build_prompt(query:str,chunks:List[Dict[str,Any]]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks,1): #giving the chunks a number so that we can refer to them in the prompt, we start from 1
        text = chunk['text'].strip() #clean the text from the prompt
        source = f"[Source {i} - Page{chunk['page_number']}]"
        context_parts.append(f"{source}\n{text}\n")
    context = '\n'.join(context_parts)
    prompt = f'''You are a helpful assistant answering questions based ONLY on the provided context
    Context : {context}
    Question : {query}
    Instruction:
    1. Asnwer ONLY using information from the context above
    2. If the answer is not in the context, say "I cannot find this information in the document."
    3. Be concise and accurate
    4. If you use specific information, mention which source you used.
    ANSWER:'''
    return prompt
#the function which will generate the answer in natural language by interacting with the llm defined
def generate_answer(prompt:str,model:str=llm_model) -> str:
    print(f"Generating answer using {model}")
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role':'user',
                       'content':prompt}],
            options={'temperature':0.3,
                     'num_predict':500,
                     'stop':['\n\n\n']}
        )
        answer = response['message']['content'].strip()
        return answer
    except Exception as e:
        print(f"Failed to generate answer: {e}")
#function to save the produced results
def save_query_result(query:str,chunks:List[Dict[str,Any]],answer:str):
    result = {
        'query' : query,
        'timestamp':time.strftime("%Y-%m-%d %H:%M:%S"),
        'model':llm_model,
        'chunks_retrieved':len(chunks),
        'sources':[
            {
                'page':c['page_number'],
                'relevance' : c['relevance_score'],
                'text_preview': c['text'][:200]
            }
            for c in chunks
        ],
        'context' : '\n'.join([c['text'] for c in chunks]),
        'answer' : answer
    }
    safe_query = ''.join(c for c in query if c.isalnum() or c.isspace()).replace(' ','_')
    file_name = retrieval_dir/f"qa_{safe_query}_{time.strftime("%Y%m%d_%H%M%S")}.json"
    with open(file_name,"w",encoding='utf-8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
    print(f"\n Saved to {file_name}")
    return result
#the main function which connects the retrive - augment and generate process
def answer_question(collection)
        

            
