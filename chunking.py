import os
import json
import re
import sys
from pathlib import Path
from collections import Counter

data_dir = Path("data")
input_file = data_dir/"cleaned.json"
output_file = data_dir/"chunked.json"

chunk_size = 800  # Number of characters per chunk
chunk_overlap = 150 #overlaps text between the chunks
min_chunk_size = 50 #to prevent very small chunks from entering the pipeline (e.g., headers)

def chunk_by_paragraphs(text:str,page_num:int):
    '''Paragraph level chunking
    '''
    chunks = [] #empty placeholder
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue #skipping empty paragraphs

        chunk_dict = {"text":para,
                      "page_number":page_num,
                      "chunk_type":"paragraph",
                      "char_count":len(para)}

        chunks.append(chunk_dict)

    return chunks

def chunk_recursive(text:str,page_num:int):
    '''Recursive Chunking - splits the text hierarchially based on semantic structures like paragraphs, sentence and/or words (source : my interpretation of gemini answer)
    Follows - TOP-DOWN FALLBACK LOOPS driven by ordered list of separator rules
    1. Instead of splitting at some fixed single charecter like "\n" i give it a list of seperators like newline, space etc in a ranked list based on the largest to the smallest strcutural boundary
    2. So for a given peice of text with a target chunk size "K", we check if the current text < k, if that is true then we take most important separator from the list that exists in the text and sepearet the text based on smaller segments
    2.1 once we have smaller segments, we combine the smaller segment into a single chunk with the condition that the chunk size should be less than k (this is done after applying a chunk overlap parameter)
    2.2 if any particular segment becomes larger than k, we go for the seperator in the next order and then follow the process.

    This is what is done in LangChain's RecursiceCharacterTextSplitter as well
    '''
    chunks = [] #placeholder
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunk = {
                'text':para,
                'page_number':page_num,
                'chunk_type':'paragraph',
                'char_count':len(para)
            }
            chunks.append(chunk)
        else:
            sentences = split_into_sentences(para)
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk: #check if adding the sentance makes the current chunk size exceed the defined chunk size.
                    chunk = {
                        'text':current_chunk.strip(),
                        "page_number":page_num,
                        "chunk_type":'sentence_group',
                        'char_count':len(current_chunk)
                    }
                    chunks.append(chunk)
                    if chunk_overlap > 0:
                        words = current_chunk.split()
                        overlap_words = min(10,len(words)) #take last 10 words or less if the chunk has less than 10 words
                        current_chunk = ' '.join(words[-overlap_words:]) + " " #get the last
                    else:
                        current_chunk = "" #reset the current chunk if no overlap is defined

                current_chunk += sentence + " "

            if current_chunk.strip(): #for the last chunk after the loop ends, we need to add it to the chunks list if it has any content
                chunk = {
                    "text" : current_chunk.strip(),
                    'page_number' : page_num,
                    "chunk_type" : 'sentence_group',
                    "char_count" : len(current_chunk)
                }
                chunks.append(chunk)
    return chunks

def split_into_sentences(text:str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+',text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text:str, page_num:int, strategy:str = 'recursive') -> list[dict[str,any]]:
    '''this is the function which choses the chunking strategy between recursive and normal'''
    if not text or not text.strip():
        return []
    if strategy == 'paragraph':
        chunks = chunk_by_paragraphs(text,page_num)
    else:
        chunks = chunk_recursive(text,page_num)
    chunks = [c for c in chunks if c['char_count'] >= min_chunk_size]
    return chunks
def process_page(page_data:dict[str,any],strategy: str = 'recursive') -> list[dict[str,any]]:
    '''purpose of this function is to chunk the cleaned text of a single page'''
    '''input is the page dictionary containing the "cleaned_text" and "page_number" keys'''

    page_num = page_data['page_number']
    text = page_data.get('cleaned_text')
    if not text or not text.strip(): #check wehter we have cleaned text in that page
        return []
    chunks = chunk_text(text,page_num,strategy) #perform the chunking using the strategy chooser function
    for chunk in chunks:
        chunk['chunk_id'] = None # assign placeholder to store the chunk_id of each chunk.
    return chunks
def analyse_chunks(chunks:list[dict[str, any]]) -> dict[str,any]:
    '''function to analyse chunk statistics'''
    if not chunks:
        return {"total_chunks":0}
    sizes = [c['char_count'] for c in chunks]
    chunk_statistics = {
        'total_chunks': len(chunks),
        'min_size_of_chunk' : min(sizes),
        'max_size_of_chunk' : max(sizes),
        'avg_size' : sum(sizes) / len(sizes),
        'total_chars':sum(sizes),
    }
    chunk_types = {}
    for c in chunks:
        ctype = c.get('chunk_type',"unknown")
        chunk_types[ctype] = chunk_types.get(ctype,0)+1
    chunk_statistics['chunk_types'] = chunk_types
    return chunk_statistics

#passing the input file
try:
    print(f"Loading the cleaned data from {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found {input_file}")
    with open(input_file,'r',encoding='utf-8') as file:
        pages = json.load(file)
    print(f"Loaded {len(pages)} pages")
#page tracking
    total_pages = len(pages)
    pages_with_texxt = sum(1 for p in pages if p.get('cleaned_text','').strip())
    print(f"Pages with text = {pages_with_texxt}")
#chunking each page
    print("Started chunking process")
    all_chunks = []
    chunks_per_page = []
    for idx, page in enumerate(pages):
        if not page.get('cleaned_text','').strip(): #if the page dict do not contains cleaned text
            chunks_per_page.append(0)
            continue
        #strating actual chunking
        page_chunks = process_page(page,strategy='paragraph')
        chunks_per_page.append(len(page_chunks))
        all_chunks.extend(page_chunks)
        if (idx + 1) % 50 == 0:
            print(f"Processed Page {idx+1}/{total_pages}") #progress update statement
    #assign global id to each chunk
    for idx, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = f"chunk_{idx:06d}"
    print(f"\n Created {len(all_chunks)} chunks from {pages_with_texxt} pages")
    stats = analyse_chunks(all_chunks)
    print(f'\nChunk Statistics\nTotal chunks = {stats["total_chunks"]}\nAvg Chunk size = {stats["avg_size"]}')
    with open(output_file,'w',encoding='utf-8') as file:
        json.dump(all_chunks,file,indent=2,ensure_ascii=False)
    print("chunking_complete")
except FileNotFoundError as e:
    print(f"\nError : {e}")
except json.JSONDecodeError as e:
    print("invalid json file")
except Exception as e :
    print(f"Unexpected error : {e}")
    import traceback
    traceback.print_exc()
        