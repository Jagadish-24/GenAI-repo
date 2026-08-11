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
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunk = {
                        'text':current_chunk.strip(),
                        "page_numner":page_num,
                        "chunk_type":'sentence_group',
                        'char_count':len(current_chunk)
                    }
                    chunks.append(chunk)
                    if current_chunk:
                        overlap_text = current_chunk.split('.')[-1].strip() 
                        current_chunk = overlap_text + ". " if overlap_text else ""

                current_chunk += sentence + " "

                if current_chunk.strip():
                    chunk = {
                        "text" : current_chunk.strip(),
                        'page_number' : page_num,
                        "chunk_type" : 'sentence_group',
                        "char_count" : len(current_chunk)
                    }
                    chunks.append(chunk)
    return chunks

def split_into_sentences(text:str) -> list[str]:
    sentences = re.split(r'?<=[.!?])\s+',text)
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