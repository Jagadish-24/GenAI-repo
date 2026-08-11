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
    chunks = [] #empty placeholder
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        para = para.strip()
        if not in para: