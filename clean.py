import os
import json
import re
import sys
from pathlib import Path
from collections import Counter

data_dir = Path("data")
input_file = data_dir/"extracted.json"
output_file = data_dir/"cleaned.json"

patterns_to_remove = {
    "page_numbers" : [r'(?i)page\s*\d+', r'^\s*\d+\s*[\|\-]',r'[\|\-]\s*\d+\s*[\|\-]',r'^\s*\d+\s*/\s*\d+\s*$'],
    'headers_footers': [
        r'^.*?(?:Chapter|CHAPTER)\s+\d+.*?$',  # Chapter headers
        r'^.*?(?:Part|PART)\s+\w+.*?$',        # Part headers
        r'^.*?(?:CONTENTS|INDEX|APPENDIX).*?$', # Table of contents markers
        r'^\s*(?:Copyright|©).*$',             # Copyright notices
        r'^\s*(?:All rights reserved).*$',      # Legal text
    ],
    'whitespace': [
        r'\n\s*\n\s*\n',             # 3+ newlines -> 2 newlines
        r'[ \t]+',                   # Multiple spaces/tabs -> single space
    ],
    'ocr_artifacts': [
        r'[^\x00-\x7F]+',            # Non-ASCII characters (may be unwanted)
        # Be careful with this one - some books legitimately use non-ASCII
    ],
    
    # Line numbers (often in poetry or code)
    'line_numbers': [
        r'^\s*\d+\s+',               # Line numbers at start of line
    ],   
}

def remove_patterns(text,patterns, pattern_name = "pattern"):
    orignal_text_len = len(text)
    cleaned_text = text

    for pattern in patterns:
        try:
            cleaned_text = re.sub(pattern,"",cleaned_text,flags=re.MULTILINE)
        except re.error as e:
            print(f"Invalid regex {pattern} : {e}")
    return cleaned_text


def clean_page(page_data):
    cleaned_data = page_data.copy()
    text = page_data['raw_text']

    original_length = len(text)

    text = remove_patterns(text,patterns_to_remove['page_numbers'],"page_numbers")
    text = remove_patterns(text,patterns_to_remove['headers_footers'],'header/footer')
    text = remove_patterns(text,patterns_to_remove['line_numbers'],"line_numbers")

    text = re.sub(r"\n\s*\n\s*\n",'\n\n',text)

    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    text = '\n'.join(lines)

    text = text.strip()

    cleaned_data['cleaned_text'] = text
    cleaned_data['original_length'] = original_length
    cleaned_data['cleaned_length'] = len(text)
    cleaned_data['removed_chars'] = original_length - len(text)

    return cleaned_data

def detect_repeated_patterns(data, min_occurences = 10):

    all_text = ' '.join(page['raw_text'] for page in data)
    lines = all_text.split('\n')

    line_counter = Counter()
    for line in lines:
        line = line.strip()
        if line and len(line) > 5:
            line_counter[line] += 1

    repeated = {
        line: count for line, count in line_counter.items() if count >= min_occurences
    }

    if repeated:
        print(f"\nFound {len(repeated)} patterns apperaing {min_occurences}+ times")
        for line, count in sorted(repeated.items(), key = lambda x: x[1], reverse=True)[:5]:
            display_line = line[:50] + "..." if len(line) > 50 else line
            print(f" '{display_line}' appears {count} times")
    else:
        print("No major repeating patterns detected")
    return repeated

if not input_file.exists():
    raise FileNotFoundError(f"Input File : {input_file} not found")

with open(input_file,'r',encoding='utf-8') as f:
    data = json.load(f)




print(f"Loading input json file {input_file}")

if not input_file.exists():
    raise FileNotFoundError(f"Input File : {input_file} not found")

with open(input_file,'r',encoding='utf-8') as f:
    data = json.load(f)
print(f"Loaded {len(data)} pages")

cleaned_data = []
total_removed = 0
pages_with_removal = []

for idx, page in enumerate(data):
    cleaned_page = clean_page(page)
    cleaned_data.append(cleaned_page)

    if cleaned_page['removed_chares'] > 0:
        total_removed += cleaned_page['removed_chars'] #add the page wise omits to the total numbers
        pages_with_removal += 1 # add to count of pages with some removed content

    if (idx + 1) % 50 == 0:
        print(f"Cleaned page {idx+1}/{len(data)}") # show progress which page is running

with open(output_file,"w",encoding="utf-8") as f:
    json.dump(data,f,indent=2,ensure_ascii=True)

print(f"saved_successfully\n{total_removed}")
    
    