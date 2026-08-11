import os
import json
import sys
from pathlib import Path

import pymupdf

pdf_file_name = "nlp-book.pdf"

data_dir = Path("data")
output_file = data_dir/"extracted.json"

if not os.path.exists(pdf_file_name):
    raise FileNotFoundError(f"PDF file not found : {pdf_file_name}")

with pymupdf.open(pdf_file_name) as doc:
    total_pages = len(doc)
    print(f"Total Number of pages in {pdf_file_name} is {total_pages}")

    #container for extracted data
    extracted_data = []

    #iterate over every page
    for page_idx in range(total_pages):
        #get each page as an object
        page = doc[page_idx]

        raw_text = page.get_text() #extract text using .get_text() method - all text as string, preserve paragraph structure but may include headers/footers

        page_data = {"page_number" : page_idx + 1,
                     "raw_text" : raw_text,
                     "charecter_count" : len(raw_text),
                     "has_text": bool(raw_text.strip())
        }

        extracted_data.append(page_data)

        if (page_idx+1) % 50 == 0:
            print(f" Extracted page {page_idx +1}/{total_pages}")
            

        print(f"Extraction complete!")
        # print(extracted_data)



total_pages = len(extracted_data)

pages_with_text = sum(1 for page in extracted_data if page["has_text"])
pages_without_text = total_pages - pages_with_text

total_chars = sum(page["charecter_count"] for page in extracted_data)
avg_chars = total_chars/total_pages if total_pages > 0 else 0

print(f"total_pages : {total_pages}\npages_with_text : {pages_with_text}\npages_without_text : {pages_without_text}\navg_characters : {avg_chars}")

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(extracted_data,file,indent=4,ensure_ascii=False)



