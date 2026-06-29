"""Audit the final thesis docx file for consistency and leftover issues."""

import zipfile
import xml.etree.ElementTree as ET
import re
import sys
from pathlib import Path

def extract_text_from_docx(docx_path):
    with zipfile.ZipFile(docx_path) as z:
        xml_content = z.read('word/document.xml')
    tree = ET.fromstring(xml_content)
    
    # XML namespace for Word
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    
    paragraphs = []
    for p in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        texts = [node.text for node in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
        if texts:
            paragraphs.append(''.join(texts))
            
    return paragraphs

def audit_docx(docx_path):
    print(f"Auditing: {docx_path}")
    if not Path(docx_path).exists():
        print("Error: File not found!")
        return
        
    paragraphs = extract_text_from_docx(docx_path)
    full_text = "\n".join(paragraphs)
    print(f"Extracted {len(paragraphs)} paragraphs, total {len(full_text)} characters.\n")
    
    # Check 1: AI / forbidden words
    ai_matches = re.findall(r'\b(?:AI|inexplainable AI|artificial intelligence)\b', full_text, re.IGNORECASE)
    print("=== CHECK 1: 'AI' Terminology Check ===")
    if ai_matches:
        print(f"Found {len(ai_matches)} mentions of AI/Artificial Intelligence:")
        for idx, p in enumerate(paragraphs):
            if re.search(r'\b(?:AI|inexplainable AI|artificial intelligence)\b', p, re.IGNORECASE):
                print(f"  [Para {idx}] {p[:120]}...")
    else:
        print("PASSED: No forbidden 'AI' buzzwords found.")
        
    # Check 2: Placeholders / TODOs
    placeholders = re.findall(r'\[.*?\]|TODO|FIXME|XXXX|XX%|0\.000|Placeholder', full_text, re.IGNORECASE)
    # filter out normal citation brackets like [1], [2-5]
    real_placeholders = [ph for ph in placeholders if not re.match(r'^\[\d+([,\-\s]\d+)*\]$', ph)]
    print("\n=== CHECK 2: Placeholders / TODO Check ===")
    if real_placeholders:
        print(f"Found potential placeholders: {list(set(real_placeholders))[:15]}")
        for idx, p in enumerate(paragraphs):
            for ph in real_placeholders:
                if ph in p and not re.match(r'^\[\d+([,\-\s]\d+)*\]$', ph):
                    print(f"  [Para {idx}] {p[:120]}...")
                    break
    else:
        print("PASSED: No unaddressed placeholders found.")
        
    # Check 3: Data split phrasing
    print("\n=== CHECK 3: Data Split Phrasing Check ===")
    split_paras = [p for p in paragraphs if re.search(r'80/10/10|80/20|70/30|split|partition|holdout', p, re.IGNORECASE)]
    print(f"Found {len(split_paras)} paragraphs mentioning splits/partitions. Samples:")
    for p in split_paras[:10]:
        print(f"  -> {p[:140]}...")
        
    # Check 4: Key Metrics Check
    print("\n=== CHECK 4: Key Metrics Presence Check ===")
    metrics_to_check = {
        "PR-AUC (0.759)": r"0\.759",
        "ROC-AUC (0.863)": r"0\.863",
        "Optimal Threshold (0.06)": r"0\.06",
        "Test Cost (€71,135)": r"71,135",
        "Savings vs No Model (€2.25M)": r"2,251,658|2\.25",
        "Savings vs Baseline (€598k)": r"598,501|598",
        "p-value (0.177)": r"0\.177"
    }
    for label, pattern in metrics_to_check.items():
        if re.search(pattern, full_text):
            print(f"  [FOUND] {label}")
        else:
            print(f"  [MISSING] {label} not found in text.")

if __name__ == "__main__":
    path = r"d:\PythonProject1\A Strategic Business Intelligence Approach to Predicting Hotel Booking Cancellations.docx"
    audit_docx(path)
