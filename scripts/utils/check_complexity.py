"""Scan the thesis docx for over-complicated paragraphs, dense jargon, and overly verbose sections."""

import sys
from pathlib import Path
from scripts.utils.audit_docx import extract_text_from_docx
import re

def scan_complexity(docx_path):
    paragraphs = extract_text_from_docx(docx_path)
    print(f"Total paragraphs analyzed: {len(paragraphs)}\n")
    
    # Track sections
    current_section = "Front Matter"
    section_counts = {}
    long_paras = []
    dense_jargon_paras = []
    
    jargon_terms = [
        "heteroscedasticity", "stochastic gradient descent", "hessian matrix",
        "second-order approximation", "monotonicity constraint", "piecewise constant",
        "hyperplane", "orthogonal", "eigenvalue", "diffeomorphism"
    ]
    
    for idx, p in enumerate(paragraphs):
        # Detect section headers
        if re.match(r'^(Chapter|Section|\d+\.\d+|\b[A-Z\s]{5,}\b)', p.strip()) and len(p.strip()) < 80:
            if "CHAPTER" in p.upper() or "ABSTRACT" in p.upper() or "INTRODUCTION" in p.upper() or "METHODOLOGY" in p.upper() or "RESULTS" in p.upper() or "DISCUSSION" in p.upper():
                current_section = p.strip()
                
        words = p.split()
        if len(words) > 120:
            long_paras.append((idx, current_section, len(words), p[:150]))
            
        # Check for heavy academic jargon that might confuse business panelists
        found_jargon = [term for term in jargon_terms if term in p.lower()]
        if found_jargon:
            dense_jargon_paras.append((idx, current_section, found_jargon, p[:150]))

    print(f"=== 1. VERY LONG PARAGRAPHS (>120 words) ===")
    print(f"Found {len(long_paras)} very dense paragraphs.")
    for idx, sec, wcount, snippet in long_paras[:10]:
        safe_snippet = snippet.encode('ascii', 'ignore').decode('ascii')
        safe_sec = sec.encode('ascii', 'ignore').decode('ascii')
        print(f"  [Para {idx} | {safe_sec} | {wcount} words]: {safe_snippet}...")
        
    print(f"\n=== 2. DENSE STATISTICAL JARGON CHECK ===")
    print(f"Found {len(dense_jargon_paras)} paragraphs with heavy theoretical math jargon.")
    for idx, sec, terms, snippet in dense_jargon_paras[:10]:
        safe_snippet = snippet.encode('ascii', 'ignore').decode('ascii')
        safe_sec = sec.encode('ascii', 'ignore').decode('ascii')
        print(f"  [Para {idx} | {safe_sec} | Terms: {terms}]: {safe_snippet}...")
        
    # Check specific complex chapters
    print("\n=== 3. CHAPTER EXCERPT INSPECTION ===")
    method_samples = [p for p in paragraphs if "isotonic regression" in p.lower() or "lightgbm" in p.lower()]
    print(f"Sample LightGBM / Isotonic explanations ({len(method_samples)} found):")
    for p in method_samples[:3]:
        safe_p = p[:200].encode('ascii', 'ignore').decode('ascii')
        print(f"  -> {safe_p}...\n")

if __name__ == "__main__":
    path = r"d:\PythonProject1\A Strategic Business Intelligence Approach to Predicting Hotel Booking Cancellations.docx"
    scan_complexity(path)
