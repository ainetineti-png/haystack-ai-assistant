#!/usr/bin/env python3
"""
Debug script to test PDF parsing for specific problematic files
"""
import PyPDF2
import pdfplumber
import os

def test_pypdf2(pdf_path):
    """Test PyPDF2 parsing"""
    print(f"\n=== Testing PyPDF2 on {pdf_path} ===")
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            print(f"Number of pages: {len(reader.pages)}")
            
            content = ""
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    content += page_text + "\n"
                    print(f"Page {i+1}: {len(page_text)} characters extracted")
                    if i == 0:  # Show first page preview
                        print(f"First page preview: {page_text[:200]}...")
                except Exception as e:
                    print(f"Error on page {i+1}: {e}")
            
            print(f"Total content length: {len(content)} characters")
            return content
    except Exception as e:
        print(f"PyPDF2 failed: {e}")
        return None

def test_pdfplumber(pdf_path):
    """Test pdfplumber parsing"""
    print(f"\n=== Testing pdfplumber on {pdf_path} ===")
    try:
        content = ""
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Number of pages: {len(pdf.pages)}")
            
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
                        print(f"Page {i+1}: {len(page_text)} characters extracted")
                        if i == 0:  # Show first page preview
                            print(f"First page preview: {page_text[:200]}...")
                    else:
                        print(f"Page {i+1}: No text extracted")
                except Exception as e:
                    print(f"Error on page {i+1}: {e}")
            
            print(f"Total content length: {len(content)} characters")
            return content
    except Exception as e:
        print(f"pdfplumber failed: {e}")
        return None

def main():
    # Test the specific problematic PDF
    pdf_path = "backend/data/kebo1dd/kebo115.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return
    
    print(f"File size: {os.path.getsize(pdf_path)} bytes")
    
    # Test both libraries
    pypdf2_content = test_pypdf2(pdf_path)
    pdfplumber_content = test_pdfplumber(pdf_path)
    
    # Compare results
    print(f"\n=== Comparison ===")
    print(f"PyPDF2 success: {pypdf2_content is not None}")
    print(f"pdfplumber success: {pdfplumber_content is not None}")
    
    if pypdf2_content and pdfplumber_content:
        print(f"PyPDF2 length: {len(pypdf2_content)}")
        print(f"pdfplumber length: {len(pdfplumber_content)}")
        print(f"Content match: {pypdf2_content.strip() == pdfplumber_content.strip()}")

if __name__ == "__main__":
    main()
