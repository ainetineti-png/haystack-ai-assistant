# Data Directory

This directory contains your knowledge base documents for the RAG system.

## Usage
- Place your `.txt` files here
- The backend automatically loads all text files from this directory on startup
- Use the `/ingest` endpoint to reload documents without restarting the server

## File Formats Supported
- `.txt` files (UTF-8 encoded)

## Example
Add files like:
- `programming_basics.txt`
- `company_docs.txt`
- `faq.txt`

The system will automatically index and make them searchable.
