import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # so we can import the connector

from connector.drive_connector import authenticate, list_subfolders, list_files_in_folder, get_file_text, PARENT_FOLDER_ID
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits a long document into smaller overlapping pieces.
    chunk_size = roughly how many characters per chunk
    overlap = how many characters repeat between chunks, so we don't
              accidentally cut a sentence in half and lose its meaning
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]  # drop any empty chunks

def extract_scope(text):
    """Pulls the SCOPE line out of a document's text, e.g. 'Zone=Americas | Sensitivity=General | Min-Role=Associate'"""
    for line in text.split('\n'):
        if line.strip().startswith('SCOPE:'):
            return line.strip().replace('SCOPE:', '').strip()
    return 'Zone=Unknown | Sensitivity=Unknown | Min-Role=Unknown'

def run():
    print("Step 1: Pulling documents from Drive (reusing the connector)...")
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)
    zones = list_subfolders(service, PARENT_FOLDER_ID)

    all_chunks = []
    for zone in zones:
        files = list_files_in_folder(service, zone['id'])
        for f in files:
            text = get_file_text(service, f['id'], f['mimeType'])
            scope = extract_scope(text)
            pieces = chunk_text(text)
            print(f"  {f['name']}: split into {len(pieces)} chunks")
            for i, piece in enumerate(pieces):
                all_chunks.append({
                    'doc_name': f['name'],
                    'zone': zone['name'],
                    'scope': scope,
                    'chunk_index': i,
                    'text': piece,
                })

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")

    print("\nStep 2: Loading the embedding model (first run downloads it, ~90MB, one time only)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, good enough for this project

    print("Step 3: Converting each chunk into its meaning fingerprint...")
    texts = [c['text'] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"\nDone. Each chunk is now a vector of {embeddings.shape[1]} numbers.")
    print(f"Example — chunk 0 is from '{all_chunks[0]['doc_name']}':")
    print(f"  Text preview: {all_chunks[0]['text'][:150]!r}")
    print(f"  First 8 numbers of its embedding: {embeddings[0][:8]}")

    return all_chunks, embeddings

if __name__ == '__main__':
    run()