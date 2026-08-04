import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from connector.drive_connector import authenticate, list_subfolders, list_files_in_folder, get_file_text, PARENT_FOLDER_ID
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
import chromadb

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]

def extract_scope(text):
    for line in text.split('\n'):
        if line.strip().startswith('SCOPE:'):
            return line.strip().replace('SCOPE:', '').strip()
    return 'Zone=Unknown | Sensitivity=Unknown | Min-Role=Unknown'

def parse_scope(scope_str):
    """Turns 'Zone=Americas | Sensitivity=General | Min-Role=Associate' into separate fields, so we can filter on each one individually later."""
    parts = {}
    for piece in scope_str.split('|'):
        if '=' in piece:
            key, val = piece.strip().split('=', 1)
            parts[key.strip()] = val.strip()
    return parts

def run():
    print("Step 1: Pulling documents from Drive...")
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)
    zones = list_subfolders(service, PARENT_FOLDER_ID)

    all_chunks = []
    for zone in zones:
        files = list_files_in_folder(service, zone['id'])
        for f in files:
            text = get_file_text(service, f['id'], f['mimeType'])
            scope = extract_scope(text)
            scope_parts = parse_scope(scope)
            pieces = chunk_text(text)
            for i, piece in enumerate(pieces):
                all_chunks.append({
                    'id': f"{f['id']}_chunk{i}",  # every chunk needs a unique ID
                    'doc_name': f['name'],
                    'zone': scope_parts.get('Zone', zone['name']),
                    'sensitivity': scope_parts.get('Sensitivity', 'Unknown'),
                    'min_role': scope_parts.get('Min-Role', 'Unknown'),
                    'chunk_index': i,
                    'text': piece,
                })

    print(f"Total chunks: {len(all_chunks)}")

    print("Step 2: Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Step 3: Generating embeddings...")
    texts = [c['text'] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print("Step 4: Saving everything into Chroma...")
    # This creates (or opens, if it already exists) a database saved permanently in the data/ folder
    client = chromadb.PersistentClient(path="data/chroma_store")
    # "get_or_create" means: reuse it if it exists, otherwise make it fresh
    collection = client.get_or_create_collection(name="policybot_docs")

    # If we've run this before, clear out old data first so we don't get duplicates
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])

    collection.add(
        ids=[c['id'] for c in all_chunks],
        embeddings=embeddings.tolist(),
        documents=[c['text'] for c in all_chunks],
        metadatas=[{
            'doc_name': c['doc_name'],
            'zone': c['zone'],
            'sensitivity': c['sensitivity'],
            'min_role': c['min_role'],
            'chunk_index': c['chunk_index'],
        } for c in all_chunks],
    )

    print(f"\nSaved {collection.count()} chunks into the Chroma database at data/chroma_store/")
    print("You can now inspect it by running: python indexing/peek_index.py")

if __name__ == '__main__':
    run()