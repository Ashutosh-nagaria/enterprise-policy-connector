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

def parse_scope(scope_str):
    """Turns 'Zone=Americas | Sensitivity=General | Min-Role=Associate' into separate fields."""
    parts = {}
    for piece in scope_str.split('|'):
        if '=' in piece:
            key, val = piece.strip().split('=', 1)
            parts[key.strip()] = val.strip()
    return parts

def split_by_role_section(text):
    """
    Finds the Associates/Senior/Executive headings in the document and splits
    the text into labeled blocks. Anything before the first heading (Purpose,
    contacts, etc) is returned with role=None, meaning 'use the document's
    own SCOPE tag' instead of a section-specific one.
    """
    role_headers = [
        ('Associate', 'Associates ('),
        ('Senior', 'Senior ('),
        ('Executive', 'Executive ('),
    ]
    markers = []
    for role, heading in role_headers:
        idx = text.find(heading)
        if idx != -1:
            markers.append((idx, role))
    markers.sort()

    if not markers:
        return [(None, text)]  # no role sections found, whole doc uses SCOPE's tag

    sections = []
    if markers[0][0] > 0:
        sections.append((None, text[:markers[0][0]]))  # preamble before first heading
    for i, (idx, role) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        sections.append((role, text[idx:end]))
    return sections

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
            scope_parts = parse_scope(scope)
            doc_level_role = scope_parts.get('Min-Role', 'Unknown')

            sections = split_by_role_section(text)
            chunk_counter = 0
            for section_role, section_text in sections:
                effective_role = section_role if section_role else doc_level_role
                pieces = chunk_text(section_text)
                for piece in pieces:
                    all_chunks.append({
                        'doc_name': f['name'],
                        'zone': zone['name'],
                        'scope': scope,
                        'min_role': effective_role,
                        'chunk_index': chunk_counter,
                        'text': piece,
                    })
                    chunk_counter += 1
            print(f"  {f['name']}: split into {chunk_counter} chunks across {len(sections)} role section(s)")

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")

    print("\nStep 2: Loading the embedding model (first run downloads it, ~90MB, one time only)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, good enough for this project

    print("Step 3: Converting each chunk into its meaning fingerprint...")
    texts = [c['text'] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"\nDone. Each chunk is now a vector of {embeddings.shape[1]} numbers.")
    print(f"Example — chunk 0 is from '{all_chunks[0]['doc_name']}' (min_role: {all_chunks[0]['min_role']}):")
    print(f"  Text preview: {all_chunks[0]['text'][:150]!r}")
    print(f"  First 8 numbers of its embedding: {embeddings[0][:8]}")

    return all_chunks, embeddings

if __name__ == '__main__':
    run()