import chromadb
from sentence_transformers import SentenceTransformer

# --- Define our demo personas ---
# Each persona has a zone (where they work) and a role level.
# Role levels: higher number = more access.
ROLE_LEVELS = {'Associate': 1, 'Senior': 2, 'Executive': 3}

PERSONAS = {
    'pam': {'name': 'Pam Beesly', 'zone': 'Americas', 'role': 'Associate'},
    'jan': {'name': 'Jan Levinson', 'zone': 'Americas', 'role': 'Senior'},
    'robert': {'name': 'Robert California', 'zone': 'Global', 'role': 'Executive'},  # CFO, sees everything
    'oscar': {'name': 'Oscar Martinez', 'zone': 'APAC', 'role': 'Senior'},
}

def is_allowed(persona, chunk_metadata):
    """
    This is the actual permission check.
    A persona can see a chunk only if:
    1. Their zone matches the chunk's zone (or they're 'Global', like the CFO)
    2. Their role level is high enough for the chunk's minimum required role
    """
    zone_ok = (persona['zone'] == 'Global') or (persona['zone'] == chunk_metadata['zone'])
    persona_role_level = ROLE_LEVELS.get(persona['role'], 0)
    required_role_level = ROLE_LEVELS.get(chunk_metadata['min_role'], 1)
    role_ok = persona_role_level >= required_role_level
    return zone_ok and role_ok

def search(query_text, persona_key, top_n=3):
    persona = PERSONAS[persona_key]
    print(f"\n=== Query: \"{query_text}\" ===")
    print(f"Asked by: {persona['name']} ({persona['zone']} / {persona['role']})\n")

    # Step 1: turn the question into a meaning-fingerprint, same as we did for the documents
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query_text])[0]

    # Step 2: search Chroma for the closest-matching chunks
    # We over-fetch (get more than we need) because some might get blocked by the permission check
    client = chromadb.PersistentClient(path="data/chroma_store")
    collection = client.get_or_create_collection(name="policybot_docs")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10,  # over-fetch
        include=['documents', 'metadatas', 'distances'],
    )

    allowed_results = []
    blocked_results = []

    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        text = results['documents'][0][i]
        distance = results['distances'][0][i]

        if is_allowed(persona, meta):
            allowed_results.append((meta, text, distance))
        else:
            blocked_results.append((meta, text, distance))

    # Step 3: show what the persona is actually allowed to see
    print(f"✅ Results shown to {persona['name']} ({len(allowed_results)} allowed, showing top {top_n}):\n")
    for meta, text, distance in allowed_results[:top_n]:
        print(f"  From: {meta['doc_name']} (chunk #{meta['chunk_index']}) — Zone: {meta['zone']}")
        print(f"  {text.strip()[:200]}")
        print()

    # Step 4: show what got blocked, for transparency/demo purposes
    if blocked_results:
        print(f"🚫 Blocked by permission check ({len(blocked_results)} chunks):")
        for meta, text, distance in blocked_results[:3]:
            print(f"  From: {meta['doc_name']} — Zone: {meta['zone']} (persona's zone: {persona['zone']})")
        print()

if __name__ == '__main__':
    # A few test queries to try out the permission logic
    search("what is the hotel budget for senior employees", 'pam')      # Americas Associate
    search("what is the hotel budget for senior employees", 'oscar')    # APAC Senior — different zone
    search("what is the hotel budget for senior employees", 'robert')   # Global CFO — sees everything