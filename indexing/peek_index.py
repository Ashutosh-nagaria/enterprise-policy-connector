import chromadb

client = chromadb.PersistentClient(path="data/chroma_store")
collection = client.get_or_create_collection(name="policybot_docs")

print(f"Total chunks stored: {collection.count()}\n")

# Pull everything back out and print it in a readable way
results = collection.get(include=['documents', 'metadatas'])

for i in range(len(results['ids'])):
    meta = results['metadatas'][i]
    text = results['documents'][i]
    print(f"--- Chunk {i+1} ---")
    print(f"  From: {meta['doc_name']}  (chunk #{meta['chunk_index']})")
    print(f"  Zone: {meta['zone']} | Sensitivity: {meta['sensitivity']} | Min-Role: {meta['min_role']}")
    print(f"  Text: {text[:120].strip()}...")
    print()