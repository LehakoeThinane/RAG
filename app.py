import os

import chromadb
from chromadb.utils import embedding_functions

ollama_url = "http://localhost:11434"
embedding_function = embedding_functions.OllamaEmbeddingFunction(
    url=ollama_url,
    model_name="nomic-embed-text",
    timeout=300,
)

# Initialize the ChromaDB client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection_ollama"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=embedding_function
)

def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "content": file.read()})
    return documents


def split_text(text, chunk_size=1000, overlap=20):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


directory_path = "./docs"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents from {directory_path}")

chunk_ids = []
chunk_texts = []
chunk_metadata = []

for document in documents:
    chunks = split_text(document["content"])
    for index, chunk in enumerate(chunks):
        chunk_ids.append(f"{document['id']}:{index}")
        chunk_texts.append(chunk)
        chunk_metadata.append({"source": document["id"], "chunk": index})

batch_size = 16
for start in range(0, len(chunk_ids), batch_size):
    end = start + batch_size
    collection.upsert(
        ids=chunk_ids[start:end],
        documents=chunk_texts[start:end],
        metadatas=chunk_metadata[start:end],
    )
    print(f"Stored chunks {min(end, len(chunk_ids))}/{len(chunk_ids)}")

print(f"Created {len(chunk_texts)} chunks and stored them in ChromaDB")
print("Collection count:", collection.count())
print("Sample stored rows:")
sample = collection.get(
    limit=3,
    include=["documents", "metadatas", "embeddings"],
)
print(sample)