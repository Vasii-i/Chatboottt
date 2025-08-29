import json
import os
import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ✅ Verifică dacă cheia OpenAI există
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("❌ OPENAI_API_KEY nu este setată în variabilele de mediu.")

# ✅ Încarcă rezumatele de cărți din fișierul JSON
with open("book_summaries.json", "r", encoding="utf-8") as f:
    books = json.load(f)

# ✅ Inițializează ChromaDB cu noul client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ✅ Configurare embeddings cu OpenAI
embedding_function = OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-3-small"
)

# ✅ Creează colecția dacă nu există deja
collection = chroma_client.get_or_create_collection(
    name="book_summaries",
    embedding_function=embedding_function
)

# ✅ Încarcă toate cărțile în vector store
print("📚 Încarc cărțile în vector store...")
for idx, book in enumerate(books):
    collection.add(
        ids=[str(idx)],
        documents=[book["summary"]],
        metadatas=[{"title": book["title"]}]
    )
print("✅ Vector store populat cu succes!")

# ✅ Funcție de căutare semantică
def search_books(query: str, top_k: int = 3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    print(f"\n🔎 Rezultate pentru: '{query}'\n{'='*40}")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"📘 Titlu: {meta['title']}\n📝 Rezumat: {doc}\n{'-'*40}")

# ✅ Interfață interactivă CLI
if __name__ == "__main__":
    print("\n🤖 Poți căuta cărți după temă! Scrie 'exit' pentru a ieși.")
    while True:
        query = input("\nCe fel de carte cauți? > ")
        if query.lower().strip() in {"exit", "quit"}:
            print("👋 La revedere!")
            break
        search_books(query)
