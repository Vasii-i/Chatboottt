import os
import chromadb
from tools import get_summary_by_title
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

# Setare API key pentru OpenAI și Chroma
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ Variabila OPENAI_API_KEY nu este setată.")

# Inițializează clientul OpenAI (noua versiune)
client = OpenAI(api_key=openai_api_key)

# Inițializează ChromaDB Persistent Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Embeddings cu OpenAI
embedding_function = OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)

# Obține colecția (sau o creează)
collection = chroma_client.get_or_create_collection(
    name="book_summaries",
    embedding_function=embedding_function
)

# 🔍 RAG: căutare în vector store
def retrieve_relevant_titles(user_query, top_k=1):
    results = collection.query(query_texts=[user_query], n_results=top_k)
    return [meta["title"] for meta in results["metadatas"][0]]

# 🤖 GPT: generează răspuns conversațional
def ask_gpt_for_recommendation(user_query, title):
    prompt = (
        f"User: {user_query}\n"
        f"Assistant: Îți recomand cartea '{title}'. Este foarte potrivită pentru interesele tale. "
        f"Dorești un rezumat detaliat?"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # sau "gpt-4", dacă ai acces
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 🧑‍💻 CLI: interfața utilizator
def chatbot_loop():
    print("📚 Chatbot Smart Librarian – Scrie 'exit' pentru a ieși.\n")
    while True:
        user_input = input("Tu: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 La revedere!")
            break

        titles = retrieve_relevant_titles(user_input)
        if not titles:
            print("🤖 Nu am găsit cărți potrivite.")
            continue

        recommended_title = titles[0]
        response = ask_gpt_for_recommendation(user_input, recommended_title)
        print(f"\n🤖 {response}")

        full_summary = get_summary_by_title(recommended_title)
        print(f"\n📘 Rezumat detaliat pentru '{recommended_title}':\n{full_summary}\n{'-'*60}")

if __name__ == "__main__":
    chatbot_loop()
