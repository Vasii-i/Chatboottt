# app.py
import streamlit as st
import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from tools import get_summary_by_title

# Setare cheia OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY nu este setată în mediu.")
    st.stop()

# Init OpenAI client
client = OpenAI(api_key=openai_api_key)

# Init Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)
collection = chroma_client.get_or_create_collection(
    name="book_summaries",
    embedding_function=embedding_function
)

# Căutare semantică în vector store
def retrieve_relevant_titles(query, top_k=1):
    results = collection.query(query_texts=[query], n_results=top_k)
    return [meta["title"] for meta in results["metadatas"][0]]

# Recomandare GPT
def ask_gpt_for_recommendation(user_query, title):
    prompt = (
        f"User: {user_query}\n"
        f"Assistant: Îți recomand cartea '{title}'. Este foarte potrivită pentru interesele tale. "
        f"Dorești un rezumat detaliat?"
    )
    try:
        # Încearcă modelul cel mai nou acceptat de API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        # Fallback la modelul vechi dacă cel nou nu merge
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e2:
            st.error(f"Eroare OpenAI: {e2}")
            return "Eroare la generarea răspunsului."
    return response.choices[0].message.content

# UI Streamlit
st.set_page_config(page_title="📚 Smart Librarian", page_icon="📘")
st.title("📚 Smart Librarian – Recomandări de Cărți")

with st.form("book_form"):
    user_query = st.text_input("Ce fel de carte cauți?", placeholder="ex: Vreau o carte cu magie și prietenie")
    submitted = st.form_submit_button("Caută")

if submitted and user_query:
    titles = retrieve_relevant_titles(user_query)
    if not titles:
        st.warning("🤖 Nu am găsit nicio carte potrivită.")
    else:
        recommended_title = titles[0]
        gpt_response = ask_gpt_for_recommendation(user_query, recommended_title)
        
        st.success(f"📖 Recomandare: **{recommended_title}**")
        st.write(gpt_response)

        detailed = get_summary_by_title(recommended_title)
        st.info(f"📝 Rezumat detaliat:\n\n{detailed}")
