# main.py
import os
import io
import csv
import json
import random
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document, SystemMessage, HumanMessage

# ---------- Page & ENV ----------
st.set_page_config(page_title=" Librarian", page_icon="📚", layout="wide")
load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Add it to a .env file or export it in your environment.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PERSIST_DIR = "data/chroma_db"

# ---------- Helpers: assets ----------
def inject_css(path: str) -> None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def inject_js(path: str) -> None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<script>{f.read()}</script>", unsafe_allow_html=True)

# ---------- Data ----------
@st.cache_data(show_spinner=False)
def load_books(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_books(books: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm = []
    for b in books:
        norm.append({
            "title": b.get("title", "-"),
            "author": b.get("author", "-"),
            "genre": b.get("genre", "-"),
            "year": b.get("year", "-"),
            "summary": b.get("summary", "-"),
            "cover_url": b.get("cover_url", ""),
            "external_link": b.get("external_link", "")
        })
    return norm

@st.cache_resource(show_spinner=False)
def get_vectorstore(books: List[Dict[str, Any]]):
    texts = [f"{b['title']}: {b['summary']}" for b in books]
    docs = [Document(page_content=t) for t in texts]
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)  
    if os.path.isdir(PERSIST_DIR) and any(os.scandir(PERSIST_DIR)):
        vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        vs = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
        vs.persist()
    return vs

def get_retriever(vs, k: int = 8):
    return vs.as_retriever(search_kwargs={"k": k})

# ---------- Tiny utils ----------
def hide_unknown(value: str) -> bool:
    return value and value != "-" and value.lower() not in {"gen necunoscut", "autor necunoscut"}

def clamp(text: str, max_chars: int = 240) -> str:
    t = (text or "").strip()
    return t if len(t) <= max_chars else t[:max_chars - 1].rstrip() + "…"

def text_to_speech_bytes(text: str, slow: bool = False) -> bytes:
    tts = gTTS(text, lang="ro", slow=slow)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return open(fp.name, "rb").read()

def transcribe_audio_whisper(audio_bytes: bytes) -> Optional[str]:
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model": "whisper-1"}
        r = requests.post("https://api.openai.com/v1/audio/transcriptions",
                          headers=headers, files=files, data=data, timeout=60)
        if r.ok:
            return r.json().get("text", "").strip()
        st.error(f"Whisper error (HTTP {r.status_code}).")
    except Exception as e:
        st.error(f"Eroare transcriere: {e}")
    return None

def get_book_by_title(books: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
    for b in books:
        if b["title"].lower() == title.lower():
            return b
    return {"title": title, "author": "-", "genre": "-", "year": "-", "summary": "-", "cover_url": "", "external_link": ""}

def trending_picker(books: List[Dict[str, Any]], likes: List[str]) -> List[Dict[str, Any]]:
    picks: List[Dict[str, Any]] = []
    liked = [b for b in books if b["title"] in set(likes)]
    if liked:
        picks.extend(liked[:2])
    pool = [b for b in books if b not in picks]
    random.seed(datetime.now().strftime("%Y-%m-%d"))  # rotate daily
    if pool:
        picks.extend(random.sample(pool, k=min(3 - len(picks), len(pool))))
    return picks[:3]

def recommend_with_llm(query: str, candidates: List[Dict[str, Any]], temperature: float) -> str:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=temperature)  # uses env var key
    context = []
    for c in candidates[:2]:
        context.append(
            f"- Titlu: {c['title']} | Autor: {c['author']} | Gen: {c['genre']} | An: {c['year']}\n  Rezumat: {c['summary']}"
        )
    sys = (
        "Ești un bibliotecar virtual în limba română. "
        "Oferă o recomandare concisă (4–6 fraze) și explică de ce se potrivește cerinței. "
        "Recomandă 1 titlu principal și menționează opțional încă o alternativă. Nu inventa detalii."
    )
    human = f"Cerință: {query}\nCandidate:\n" + ("\n".join(context) if context else "N/A") + \
            "\nFormatează titlul cu **bold** și include autorul și anul."
    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=human)])
    return resp.content.strip()

# ---------- State ----------
if "history" not in st.session_state:
    st.session_state["history"] = []
if "profile" not in st.session_state:
    st.session_state["profile"] = {"preferinte": [], "like": [], "dislike": []}
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.5
if "tts_slow" not in st.session_state:
    st.session_state["tts_slow"] = False
if "reading_list" not in st.session_state:
    st.session_state["reading_list"] = []

inject_css("ui/custom.css")
inject_js("ui/main.js")

BOOKS_RAW = load_books("book_summaries_large.json")
BOOKS = normalize_books(BOOKS_RAW)
VS = get_vectorstore(BOOKS)
RETRIEVER = get_retriever(VS, k=8)

st.markdown("""
<div class="hero">
  <h1>📚 Smart Librarian</h1>
  <p>Rapid. Curat. Recomandări inteligente cu RAG + UI modern.</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["✨ Recomandări", "🔍 Căutare", "🧰 Compare", "🕑 Istoric", "📥 Reading List", "⚙️ Setări"])

with tabs[0]:
    st.header("✨ Recomandări personalizate")
    likes = st.session_state["profile"]["like"]
    top3 = trending_picker(BOOKS, likes)
    st.markdown(
        "<div class='trending'><span class='fire'>🔥</span> Trending: " +
        " | ".join([f"<b>{b['title']}</b>" for b in top3]) +
        "</div>", unsafe_allow_html=True
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input("Ce fel de carte cauți? (ex: „Vreau o carte despre distopie și manipularea adevărului”)",
                              key="query1")
    with c2:
        st.caption("Dictează întrebarea")
        mic = mic_recorder(key="mic_rec1", start_prompt="🎙️ Start", stop_prompt="⏹️ Stop", use_container_width=True)
        audio_bytes = None
        if mic is not None:
            if isinstance(mic, dict) and "bytes" in mic: audio_bytes = mic["bytes"]
            elif isinstance(mic, (tuple, list)): audio_bytes = mic[0]
            elif isinstance(mic, (bytes, bytearray)): audio_bytes = mic

    final_query = query
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("📝 Transcriu audio..."):
            tx = transcribe_audio_whisper(audio_bytes)
            if tx:
                st.success(f"Transcris: {tx}")
                final_query = tx

    use_voice = st.checkbox("Redă recomandarea în audio", key="use_voice", value=False)

    if final_query:
        bad_words = {"idiot", "prost", "urât", "tâmpit"}
        if any(w in final_query.lower() for w in bad_words):
            st.warning("Te rog să folosești un limbaj adecvat.")
        else:
            with st.spinner("🔎 Caut în colecție..."):
                docs = RETRIEVER.get_relevant_documents(final_query)
            if not docs:
                st.error("Nu am găsit potriviri. Încearcă să reformulezi.")
            else:
                titles = [d.page_content.split(":")[0].strip() for d in docs]
                seen = set()
                uniq_titles = [t for t in titles if not (t in seen or seen.add(t))]
                candidates = [get_book_by_title(BOOKS, t) for t in uniq_titles[:5]]

                selected_title = st.selectbox("Alege titlul sugerat", uniq_titles[:10], key="sel1")
                with st.spinner("🧠 Generez recomandarea..."):
                    msg = recommend_with_llm(final_query, candidates, st.session_state["temperature"])
                st.success(f"🤖 Recomandare: {msg}")

                details = get_book_by_title(BOOKS, selected_title)
                cover_html = f"<img src='{details['cover_url']}' class='book-cover'/>" if details.get("cover_url") else ""
                st.markdown(
                    f"""
                    <div class='book-card'>
                        <div class='book-row'>
                            {cover_html}
                            <div class='book-info'>
                                <div class='badges'>
                                    {f"<span class='badge'>{details['genre']}</span>" if hide_unknown(details.get('genre','')) else ""}
                                    {f"<span class='badge'>{details['year']}</span>" if hide_unknown(str(details.get('year',''))) else ""}
                                    {f"<span class='badge'>{details['author']}</span>" if hide_unknown(details.get('author','')) else ""}
                                </div>
                                <h3 class='title'>{details['title']}</h3>
                                <div class='summary' style='-webkit-line-clamp:unset;'>{details['summary']}</div>
                                {f"<a class='book-link' target='_blank' href='{details.get('external_link','')}'>Vezi detalii</a>" if details.get('external_link') else ""}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )

                colA, colB = st.columns(2)
                with colA:
                    if st.button("👍 Îmi place", key="like_btn"):
                        if selected_title not in st.session_state["profile"]["like"]:
                            st.session_state["profile"]["like"].append(selected_title)
                        st.success("Adăugat la favorite! 🎉")
                        st.markdown("<script>window.confettiLike && window.confettiLike();</script>",
                                    unsafe_allow_html=True)
                with colB:
                    if st.button("👎 Nu-mi place", key="dislike_btn"):
                        if selected_title not in st.session_state["profile"]["dislike"]:
                            st.session_state["profile"]["dislike"].append(selected_title)
                        st.info("Vom ține cont pe viitor.")

                if use_voice and msg:
                    st.audio(text_to_speech_bytes(msg, slow=st.session_state["tts_slow"]), format="audio/mp3")

                st.session_state["history"].append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": final_query,
                    "response": msg,
                    "book": selected_title
                })

with tabs[1]:
    st.header("🔍 Căutare rapidă")
    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
    with c1:
        search_text = st.text_input("Caută în titlu / rezumat", placeholder="ex: distopie, magie, prietenie…")
    with c2:
        genre = st.selectbox("Gen", ["Oricare"] + sorted({b["genre"] for b in BOOKS if hide_unknown(b.get("genre",""))}))
    with c3:
        author = st.selectbox("Autor", ["Oricare"] + sorted({b["author"] for b in BOOKS if hide_unknown(b.get("author",""))}))
    with c4:
        years = sorted({str(b["year"]) for b in BOOKS if hide_unknown(str(b.get("year","")))})
        year = st.selectbox("An", ["Oricare"] + years)

    if st.button("Caută", use_container_width=True):
        st.session_state["__run_search"] = True

    filtered = BOOKS
    if st.session_state.get("__run_search"):
        if genre != "Oricare":
            filtered = [b for b in filtered if b.get("genre") == genre]
        if author != "Oricare":
            filtered = [b for b in filtered if b.get("author") == author]
        if year != "Oricare":
            filtered = [b for b in filtered if str(b.get("year")) == year]
        if search_text:
            s = search_text.lower()
            filtered = [b for b in filtered if s in b["title"].lower() or s in b["summary"].lower()]

        st.caption(f"Rezultate: **{len(filtered)}**")

        cards = []
        for i, b in enumerate(filtered):
            badges = []
            if hide_unknown(b.get("genre","")): badges.append(f"<span class='badge'>{b['genre']}</span>")
            if hide_unknown(str(b.get("year",""))): badges.append(f"<span class='badge'>{b['year']}</span>")
            if hide_unknown(b.get("author","")): badges.append(f"<span class='badge'>{b['author']}</span>")
            badges_html = "".join(badges)
            cover = b.get("cover_url") or f"https://picsum.photos/seed/book{i}/600/360"
            cards.append(f"""
            <article class="tile">
              <img src="{cover}" alt="cover" class="cover"/>
              <div class="body">
                <div class="badges">{badges_html}</div>
                <div class="title">{b['title']}</div>
                <div class="summary">{clamp(b['summary'])}</div>
              </div>
            </article>
            """)
        st.markdown("<section class='grid'>" + "\n".join(cards) + "</section>", unsafe_allow_html=True)

        titles = [b["title"] for b in filtered]
        sel = st.selectbox("Afișează detalii pentru:", ["— alege —"] + titles, index=0)
        if sel != "— alege —":
            book = next(b for b in filtered if b["title"] == sel)
            st.markdown(
                """
                <div class='book-card'>
                  <div class='book-row'>
                """ +
                (f"<img src='{book.get('cover_url','')}' class='book-cover'/>" if book.get("cover_url") else "") +
                f"""
                    <div class='book-info'>
                      <div class='badges'>
                        {f"<span class='badge'>{book['genre']}</span>" if hide_unknown(book.get('genre','')) else ""}
                        {f"<span class='badge'>{book['year']}</span>" if hide_unknown(str(book.get('year',''))) else ""}
                        {f"<span class='badge'>{book['author']}</span>" if hide_unknown(book.get('author','')) else ""}
                      </div>
                      <h3 class='title'>{book['title']}</h3>
                      <div class='summary' style='-webkit-line-clamp:unset;'>{book['summary']}</div>
                      {f"<a class='book-link' href='{book.get('external_link','')}' target='_blank'>Vezi detalii</a>" if book.get('external_link') else ""}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            cc1, cc2, cc3 = st.columns([1,1,2])
            with cc1:
                if st.button("➕ Add to Reading List", key=f"add_{sel}"):
                    if sel not in st.session_state["reading_list"]:
                        st.session_state["reading_list"].append(sel)
                    st.success("Adăugată în Reading List.")
            with cc2:
                if st.button("🔊 TTS", key=f"tts_{sel}"):
                    st.audio(text_to_speech_bytes(clamp(book['summary'], 800), slow=st.session_state["tts_slow"]),
                             format="audio/mp3")

with tabs[2]:
    st.header("🧰 Compare două cărți")
    base = [b["title"] for b in BOOKS]
    pair = st.multiselect("Alege exact două titluri", base, max_selections=2)
    if len(pair) == 2:
        b1 = get_book_by_title(BOOKS, pair[0])
        b2 = get_book_by_title(BOOKS, pair[1])
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(b1["title"])
            st.write(f"**Autor:** {b1.get('author','-')}")
            st.write(f"**Gen:** {b1.get('genre','-')}")
            st.write(f"**An:** {b1.get('year','-')}")
            st.write(clamp(b1["summary"], 900))
        with c2:
            st.subheader(b2["title"])
            st.write(f"**Autor:** {b2.get('author','-')}")
            st.write(f"**Gen:** {b2.get('genre','-')}")
            st.write(f"**An:** {b2.get('year','-')}")
            st.write(clamp(b2["summary"], 900))

with tabs[3]:
    st.header("🕑 Istoric conversații")
    hist = st.session_state["history"]
    if not hist:
        st.info("Nu există conversații salvate încă.")
    else:
        colx, coly = st.columns([4,1])
        with coly:
            if st.button("Șterge istoricul 🗑️"):
                st.session_state["history"] = []
                st.rerun()
        for h in reversed(hist):
            st.markdown(
                f"**{h['timestamp']}** | Întrebare: _{h['query']}_\n\n"
                f"**Răspuns:** {h['response']}\n\n**Carte:** {h['book']}\n---"
            )

with tabs[4]:
    st.header("📥 Reading List")
    rl = st.session_state["reading_list"]
    if not rl:
        st.write("Încă nu ai adăugat nimic.")
    else:
        for t in rl:
            st.markdown(f"- **{t}**")
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["title"])
        for t in rl: w.writerow([t])
        st.download_button("⬇️ Export CSV", data=buf.getvalue(),
                           file_name="reading_list.csv", mime="text/csv")
        if st.button("🗑️ Golește lista"):
            st.session_state["reading_list"] = []
            st.success("Golita.")

with tabs[5]:
    st.header("⚙️ Setări")
    st.session_state["temperature"] = st.slider(
        "Creativitate (temperature)", 0.0, 1.0, st.session_state.get("temperature", 0.5), 0.05
    )
    tts_speed = st.radio("Viteză TTS", ["Normal", "Lent"], horizontal=True)
    st.session_state["tts_slow"] = (tts_speed == "Lent")
    st.caption("Poți schimba modelele prin .env: LLM_MODEL=gpt-4o-mini, EMBEDDING_MODEL=text-embedding-3-small")
