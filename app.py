import streamlit as st
import streamlit.components.v1 as components
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="YouTube RAG Chat", layout="wide")
st.title("🎥 YouTube Video Chat (Persistent Player)")
st.write("Watch and ask questions without video reload.")

# =============================
# EXTRACT VIDEO ID
# =============================
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# =============================
# CACHE MODELS
# =============================
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    return ChatOllama(
        model="qwen2.5",
        temperature=0
    )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the transcript context.
If insufficient context, say you don't know.

Context:
{context}

Question:
{query}
""",
    input_variables=["context", "query"]
)

# =============================
# BUILD RAG
# =============================
def build_chain(transcript):

    embedding_model = load_embedding_model()
    llm = load_llm()

    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "query": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# =============================
# SESSION STATE INIT
# =============================
if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "video_id" not in st.session_state:
    st.session_state.video_id = None

# =============================
# INPUT
# =============================
youtube_url = st.text_input("Enter YouTube URL")

if st.button("Load Video"):

    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("Invalid URL")
        st.stop()

    # Only rebuild if new video
    if video_id != st.session_state.video_id:

        st.session_state.video_id = video_id
        st.session_state.chat_history = []

        try:
            with st.spinner("Fetching transcript..."):
                # <<< Use fetch() here for older versions >>>
                transcript_list = YouTubeTranscriptApi().fetch(video_id)
                transcript = " ".join(x.text for x in transcript_list)

            with st.spinner("Building RAG index..."):
                st.session_state.chain = build_chain(transcript)

            st.success("Video ready!")

        except TranscriptsDisabled:
            st.error("Transcripts disabled.")
        except NoTranscriptFound:
            st.error("No transcript found.")
        except VideoUnavailable:
            st.error("Video unavailable.")
        except Exception as e:
            st.error(str(e))

# =============================
# PERSISTENT VIDEO PLAYER
# =============================
if st.session_state.video_id:

    st.subheader("📺 Video Player")

    # Always render video
    video_html = f"""
    <div style="display:flex; justify-content:center;">
        <iframe
            width="720"
            height="405"
            src="https://www.youtube.com/embed/{st.session_state.video_id}?enablejsapi=1"
            frameborder="0"
            allow="autoplay; encrypted-media"
            allowfullscreen>
        </iframe>
    </div>
    """
    components.html(video_html, height=430)

# =============================
# CHAT SECTION
# =============================
if st.session_state.chain:

    st.subheader("💬 Ask Questions")

    user_question = st.chat_input("Ask about the video...")

    if user_question:
        st.session_state.chat_history.append(("user", user_question))

        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(user_question)

        st.session_state.chat_history.append(("assistant", response))

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
