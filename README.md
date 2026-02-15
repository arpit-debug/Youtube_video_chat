# Youtube_video_chat
YouTube RAG Chat web app built with Streamlit. It fetches a video's transcript using YouTubeTranscriptApi.fetch() and splits it into chunks. A FAISS vector store with HuggingFace embeddings enables semantic retrieval from the transcript. Users can ask questions, and ChatOllama answers based only on the transcript context.
