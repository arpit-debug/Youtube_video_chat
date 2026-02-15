from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama

video_id = "p3F-1QyvHnY"


ytt_api = YouTubeTranscriptApi()

try:
    transcript_list = ytt_api.fetch(video_id)

    transcript = " ".join(chunk.text for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")

except NoTranscriptFound:
    print("No transcript found for this video.")

except VideoUnavailable:
    print("Video unavailable.")

except Exception as e:
    print("Unexpected error:", e)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.create_documents([transcript])
print(len(chunks))


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)
retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})

query = "Summarize the whole video is about?"
print("Question:",query)
# retrive_chats= retriver.invoke(query)

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
llm = ChatOllama(
    model="qwen2.5",
    temperature=0
)


prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {query}
    """,
    input_variables = ['context', 'query']
)


from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
#Parellel chains
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriver | RunnableLambda(format_docs),
    'query': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

response = main_chain.invoke(query)

print("response:",response)