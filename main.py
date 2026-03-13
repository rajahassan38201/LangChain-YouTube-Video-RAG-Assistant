import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

active_chain = None

class VideoRequest(BaseModel):
    url: str
    language: str = "en"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []

def extract_video_id(url: str):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def format_history(history):
    return "\n".join(f"{msg.role}: {msg.content}" for msg in history)

@app.post("/process")
def process_video(req: VideoRequest):
    global active_chain
    video_id = extract_video_id(req.url)
    
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=[req.language])
        transcript_list = fetched_transcript.to_raw_data()
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
    except Exception as e:
        error_message = str(e)
        if "No transcripts were found" in error_message:
            friendly_error = "This video does not have a transcript in the selected language. Please try selecting 'English'."
        elif "TranscriptsDisabled" in str(type(e)):
            friendly_error = "The creator has disabled closed captions for this video."
        else:
            friendly_error = "Could not retrieve the video transcript. Please ensure closed captions are enabled."
        raise HTTPException(status_code=400, detail=friendly_error)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    # Updated prompt to include chat history
    prompt = PromptTemplate(
    template="""
    You are a helpful and educational YouTube video assistant.
    Your goal is to help the user learn by explaining concepts from the video clearly.

    CRITICAL RULES:
    1. ALWAYS reply entirely in English, using simple, clear, and easy-to-understand wording.
    2. Base your answers strictly on the provided Context and the Chat History. 
    3. Pay close attention to the Chat History. If the user asks to elaborate on a previous point (e.g., "explain the second point"), identify that point from the history and explain it thoroughly.
    4. When asked to explain in detail, provide practical examples to make learning easier.
    5. If the user asks something unrelated to the video, politely guide them back to the video's content based on the transcript.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question: {question}
    
    Helpful Educational Answer in English:
    """,
    input_variables=['context', 'chat_history', 'question']
    )
    # Parallel chain passes both question and history into the prompt
    parallel_chain = RunnableParallel({
        # FIX: Explicitly extract 'question' before sending it to the retriever
        'context': RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
        'chat_history': RunnableLambda(lambda x: format_history(x["history"])),
        'question': RunnableLambda(lambda x: x["question"])
    })
    
    active_chain = parallel_chain | prompt | llm | StrOutputParser()

    return {"message": "Video processed successfully!"}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not active_chain:
        raise HTTPException(status_code=400, detail="Please process a video first.")
    
    async def generate():
        # Pass both question and history to the chain
        for chunk in active_chain.stream({"question": req.question, "history": req.history}):
            yield chunk


    return StreamingResponse(generate(), media_type="text/plain")
