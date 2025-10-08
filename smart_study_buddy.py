!pip install -qU langchain langchain-pinecone langchain-openai langchain-google-genai pinecone pypdf pdfplumber langchain-community

import os
import time
from getpass import getpass
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.schema import Document

# --- API keys ---
os.environ["PINECONE_API_KEY"] = getpass("Enter your Pinecone API key: ")
os.environ["GOOGLE_API_KEY"] = getpass("Enter your Google AI API key: ")

pc = Pinecone()

index_name = "smart-study-index"

embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
test_vector = embed_model.embed_query("test")
dimension = len(test_vector)  # usually 768

if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
print("Connected to Pinecone index:", index_name)

txt_loader = TextLoader("/content/graphs.txt", encoding="utf-8")
pdf1_loader = PDFPlumberLoader("/content/Sorting.pdf")
pdf2_loader = PDFPlumberLoader("/content/Trees.pdf")

docs = txt_loader.load() + pdf1_loader.load() + pdf2_loader.load()

# --- Split intelligently ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)
print("Split into", len(split_docs), "chunks")


vector_store = PineconeVectorStore.from_documents(
    documents=split_docs,
    embedding=embed_model,
    index_name=index_name
)

print(f"Stored {len(split_docs)} chunks in Pinecone index '{index_name}'")

llm = GoogleGenerativeAI(model="gemini-2.0-flash")
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    output_key="answer",
    return_messages=True,
)

qa_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""
You are a helpful assistant answering questions based ONLY on the provided notes.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

If the answer is not in the notes, say:
"The information is not in the notes."

Answer:
"""
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    return_source_documents=True,
)

questions = [
    "What are applications of sorting?",
    "What is its importance?",
    "What are the common tree operations?",
    "What are its applications?",
    "Who is the president of USA?",
    "What are the Types of Graphs?",
    "Could you explain Cyclic vs. Acyclic Graphs in detail?"
]

for q in questions:
    result = qa_chain.invoke({"question": q})
    print(f"\nQ: {q}\nA: {result['answer']}")
    time.sleep(10)