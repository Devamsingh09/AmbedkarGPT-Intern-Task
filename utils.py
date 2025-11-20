from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import os


def load_text(speech_path: str) -> str:
    """Read the speech.txt and return text as one string."""
    loader = TextLoader(speech_path, encoding="utf-8")
    docs = loader.load()
    return "\n\n".join([d.page_content for d in docs])


def split_text(text: str, chunk_size: int = 80, chunk_overlap: int = 25):
    """Split text into chunks using CharacterTextSplitter."""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = splitter.split_text(text)
    return [Document(page_content=t) for t in texts]


def create_embeddings(documents, persist_directory: str = "./chroma_db"):
    """Create Chroma vectorstore and persist embeddings."""
    os.makedirs(persist_directory, exist_ok=True)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents,
        embedding,
        persist_directory=persist_directory,
        collection_name="ambedkar_speech",
    )

    return vectordb


def get_retriever(persist_directory: str = "./chroma_db", k: int = 4):
    """Load persisted Chroma DB and return retriever."""
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
        collection_name="ambedkar_speech",
    )

    return vectordb.as_retriever(search_kwargs={"k": k})


def answer_query(question: str, persist_dir: str = "./chroma_db") -> str:
    """Retrieve context and generate an answer using Ollama + Mistral 7B with LCEL."""
    retriever = get_retriever(persist_directory=persist_dir)

    llm = ChatOllama(model="mistral")

    template = """Answer the question based only on the following context:
{context}
Otherwise just say - "Answer to yur query is out of the context."

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)


def ingest(speech_path: str = "speech.txt", persist_dir: str = "./chroma_db"):
    """Full ingestion pipeline: load → split → embed → persist."""
    print(f"Loading speech from: {speech_path}")
    text = load_text(speech_path)

    print("Splitting text into chunks...")
    docs = split_text(text)
    print(f"Created {len(docs)} chunks.")

    print(f"Creating embeddings + Chroma DB at {persist_dir}...")
    create_embeddings(documents=docs, persist_directory=persist_dir)

    print("Ingestion complete.")
