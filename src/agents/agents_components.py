import sys
sys.dont_write_bytecode = True
import os
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv

from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

default_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        add_start_index=True,
        separators=["\n\n", "\n"]
    )

default_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

default_llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.1 # 0 is the most deterministic, 1 is the most random
)

def build_default_RAG_agent(
    data_dir: str,
    role: str,
    embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
    splitter,
    llm: ChatOpenAI,
    langfuse_handler: Optional[BaseCallbackHandler] = None,
    k_nearest_neighbors: int = 5

):
    """
    Build a default RAG agent to be customized for each specific use case.
    
    Parameters
    ----------
    data_dir : str
        Directory containing .txt documents.
    role : str
        Role of the agent.
    embeddings : Union[HuggingFaceEmbeddings, OpenAIEmbeddings]
        Embeddings model.
    splitter : some of the classes in langchain.text_splitter
        Text splitter.
    llm : ChatOpenAI
        Chat model.
    langfuse_handler : Optional[BaseCallbackHandler]
        Langfuse callback handler for tracing.
    k_nearest_neighbors : int
        Number of nearest neighbors to retrieve.
    Returns
    -------
    RetrievalQA
        LangChain RetrievalQA chain that answers questions about the documents.
    """
    callbacks = [langfuse_handler] if langfuse_handler else None

    # Load documents
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"{role} data directory not found: {data_dir}")

    docs = []
    for file in data_path.glob("*.txt"):
        loader = TextLoader(file)
        docs.extend(loader.load())

    if not docs:
        raise ValueError(f"No {role} documents found in {data_dir}")

    split_docs = splitter.split_documents(docs)

    # Vector store
    vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_nearest_neighbors})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        callbacks=callbacks,
        chain_type="stuff",
        name=f"{role} RAG Agent",
    )

    return qa