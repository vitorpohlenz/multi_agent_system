import sys
sys.dont_write_bytecode = True
from pathlib import Path
from typing import Union, Optional  

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

ROOT_DIR = Path(__file__).resolve().parents[2]
AGENTS_DIR = ROOT_DIR / "src" / "agents"
DATA_DIR = ROOT_DIR / "data"
HR_DATA_DIR = DATA_DIR / "hr_docs"

sys.path.append(str(AGENTS_DIR))
from agents_components import default_embeddings, default_splitter, default_llm, build_default_RAG_agent

def build_hr_agent(
    role: str = "HR",
    data_dir: str = HR_DATA_DIR,
    embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings] = default_embeddings,
    splitter = default_splitter,
    llm: ChatOpenAI = default_llm,
    langfuse_handler: Optional[BaseCallbackHandler] = None,
    k_nearest_neighbors: int = 5
):
    """
    Build a HR RAG agent backed by HR policies and FAQs.

    Parameters
    ----------
    data_dir : str
        Directory containing .txt documents.
    embeddings : Union[HuggingFaceEmbeddings, OpenAIEmbeddings], optional
        Embeddings model. Defaults to default_embeddings.
    splitter : some of the classes in langchain.text_splitter, optional
        Text splitter. Defaults to default_splitter.
    llm : ChatOpenAI, optional
        Chat model. Defaults to default_llm.
    langfuse_handler : BaseCallbackHandler, optional
        Langfuse callback handler for tracing.
    k_nearest_neighbors : int, optional
        Number of nearest neighbors to retrieve. Defaults to 5.
    Returns
    -------
    RetrievalQA
        LangChain RetrievalQA chain that answers questions.
    """
    
    return build_default_RAG_agent(
        role=role,
        data_dir=data_dir,
        embeddings=embeddings,
        splitter=splitter,
        llm=llm,
        langfuse_handler=langfuse_handler,
        k_nearest_neighbors=k_nearest_neighbors
    )