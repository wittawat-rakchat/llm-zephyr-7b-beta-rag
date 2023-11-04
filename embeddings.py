import os
import torch
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

embedding_model = "BAAI/bge-large-en"
model_device = {'device': DEVICE}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=embedding_model,
    model_kwargs=model_device,
    encode_kwargs=encode_kwargs
)

pdf_loader = DirectoryLoader('./papers/', glob="./*.pdf",
                         loader_cls=PyPDFLoader)

# pdf_loader = PyPDFLoader("pet.pdf")
documents = pdf_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)

texts = text_splitter.split_documents(documents)

vector_store = Chroma.from_documents(texts,
                                     embeddings,
                                     collection_metadata={"hnsw:space": "cosine"},
                                     persist_directory="stores/paper_cosine")

print("Vector Store: Done!!!")