from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

text_file_path = "/home/ubuntu/RAG/datasets/TXTs/Accounts101.txt"

loader = TextLoader(text_file_path, encoding="utf8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}


class CustomEmbeddingFunction:
    def __init__(self):
        
        

    def __call__(self, input):
        embedding_output = HuggingFaceEmbeddings(model_name, model_kwargs).embed(input)
        return embedding_output

embeddings = CustomEmbeddingFunction()

vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")


