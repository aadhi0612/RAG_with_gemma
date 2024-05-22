import os
import faiss
import numpy as np
import glob
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import SimpleDirectoryReader, load_index_from_storage, VectorStoreIndex, StorageContext, Settings
import pprint


cache_dir = "/home/ubuntu/RAG/CACHE"
dataset_path = "/home/ubuntu/RAG/datasets/chemistry/*.pdf"
input_files = glob.glob(dataset_path)
reader = SimpleDirectoryReader(input_files=input_files)
documents = reader.load_data()
parser = SentenceSplitter.from_defaults(chunk_size=1024, chunk_overlap=30) # starting to increase chunk_overlap from 20 to 30% to see if it helps with the issue
nodes = parser.get_nodes_from_documents(documents)
print(f"Number of nodes created: {len(nodes)}")

pprint.pprint([nodes[i] for i in range(3)])
output_file = "output.txt"
file_path = os.path.join(cache_dir, output_file)

formatted_output = pprint.pformat([nodes[i] for i in range(3)])

with open(file_path, "w", encoding="utf-8") as file:
    file.write(formatted_output)

print(f"Output saved successfully to: {file_path}")


faiss_index = faiss.IndexFlatL2(768)
Settings.embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes, storage_context=storage_context
)

index.storage_context.persist()

