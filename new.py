from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader, load_index_from_storage, VectorStoreIndex, StorageContext, Settings
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from pathlib import Path
import glob
import pprint
from transformers import AutoTokenizer, Pipeline, AutoModelForCausalLM
import os
from llama_index.core import load_index_from_storage

cache_dir = "/home/ubuntu/RAG/CACHE"


dataset_path = "/home/ubuntu/RAG/datasets/chemistry/*.pdf"
input_files = glob.glob(dataset_path)
reader = SimpleDirectoryReader(input_files=input_files)
documents = reader.load_data()

print('Number of pages:', len(documents))
print(documents)

parser = SentenceSplitter.from_defaults(chunk_size=1024, chunk_overlap=30) # starting to increase chunk_overlap from 20 to 30% to see if it helps with the issue
nodes = parser.get_nodes_from_documents(documents)
print(f"Number of nodes created: {len(nodes)}")

pprint.pprint([nodes[i] for i in range(3)])
output_file = "output.txt"
file_path = os.path.join(cache_dir, output_file)

# Use pprint to format the list of nodes
formatted_output = pprint.pformat([nodes[i] for i in range(3)])

# Write the formatted output to the text file
with open(file_path, "w", encoding="utf-8") as file:
    file.write(formatted_output)

print(f"Output saved successfully to: {file_path}")


faiss_index = faiss.IndexFlatL2(768)
Settings.embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)
subject_dir = "/home/ubuntu/RAG/storage/chemistry"
# service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model)

vector_store = FaissVectorStore(faiss_index=faiss_index)
# Check for environment variable first
storage_dir = os.getenv("STORAGE_DIR", "./storage/chemistry")

# Use storage_dir in your code
storage = StorageContext.from_defaults( vector_store=vector_store)

index = VectorStoreIndex(
    nodes, storage_context=storage
)

index.storage_context.persist(persist_dir=subject_dir)

Settings.llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1, "do_sample": True},
    tokenizer_name="/home/ubuntu/RAG/models",
    model_name="/home/ubuntu/RAG/models",
    tokenizer_kwargs={"max_length": 10000},
    model_kwargs={"torch_dtype": torch.float16}
)
   

storage_context = StorageContext.from_defaults(persist_dir=subject_dir, vector_store=vector_store)

stored_index = load_index_from_storage(storage_context)

query_engine = stored_index.as_query_engine()
prompt="what is amines"

import time
t0=time.time()
response = query_engine.query(prompt)
print(f"Time: {time.time()-t0}")
print(response)
   