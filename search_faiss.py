# Description: This file contains the function to search the faiss index for the given prompt.
import faiss
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
import numpy as np
from pdf_to_txt import convert_pdf_to_txt_with_ocr
from llama_index.core import load_index_from_storage
import time
import os
import pprint
import glob

def search_faiss_index(promp, sub_db, subject): #-> list
    cache_dir = "./CACHE"
    dataset_path = f"./datasets/{subject}/*.pdf"
    input_files = glob.glob(dataset_path)
    reader = SimpleDirectoryReader(input_files=input_files)
    documents = reader.load_data()
    parser = SentenceSplitter.from_defaults(chunk_size=2048, chunk_overlap=30) # starting to increase chunk_overlap from 20 to 30% to see if it helps with the issue
    nodes = parser.get_nodes_from_documents(documents)

    pprint.pprint([nodes[i] for i in range(3)])
    output_file = "output.txt"
    file_path = os.path.join(cache_dir, output_file)
    formatted_output = pprint.pformat([nodes[i] for i in range(3)])
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(formatted_output)

  
    faiss_index = faiss.IndexFlatL2(768)
    Settings.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_dir = os.getenv("STORAGE_DIR", sub_db)
    print(storage_dir, "storage_dir")
    Settings.llm = HuggingFaceLLM(
    context_window=2048, max_new_tokens=512,
    generate_kwargs={"temperature": 0.1, "do_sample": True},
    tokenizer_name="./models", model_name="./models",
    tokenizer_kwargs={"max_length": 10000},
    model_kwargs={"torch_dtype": torch.float16})
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir, vector_store=vector_store)
    stored_index = load_index_from_storage(storage_context)
    query_engine = stored_index.as_query_engine()
    prompt="what is amines"
    t0=time.time()
    response = query_engine.query(prompt)
    print(f"Time: {time.time()-t0}")
    print(response)
    return response
    
    