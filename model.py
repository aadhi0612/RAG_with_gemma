import os
import fitz
import torch
import faiss
import glob
import pprint
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
import logging
import numpy as np
from pdf_to_txt import convert_pdf_to_txt_with_ocr
from llama_index.core import load_index_from_storage
import time
import sys
import json
from search_faiss import search_faiss_index

sys.setrecursionlimit(30000)
logging.basicConfig(filename='./CACHE/errors.txt', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')
logger = logging.getLogger()

app = Flask(__name__)

# Configuration
MODEL_DIR = "./models"
DATASET_DIR = "./datasets"
TXT_DIR = "./txt_files"
STORAGE_DIR = "./storage"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# Load tokenizer and sentence model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

embedding_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

def load_texts_from_folder(folder_path: str) -> list:
    text_files = glob.glob(os.path.join(folder_path, '*.txt'))
    texts = []
    for text_file in text_files:
        with open(text_file, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def move_storage_files(subject: str):
    subject_storage_dir = os.path.join(STORAGE_DIR, subject)
    os.makedirs(subject_storage_dir, exist_ok=True)

    storage_files = [
        "default__vector_store.json",
        "docstore.json",
        "graph_store.json",
        "image__vector_store.json",
        "index_store.json"
    ]

    for file_name in storage_files:
        src = os.path.join(STORAGE_DIR, file_name)
        dst = os.path.join(subject_storage_dir, file_name)
        if os.path.exists(src):
            os.rename(src, dst)
    return subject_storage_dir

def create_faiss_index_for_subject(subject: str):
    subject_folder = os.path.join(DATASET_DIR, subject)
    os.makedirs(subject_folder, exist_ok=True)
    subject_txt_dir = os.path.join(TXT_DIR, subject)
    os.makedirs(subject_txt_dir, exist_ok=True)

    convert_pdf_to_txt_with_ocr(subject_folder, subject_txt_dir)

    input_files = glob.glob(os.path.join(subject_txt_dir, '*.txt'))
    reader = SimpleDirectoryReader(input_files=input_files)
    documents = reader.load_data()
    parser = SentenceSplitter.from_defaults(chunk_size=2048, chunk_overlap=30)  # 30% overlap
    nodes = parser.get_nodes_from_documents(documents)
    pprint.pprint([nodes[i] for i in range(3)])
    faiss_index = faiss.IndexFlatL2(768)
    Settings.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    index.storage_context.persist()
    return index

# def search_faiss_index(tokenizer, prom, subject_dir) -> list:
#     faiss_index = faiss.IndexFlatL2(768)
#     Settings.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
#     Settings.llm = HuggingFaceLLM(
#     context_window=2048,
#     max_new_tokens=512,
#     generate_kwargs={"temperature": 0.1, "do_sample": True},
#     tokenizer_name="/home/ubuntu/RAG/models",
#     model_name="/home/ubuntu/RAG/models",
#     tokenizer_kwargs={"max_length": 10000},
#     model_kwargs={"torch_dtype": torch.float16})
    
#     vector_store = FaissVectorStore(faiss_index=faiss_index)
#     storage_context = StorageContext.from_defaults(persist_dir=subject_dir, vector_store=vector_store)
#     stored_index = load_index_from_storage(storage_context)
#     query_engine = stored_index.as_query_engine()

#     t0 = time.time()
#     response = query_engine.query(prom)
#     print(f"Time: {time.time() - t0}")
#     return response

def llm(model_dir, prompt) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    output_ids = model.generate(input_ids,
                                max_length=2048,
                                num_return_sequences=1,
                                no_repeat_ngram_size=2,
                                eos_token_id=tokenizer.eos_token_id,
                                top_k=50, do_sample=False)
    decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded_text

def elaborate(llm, context: str, prompt: str) -> str:
    elaborate_prompt = f"Please provide a detailed explanation on the following topic: {prompt}. Here is the context: {context}"
    response = llm(MODEL_DIR, elaborate_prompt)
    return response

def tldr(llm, context: str) -> str:
    tldr_prompt = f"Please provide a summary of the following content: {context}"
    response = llm(MODEL_DIR, tldr_prompt)
    return response

class Query(BaseModel):
    prompt: str
    subject: str

@app.route('/api/query', methods=['POST'])
def query_rag():
    try:
        query = Query(**request.json)
    except ValidationError as e:
        return jsonify(e.errors()), 400

    subject = query.subject.lower()
    subject_folder = os.path.join(DATASET_DIR, subject)
    if not os.path.isdir(subject_folder):
        return jsonify({"error": "Subject not found"}), 404

    try:
        index = create_faiss_index_for_subject(subject)
        sub_db = move_storage_files(subject)
        prom = "what is amines"
        context_nodes = search_faiss_index(prom, sub_db, subject) # query.prompt --> prom

        if isinstance(context_nodes, dict) and "error" in context_nodes:
            return jsonify(context_nodes), 500

        context_text = " ".join([node.text for node in context_nodes])

        if not context_text:
            return jsonify({"error": "No relevant context found."}), 404

        detailed_response = elaborate(llm, context_text, query.prompt)
        summary_response = tldr(llm, context_text)

        return jsonify({
            "response": context_text,
            "detailed_response": detailed_response,
            "summary_response": summary_response
        })

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/initialize', methods=['POST'])
def initialize():
    try:
        for subject in os.listdir(DATASET_DIR):
            if os.path.isdir(os.path.join(DATASET_DIR, subject)):
                create_faiss_index_for_subject(subject)
        return jsonify({"message": "Initialization completed successfully."}), 200
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    try:
        subject = request.form['subject'].lower()
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and file.filename.endswith('.pdf'):
            subject_folder = os.path.join(DATASET_DIR, subject)
            os.makedirs(subject_folder, exist_ok=True)
            file_path = os.path.join(subject_folder, file.filename)
            file.save(file_path)
            return jsonify({"message": "File uploaded successfully"}), 200
        else:
            return jsonify({"error": "Invalid file format"}), 400

    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    config_path = os.path.join(MODEL_DIR, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['hidden_activation'] = 'gelu_pytorch_tanh'
        with open(config_path, 'w') as f:
            json.dump(config, f)

    app.run(host='0.0.0.0', port=8000)
