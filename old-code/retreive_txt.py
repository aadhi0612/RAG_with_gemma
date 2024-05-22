from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer, pipeline
from time import time
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer

CACHE_DIR = "/home/ubuntu/RAG/CACHE"


def prepare_model(model_id):
    """
    Load and prepare the language model and tokenizer.
    """
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


def test_model(tokenizer, model, prompt_to_test):
    """
    Perform text generation using the provided model and tokenizer.
    """
    input_ids = tokenizer.encode(prompt_to_test, return_tensors="pt").to(model.device)
    sequences = model.generate(
        input_ids,
        max_length=1000000,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        top_k=50,
        do_sample=False
    )

    for seq in sequences:
        generated_text = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"Generated text: {generated_text}")

#################




class Encoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2", device="cpu"):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=CACHE_DIR,
            model_kwargs={"device": device})

class FaissDb:
    def __init__(self, docs, embedding_function):
        self.db = FAISS.from_documents(docs, embedding_function, distance_strategy=DistanceStrategy.COSINE)

    def similarity_search(self, question: str, k: int = 3):
        retrieved_docs = self.db.similarity_search(question, k=k)
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        return context

def load_and_split_pdfs(file_paths: list, chunk_size: int = 256):
    loaders = [PyPDFLoader(file_path) for file_path in file_paths]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2"),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        strip_whitespace=True)
        
    docs = text_splitter.split_documents(pages)
    return docs

#################


def embed_text(all_splits, model_name):
    """
    Embed the text using HuggingFace embeddings and store in FAISS.
    """
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    db = FAISS.from_documents(all_splits, embeddings)
    print(f"DB: {db}")
    return db

if __name__ == "__main__":
    model_id = "/home/ubuntu/RAG/models/"
    input_file = "/home/ubuntu/RAG/datasets/TXTs/Accounts101.txt"
    model, tokenizer = prepare_model(model_id)
    # test_model(tokenizer, model, "Please explain what is the State of the Union address. Give just a definition. Keep it in 500 words.")

    all_splits = document_load_split(input_file)
    db = embed_text(all_splits, "sentence-transformers/all-MiniLM-l6-v2")
    pipe = pipeline( "text-generation", model=model, tokenizer=tokenizer, return_tensors='pt', max_length=1000000, model_kwargs={"torch_dtype": torch.bfloat16},)
    
    llm = HuggingFacePipeline( pipeline=pipe, model_kwargs={"temperature": 0.7, "max_length": 1000000}, )
    
    # qa = RetrievalQA.from_chain_type( llm=HuggingFacePipeline(model=model, tokenizer=tokenizer), chain_type="stuff", retriever=db.as_retriever())
    qa = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff", retriever=db.as_retriever())

    qa.invoke("explain accounts in page1")

    query = "explain accounts in page1"
    response = pipeline(query)
    print(f"Pipeline response: {response}")

    # Extract generated text from response
    generated_text = response.get("generated_text", None)
    if generated_text:
        print(f"Generated text: {generated_text}")
    else:
        print("Generated text not found in response.")