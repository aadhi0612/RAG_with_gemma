from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from time import time
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset

data = "/home/ubuntu/RAG/datasets/TXTs/*.txt"


def prepare_model():
    model_id = "/home/ubuntu/RAG/models/"  # Assuming gemma is a directory containing the model files

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    time_1 = time()
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    time_2 = time()
    print(f"Prepare model, tokenizer: {round(time_2-time_1, 3)} sec.")
    return model, tokenizer

def test_model(tokenizer, model, prompt_to_test):
    """
    Perform text generation using the provided model and tokenizer.
    Args:
        tokenizer: the tokenizer
        model: the language model
        prompt_to_test (str): the prompt text to generate from
    """
    time_1 = time()
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    input_ids = tokenizer.encode(prompt_to_test, return_tensors="pt").to(model.device)
    sequences = model.generate(input_ids,
                               max_length=1000000,
                               num_return_sequences=1,
                               no_repeat_ngram_size=2,
                               eos_token_id=tokenizer.eos_token_id,
                               top_k=50,
                               do_sample=False, 
                               )
    time_2 = time()
    print(f"Test inference: {round(time_2 - time_1, 3)} sec.")

    for seq in sequences:
        generated_text = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"Result: {generated_text}")
        
        
model, tokenizer = prepare_model()
test_model(tokenizer, model, "Please explain what is the State of the Union address. Give just a definition. Keep it in 500 words.")

def document_load_split():
    loader = TextLoader("/home/ubuntu/RAG/datasets/TXTs/Accounts101.txt", encoding="utf8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)
    embed_text(all_splits)
    return all_splits
    
def embed_text(all_splits):
    
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {"device": "cuda"}  
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
      
    db = FAISS.from_documents(all_splits, embeddings)
    print(f"DB: {db}")
    return db


model, tokenizer = prepare_model()
pipe = pipeline( "text-generation", model=model, tokenizer=tokenizer, return_tensors='pt', max_length=1000000, model_kwargs={"torch_dtype": torch.bfloat16}, )

llm = HuggingFacePipeline(
    pipeline=pipe,
    model_kwargs={"temperature": 0.7, "max_length": 1000000},
)
all_splits = document_load_split()
db = embed_text(all_splits)
qa = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff", retriever=db.as_retriever())


     
if __name__ == "__main__":
    all_splits = document_load_split()
    vectordb = embed_text(all_splits)
    qa.invoke("explain accounts in page1")



 