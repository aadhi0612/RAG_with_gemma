#filepath/main.py
import transformers
from transformers import AutoTokenizer
from time import time
import torch

# Path to the gemma model directory
model_id = '/home/ubuntu/RAG/models/'

# Define the BitsAndBytesConfig for quantization
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Measure time to prepare model and tokenizer
time_1 = time()

# Load model configuration
model_config = transformers.AutoConfig.from_pretrained(model_id)
model_config.hidden_act = "gelu_pytorch_tanh"  # Change activation function

# Load model with quantization configuration
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

time_2 = time()
print(f"Prepare model and tokenizer: {round(time_2 - time_1, 3)} sec.")

def test_model(tokenizer, model, prompt_to_test):
    """
    Perform text generation using the provided model and tokenizer.
    Args:
        tokenizer: the tokenizer
        model: the language model
        prompt_to_test (str): the prompt text to generate from
    """
    time_1 = time()
    input_ids = tokenizer.encode(prompt_to_test, return_tensors="pt").to(model.device)
    sequences = model.generate(input_ids,
                               max_length=100,
                               num_return_sequences=1,
                               no_repeat_ngram_size=2,
                               eos_token_id=tokenizer.eos_token_id,
                               top_k=50,
                               do_sample=True, 
                               )
    time_2 = time()
    print(f"Test inference: {round(time_2 - time_1, 3)} sec.")

    for seq in sequences:
        generated_text = tokenizer.decode(seq, skip_special_tokens=True)
        print(f"Result: {generated_text}")

# Test the model with a prompt
test_model(tokenizer, model, "Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words.")

