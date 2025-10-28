import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "distilgpt2" # Um modelo de linguagem pequeno e rápido

def load_model_and_tokenizer():
    """
    Carrega o modelo de linguagem pré-treinado e o tokenizer
    da Hugging Face (versão TensorFlow).
    """
    
    # 1. Carregar o Tokenizer (É o mesmo para TF ou PyTorch)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Carregar o Modelo (VERSÃO TENSORFLOW)
    # TFGPT2LMHeadModel = "Language Model Head"
    model = TFGPT2LMHeadModel.from_pretrained(MODEL_NAME)
    
    print(f"Modelo pré-treinado '{MODEL_NAME}' (TensorFlow) e tokenizer carregados.")
    
    return model, tokenizer

if __name__ == '__main__':
    # Teste rápido para ver se funciona
    model, tokenizer = load_model_and_tokenizer()
    
    # Vamos testar
    prompt = "Olá, tudo bem?"
    print(f"\nTestando o modelo com o prompt: '{prompt}'")
    
    # Converte o texto em tensores TensorFlow (return_tensors="tf")
    inputs = tokenizer(prompt, return_tensors="tf")
    
    # Gera 15 novos tokens (palavras)
    # A API .generate() é a mesma
    output_ids = model.generate(inputs.input_ids, max_length=15)
    
    # Converte os IDs de volta em texto
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"Resposta gerada (antes do treino): '{output_text}'")