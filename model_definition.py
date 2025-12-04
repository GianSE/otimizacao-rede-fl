import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "distilgpt2" # Um modelo de linguagem pequeno e rápido

def load_model_and_tokenizer():
    """
    Carrega o modelo de linguagem pré-treinado e o tokenizer
    da Hugging Face.
    """
    
    # 1. Carregar o Tokenizer
    # O tokenizer converte "olá mundo" em [31373, 995]
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    
    # Modelos GPT-2 não têm um token de "padding" (preenchimento) por padrão.
    # Vamos defini-lo como o token "end of sentence" (fim da sentença).
    # Isso é necessário para treinar em lotes (batches).
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Carregar o Modelo
    # GPT2LMHeadModel = "Language Model Head", o que significa que ele
    # é feito para prever a próxima palavra (GenAI).
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    
    print(f"Modelo pré-treinado '{MODEL_NAME}' e tokenizer carregados.")
    
    return model, tokenizer

if __name__ == '__main__':
    # Teste rápido para ver se funciona
    model, tokenizer = load_model_and_tokenizer()
    
    # Vamos testar
    prompt = "Olá, tudo bem?"
    print(f"\nTestando o modelo com o prompt: '{prompt}'")
    
    # Converte o texto em números (IDs)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Gera 15 novos tokens (palavras)
    output_ids = model.generate(inputs.input_ids, max_length=15)
    
    # Converte os IDs de volta em texto
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"Resposta gerada (antes do treino): '{output_text}'")