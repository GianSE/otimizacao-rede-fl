import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Aponta para sua pasta local
MODEL_NAME = "./meu_modelo_torch"

def load_model_and_tokenizer():
    """
    Carrega o modelo base e o tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    return model, tokenizer

def apply_lora(model):
    """
    Aplica a adaptação LoRA no modelo.
    Isso reduz os parâmetros treináveis de 82 milhões para ~300 mil.
    """
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8,            # "Rank" da matriz (quanto maior, mais inteligente, mas mais pesado)
        lora_alpha=32,  # Fator de escala
        lora_dropout=0.1
    )
    
    # Envolve o modelo base com a configuração LoRA
    lora_model = get_peft_model(model, peft_config)
    
    # Imprime quanto economizamos
    lora_model.print_trainable_parameters()
    
    return lora_model

if __name__ == '__main__':
    # Teste para ver a redução
    model, _ = load_model_and_tokenizer()
    lora_model = apply_lora(model)