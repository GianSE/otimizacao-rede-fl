import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Nome da pasta onde vamos salvar
PASTA_LOCAL = "./meu_modelo_torch"

# Cria a pasta se não existir
if not os.path.exists(PASTA_LOCAL):
    os.makedirs(PASTA_LOCAL)

print(f"Baixando modelo (PyTorch) e salvando em '{PASTA_LOCAL}'...")

model_name = "distilgpt2"

# 1. Baixar e Salvar o Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(PASTA_LOCAL)

# 2. Baixar e Salvar o Modelo
model = GPT2LMHeadModel.from_pretrained(model_name)
model.save_pretrained(PASTA_LOCAL)

print("✅ Sucesso! Modelo salvo localmente.")