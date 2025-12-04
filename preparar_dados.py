# preparar_dados.py
import os
from datasets import load_dataset
from model_definition import load_model_and_tokenizer

# Nome da pasta onde vamos salvar o dataset pronto
PASTA_DATASET = "./meu_dataset_processado"
BLOCK_SIZE = 128

def main():
    if os.path.exists(PASTA_DATASET):
        print(f"O dataset já existe em '{PASTA_DATASET}'. Apague a pasta se quiser recriar.")
        return

    print("--- 1. Carregando Tokenizer ---")
    # Ele vai carregar da sua pasta local './meu_modelo_torch' se você já baixou
    _, tokenizer = load_model_and_tokenizer()

    print("--- 2. Baixando Dataset 'blended_skill_talk' ---")
    raw_datasets = load_dataset("blended_skill_talk")

    print("--- 3. Processando e Tokenizando (Isso demora um pouco...) ---")
    
    def preprocess_function(examples):
        separator = tokenizer.eos_token 
        texts = []
        for i in range(len(examples["previous_utterance"])):
            context_list = examples["previous_utterance"][i]
            message_list = examples["free_messages"][i]
            full_dialog_list = context_list + message_list
            full_text = separator.join(full_dialog_list) + separator
            texts.append(full_text)

        tokenized_output = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=BLOCK_SIZE,
            return_tensors="pt"
        )
        tokenized_output["labels"] = tokenized_output["input_ids"].clone()
        return tokenized_output

    # Remove as colunas de texto original para economizar espaço
    column_names = raw_datasets["train"].column_names 
    
    tokenized_datasets = raw_datasets.map(
        preprocess_function, 
        batched=True, 
        remove_columns=column_names 
    )

    print(f"--- 4. Salvando processado em '{PASTA_DATASET}' ---")
    tokenized_datasets.save_to_disk(PASTA_DATASET)
    print("✅ Sucesso! Dataset salvo e pronto para uso.")

if __name__ == "__main__":
    main()