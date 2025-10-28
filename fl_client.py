# Hack para o erro de DLL (SEMPRE no topo)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
import flwr as fl
import math
from transformers import TFGPT2LMHeadModel # Importamos o modelo TF

# Importando (agora carrega o modelo TF)
from model_definition import load_model_and_tokenizer

# --- Funções de Treino/Teste (Estilo Keras) ---

def train(model, tf_dataset, epochs):
    """Treina o modelo Keras no dataset local do cliente."""
    
    print(f"Iniciando treino local por {epochs} época(s)...")
    
    # O modelo já foi compilado no __init__ do cliente
    history = model.fit(
        tf_dataset,
        epochs=epochs,
        verbose=2 # Mostra o log de treino
    )
    
    loss = history.history["loss"][-1]
    print(f"Treino local concluído. Perda (Loss) final: {loss:.4f}")

def test(model, tf_dataset, num_test_examples):
    """
    Avalia o modelo Keras no dataset de teste local.
    Métrica: Perplexidade (quanto menor, melhor).
    """
    print("Iniciando avaliação local...")
    
    # model.evaluate() retorna a métrica de perda (loss)
    avg_loss = model.evaluate(tf_dataset, verbose=0)
    
    # Perplexidade é o exponencial da perda média
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")
        
    print(f"Avaliação local concluída. Perda Média: {avg_loss:.4f}, Perplexidade: {perplexity:.4f}")
    
    # Retornamos a perda, perplexidade e o número de exemplos
    return avg_loss, perplexity, num_test_examples

# --- Helper para converter dataset do Hugging Face para tf.data.Dataset ---

def create_tf_dataset(dataset, batch_size, shuffle=False):
    """
    Converte um dataset do Hugging Face (com formato "tf") 
    em um tf.data.Dataset pronto para o Keras (model.fit).
    """
    
    def format_batch_for_keras(batch):
        # Os modelos TF da Hugging Face esperam as features
        # como um dicionário no primeiro argumento (x)
        features = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"]
        }
        # E os rótulos como o segundo argumento (y)
        labels = batch["labels"]
        return features, labels
    
    if shuffle:
        dataset = dataset.shuffle(seed=42)
        
    # Usa o método .to_tf_dataset() da biblioteca 'datasets'
    tf_ds = dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        batch_size=batch_size,
        collate_fn=None # O método já cuida do batch
    )
    
    # Mapeia para o formato (features, labels) que o Keras espera
    return tf_ds.map(format_batch_for_keras)

# --- Definição do Cliente Flower (Adaptado para Keras) ---

class FlowerClient(fl.client.NumPyClient):
    """
    Define a lógica do cliente FL para o modelo Keras.
    """
    def __init__(self, model, train_dataset, test_dataset):
        self.model = model
        self.train_dataset = train_dataset # Salva o dataset HF
        self.test_dataset = test_dataset   # Salva o dataset HF
        
        # Compila o modelo Keras.
        # Os modelos TF da HF calculam a perda internamente quando
        # os 'labels' são fornecidos. O Keras gerencia isso.
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5))
        
        # BATCH_SIZE precisa ser pequeno (ex: 4 ou 8)
        BATCH_SIZE = 4
        
        # Converte os datasets do Hugging Face em tf.data.Dataset
        self.tf_train_loader = create_tf_dataset(
            train_dataset, BATCH_SIZE, shuffle=True
        )
        self.tf_test_loader = create_tf_dataset(
            test_dataset, BATCH_SIZE, shuffle=False
        )

    def get_parameters(self, config):
        """Retorna os pesos atuais do modelo (Keras)."""
        print("[Cliente] Enviando parâmetros (Keras weights) para o servidor.")
        # .get_weights() já retorna uma lista de arrays NumPy
        return self.model.get_weights()

    def set_parameters(self, parameters):
        """Atualiza o modelo local com os pesos (Keras)."""
        print("[Cliente] Recebendo parâmetros (Keras weights) do servidor.")
        # .set_weights() aceita uma lista de arrays NumPy
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        """Treina o modelo Keras localmente."""
        print("[Cliente] Iniciando 'fit' (treinamento Keras)...")
        self.set_parameters(parameters) 
        
        # Chama a função de treino do Keras
        train(self.model, self.tf_train_loader, epochs=1) 
        
        # Retorna os pesos atualizados
        return self.get_parameters(config={}), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        """Avalia o modelo Keras localmente."""
        print("[Cliente] Iniciando 'evaluate' (avaliação Keras)...")
        self.set_parameters(parameters) 
        
        # Chama a função de teste do Keras
        loss, perplexity, num_examples = test(
            self.model, self.tf_test_loader, len(self.test_dataset)
        )
        
        # Flower espera a 'loss' primeiro
        return loss, num_examples, {"perplexity": perplexity}


if __name__ == '__main__':
    # Teste rápido para o fl_client.py (versão TF)
    
    print("--- Testando o fl_client.py (TensorFlow) ---")
    
    print("Carregando modelo e tokenizer...")
    model, _ = load_model_and_tokenizer()
    
    print("Carregando dados (versão de teste)...")
    from data_loader import load_data
    train_data, test_data = load_data()
    
    # Pega subconjuntos pequenos para um teste rápido
    dummy_train = train_data.select(range(20))
    dummy_test = test_data.select(range(10))

    print("Iniciando cliente de teste...")
    client_teste = FlowerClient(model, dummy_train, dummy_test)
    
    print("\nTestando client.fit()...")
    params_iniciais = client_teste.get_parameters(config={})
    params_novos, num_exemplos, _ = client_teste.fit(params_iniciais, config={})
    print(f"Fit concluído. {num_exemplos} amostras processadas.")

    print("\nTestando client.evaluate()...")
    loss, num_ex, metrics = client_teste.evaluate(params_novos, config={})
    print(f"Evaluate concluído. Loss: {loss:.4f}, Perplexity: {metrics['perplexity']:.4f}")
    
    print("\n--- Teste de fl_client.py (TF) concluído com sucesso! ---")