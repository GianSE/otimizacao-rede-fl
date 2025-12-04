# config.py

# --- Configurações de Treinamento ---
NUM_ROUNDS = 3          # Número de rodadas de agregação federada
NUM_CLIENTS = 2         # Número total de clientes na simulação
EPOCHS = 1              # Número de épocas de treino local por rodada
MAX_STEPS = 50          # Limite de passos por treino (None para treinar a época toda)
BATCH_SIZE = 4          # Tamanho do lote (batch size)
LEARNING_RATE = 5e-4    # Taxa de aprendizado

# --- Configurações de Dados ---
# Se True, usa apenas uma fatia pequena dos dados para teste rápido
# Se False, usa o dataset completo
USE_SMALL_DATASET = True 
SMALL_TRAIN_SIZE = 100  # Quantas amostras usar se USE_SMALL_DATASET for True
SMALL_TEST_SIZE = 20

# --- Caminhos ---
MODEL_NAME = "./meu_modelo_torch"
DATASET_PATH = "./meu_dataset_processado"

# --- Configurações de Rede ---
SERVER_ADDRESS = "127.0.0.1:8080"