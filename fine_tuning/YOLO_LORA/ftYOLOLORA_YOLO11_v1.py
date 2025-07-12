import torch
from ultralytics import YOLO  # Use 'ultralytics' em vez de 'ultralyticsplus' se possível
from peft import LoraConfig, LoraModel
import time
import logging
import sys
import os
import shutil
import re

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_LORA/main/yoloft.log"),
        logging.StreamHandler()
    ]
)

#iniciando marcação de tempo
start_time = time.time()

cenario = "YOLO11n_LORA_23062025SP5"
path_modelo_ajustado = f'/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_LORA/Model/'
arquivo_yolo_yaml = '/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_LORA/main/yolo.yaml'

# Configuração do dispositivo
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

YOLO11N = '/home/aluno-pbarroso/yolo11n.pt'

# Carregar modelo YOLO 11
try:
    model = YOLO(YOLO11N).to(device)
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    sys.exit(1)

print("Antes de aplicar LoRA:", type(model))

# Configuração do LoRA para YOLO
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

'''
CAMADAS YOLOV11

BACKBONE (FEATURE EXTRACTION)= model.0 a model.12
NECK (FEATURE FUSION)= model.12 a model.22
HEAD (DETECT)= model.23

'''

# Selecionando camadas específicas para aplicar o LoRA
LORA_TARGET_MODULES_CENARIO1 = [

    "model.model.10.cv1.conv",
    "model.model.10.cv2.conv",
    "model.model.10.m.0.attn.qkv.conv", 
    "model.model.10.m.0.attn.proj.conv",
    "model.model.10.m.0.attn.pe.conv",
    "model.model.10.m.0.ffn.0.conv",
    "model.model.10.m.0.ffn.1.conv",
    "model.model.17.cv1.conv",
    "model.model.17.cv2.conv",
    "model.model.18.cv1.conv",
    "model.model.18.cv2.conv",
    "model.model.19.cv1.conv",
    "model.model.19.cv2.conv",
    "model.model.20.cv1.conv",
    "model.model.20.cv2.conv",
    "model.model.21.cv1.conv",
    "model.model.21.cv2.conv",
    "model.model.22.cv1.conv",
    "model.model.22.cv2.conv",
    "model.model.23.cv1.conv",
    "model.model.23.cv2.conv",
    "model.model.23.dfl.conv" 

]  

LORA_TARGET_MODULES = LORA_TARGET_MODULES_CENARIO1

############# PRINT DAS CONFIGURACOES ###############

# Configurações de treinamento com a API YOLO
epochs = 100
batch = 5  # Substituído 'batch_size' por 'batch'
learning_rate = 4e-4
imgsz = 640
patience = 5

print(f"CENÁRIO {cenario}")
print(f"MODELO {YOLO11N}")
print(f"LORA_TARGET_MODULES {LORA_TARGET_MODULES}")
print(f"LORA_R: {LORA_R}")
print(f"LORA_ALPHA: {LORA_ALPHA}")
print(f"LORA_DROPOUT: {LORA_DROPOUT}")
print(f"EPOCAS: {epochs}")
print(f"BATCH: {batch}")
print(f"LEARNING RATE: {learning_rate}")
print(f"imgsz: {imgsz}")
print(f"PATIENCE: {patience}")

# Ajuste com base nas camadas do YOLO

# Adicionar LoRA manualmente às camadas especificadas
# ========================================================================
# Aqui começamos o processo de aplicação manual do LoRA nas camadas do YOLO.
# ========================================================================

# Primeiro, coletar os módulos que serão alterados
modules_to_modify = {name: module for name, module in model.named_modules() if name in LORA_TARGET_MODULES}

for name, module in modules_to_modify.items():
    # Criar uma configuração LoRA específica para a camada atual
    print(f"Tratando camada {name}...")
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[name],
        lora_dropout=LORA_DROPOUT,
        bias="none"
    )
    
    # Criar uma instância de LoraModel para a camada e aplicar LoRA
    # Isso adiciona o LoRA apenas nesta camada específica sem alterar o tipo do modelo principal
    #adapter_name = f"adapter_{i}"
    lora_module = LoraModel(model, config, "default")
    
    # Substituir a camada original do YOLO pela versão com LoRA
    # ========================================================================
    # A linha abaixo substitui a camada original pela versão com LoRA,
    # mantendo a estrutura YOLO do modelo sem transformá-lo em PeftModel.
    # ========================================================================

    setattr(model, name, lora_module)

#print("Depois de aplicar LoRA:", type(model))

# Iterar sobre os parâmetros do modelo para verificar quais foram ajustados pelo LoRA
print("Parâmetros ajustados pelo LoRA (mantendo estrutura do YOLO - sem alterar para PEFT):")
#print("LORA_TARGET_MODULES ", LORA_TARGET_MODULES)
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name)

# Contadores
trainable_params = 0
total_params = sum(p.numel() for p in model.parameters())

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} - {param.numel():,} parâmetros")
        trainable_params += param.numel()

print("\nResumo:")
print(f"Total de parâmetros treináveis: {trainable_params:,}")
print(f"Total de parâmetros do modelo: {total_params:,}")
print(f"Percentual de parâmetros treináveis: {100 * trainable_params / total_params:.2f}%")

if trainable_params == 0:
    print(f"Nenhum módulo LORA reconhecido, favor verificar.")
    sys.exit()

#sys.exit()

# Treinar com a API YOLO sem alterar o tipo do modelo
results = model.train(
    data=arquivo_yolo_yaml,  # Configuração dos dados do YOLO
    epochs=epochs,                   # Número de épocas
    imgsz=imgsz,                       # Tamanho da imagem
    batch=batch,                     # Tamanho do lote
    lr0=learning_rate,               # Taxa de aprendizado inicial
    device=device,                    # Dispositivo para o treinamento
    patience=patience,  # Early stopping com paciência de 5 épocas
    # Configurações adicionais para melhor controle do early stopping
    save_period= 1,  # Salvar checkpoints a cada época para monitoramento
    plots= True     # Gerar gráficos de métricas
)

print("================================Iniciando o teste...===========================")
#logging.info("================================Iniciando o teste...===========================")
metrics = model.val(data=arquivo_yolo_yaml)
print("================================Imprimindo as métricas ...===========================")
#logging.info("================================Imprimindo as métricas...===========================")
print(metrics)
#logging.info(metrics)

# Extrair as métricas de precisão e F1-score
precision = metrics.results_dict['metrics/precision(B)']  
recall = metrics.results_dict['metrics/recall(B)']  

# Calcular F1-Score
f1_score = 2 * (precision * recall) / (precision + recall)

print("precision = ",precision)
print("recall = ", recall)
print("f1_score = ", f1_score)
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")

# === Após o treino, capturar o caminho do modelo treinado ===

output_dir = getattr(results, 'save_dir', None)
if (output_dir is None):
    output_dir = getattr(model.trainer, 'save_dir', None)

print ("output_dir =", output_dir)

if output_dir and os.path.exists(output_dir):
    print(f"Diretório salvo automaticamente: {output_dir}")

    # Caminho completo do modelo best.pt gerado
    best_model_path = os.path.join(output_dir, 'weights', 'best.pt')

    if os.path.exists(best_model_path):
        # Caminho de destino parametrizado para salvar o modelo ajustado
        path_modelo_ajustado = os.path.join(path_modelo_ajustado, f"{cenario}.pt")

        # Criar diretório de destino se não existir
        os.makedirs(os.path.dirname(path_modelo_ajustado), exist_ok=True)

        # Copiar best.pt para o local parametrizado
        shutil.copy2(best_model_path, path_modelo_ajustado)

        print(f"Modelo best.pt copiado para: {path_modelo_ajustado}")
    else:
        logging.warning(f"O arquivo best.pt não foi encontrado em: {best_model_path}")
else:
    logging.warning("Diretório de saída não encontrado em results.save_dir")


duration_time = (time.time() - start_time)/60
#print(f'\nDuration: {duration_time:.0f} minutes') # print the time elapsed  
#logging.info(f'\nDuration: {duration_time:.0f} minutes') # print the time elapsed  

print(f'\nDuration: {duration_time:.0f} minutes') # print the time elapsed  