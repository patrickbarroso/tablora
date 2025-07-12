import torch
import json
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import glob
import shutil
import os
from PIL import Image
import cv2
import time

#iniciando marcação de tempo
start_time = time.time()

# Diretórios do novo dataset de tabelas
cenario = "YOLO11n_18062025SP2_SEMLORA"
dir_modelo_ajustado = '/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_TRAIN/Model/'
arquivo_yolo_yaml = '/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_TRAIN/yolo.yaml'

# Configuração do dispositivo
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

YOLO11N = '/home/aluno-pbarroso/yolo11n.pt'

# Passo 1: Carregar o modelo YOLO pré-treinado específico para detecção de tabelas
try:
    model = YOLO(YOLO11N).to(device)
    print("modelo YOLO pre-treinado carregado!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

# Configurações de treinamento com a API YOLO
epochs = 100
batch = 5  # Substituído 'batch_size' por 'batch'
learning_rate = 4e-4
imgsz = 640
patience = 5

print(f"CENÁRIO {cenario} - FINETUNING TRADICIONAL SEM LORA")
print(f"MODELO {YOLO11N}")
print(f"EPOCAS: {epochs}")
print(f"BATCH: {batch}")
print(f"LEARNING RATE: {learning_rate}")
print(f"imgsz: {imgsz}")
print(f"PATIENCE: {patience}")

print("================INICIANDO TREINAMENTO  DO MODELO=========")
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

print("================INICIANDO VALIDAÇÃO DO MODELO=========")
metrics = model.val(data=arquivo_yolo_yaml)
print(metrics)

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
        path_modelo_ajustado = os.path.join(dir_modelo_ajustado, f"{cenario}.pt")

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

'''
print("================INICIANDO PREDIÇÃO DO MODELO=========")
path_in = '/home/aluno-pbarroso/pytorch-pbarroso/ft_tatr/Predict/LAB_04_001.jpg'
path_img_marked_ft = '/home/aluno-pbarroso/pytorch-pbarroso/ft_tatr/Predict/LAB_04_001_ft_marked.jpg'

print("Iniciando predição....")
image = Image.open(path_in)
results = model_pre_train.predict(image)

for result in results:
    #print(result)
    boxes = result.boxes.xyxy.tolist()
    print("boxes ", boxes)

    # Obter a imagem com as detecções desenhadas
    img_with_detections = result.plot()  # Este método retorna uma imagem com as detecções

    # Salvar a imagem resultante com as marcações
    cv2.imwrite(path_img_marked_ft, img_with_detections)


print("Salvando o modelo....")
model_pre_train.export(format='torchscript', imgsz=640, save_dir=dir_modelo_ajustado)

#comentado
#salvando apenas os pesos do modelo
print("Salvando os pesos do modelo - yolov8n_ft_pesos.pt")
torch.save(model_pre_train.model.state_dict(), f'{dir_modelo_ajustado}/yolov8n_ft_pesos.pt')
print("Salvando o modelo completo via export...")
#salvando o modelo completo
model_pre_train.export(format='torchscript', imgsz=640, save_dir=dir_modelo_ajustado)
#Salvar o modelo completo usando PyTorch
print("Salvando o modelo completo usando PyTorch - yolov8n_ft_full.pt")
torch.save(model_pre_train.model, f'{dir_modelo_ajustado}/yolov8n_ft_full.pt')
#model_pre_train.save(f'{dir_modelo_ajustado}/yolov8n_ft_full.pt')

'''


