from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import sys

# Caminho para o arquivo de configuração do dataset
arquivo_yolo_yaml = '/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_LORA/main/yolo.yaml'

# Carrega o modelo (coloque o caminho para o modelo treinado)
model_path = '/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_LORA/CV_YOLO11/results/YOLO-CEN04_trial_14_fold0/weights/best.pt'
#model_path = "yolov8n.pt"

# Carregar o modelo YOLO
model = YOLO(model_path)  # Carrega o modelo a partir do arquivo YAML
# Ou, se você tiver um modelo treinado:
# model = YOLO('/caminho/para/seu/modelo_treinado.pt')

batch_size = 8
imgsz = 640
conf_thres = 0.25  # Limiar de confiança para detecção
iou_thres = 0.45   # Limiar de IOU para NMS

# Realizar a validação

results = model.val(
    data=arquivo_yolo_yaml,
    batch=batch_size,
    imgsz=imgsz,
    conf=conf_thres,
    iou=iou_thres,
    split='val',  # Usar o split de validação
    save_json=True,  # Salvar resultados em JSON para análise
    save_conf=True,  # Salvar scores de confiança
    plots=True      # Gerar gráficos de métricas
)

# Extrair métricas
metrics = results.results_dict
precision = metrics['metrics/precision(B)']
recall = metrics['metrics/recall(B)']
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Função auxiliar para formatar no estilo europeu
def format_european(value):
    return f"{value:.5f}".replace('.', ',')

print("metrics = ", metrics)

# Exibir resultados
print("\n" + "="*50)
print("Métricas de Avaliação do Modelo YOLO")
print("="*50)
print(f"Precision: {format_european(precision)}")
print(f"Recall: {format_european(recall)}")
print(f"F1-score: {format_european(f1_score)}")
print("="*50 + "\n")

# Explicação das métricas
print("Interpretação das Métricas:")
print("- Precision: Proporção de detecções positivas que são realmente verdadeiras")
print("- Recall: Proporção de objetos verdadeiros que foram detectados")
print("- F1-score: Média harmônica entre Precision e Recall (ideal próximo de 1)")