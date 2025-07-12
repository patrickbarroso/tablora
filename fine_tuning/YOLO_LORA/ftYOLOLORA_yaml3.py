import torch
from ultralytics import YOLO  # Use 'ultralytics' em vez de 'ultralyticsplus' se possível
from peft import LoraConfig, LoraModel
import time
import logging
import sys

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/ROOT_PATH/yoloft.log"),
        logging.StreamHandler()
    ]
)

#iniciando marcação de tempo
start_time = time.time()

dir_modelo_ajustado = '/ROOT_PATH/FT_YOLO_LORA/Model/teste'
arquivo_yolo_yaml = '/ROOT_PATH/FT_YOLO_LORA/main/yolo.yaml'

# Configuração do dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Carregar modelo YOLO (padrao)
model = YOLO('yolov8n.pt').to(device)

print("Antes de aplicar LoRA:", type(model))

# Configuração do LoRA para YOLO
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Selecionando camadas específicas para aplicar o LoRA

'''        
LORA_TARGET_MODULES = [
'model.model.12.cv2.conv',
'model.model.12.m.0.cv1.conv',
'model.model.12.m.0.cv2.conv',   
'model.model.21.cv2.conv',
'model.model.21.m.0.cv1.conv',
'model.model.21.m.0.cv2.conv',
'model.model.22.cv3.2.0.conv', 
'model.model.22.dfl.conv', 
'model.model.22.cv3.0.1.conv']  
'''

# Módulos mais adequados para YOLOv8
LORA_TARGET_MODULES = [
    "model.model.12.m.0.cv1.conv",  # Última camada convolucional
]


# Ajuste com base nas camadas do YOLO

# Adicionar LoRA manualmente às camadas especificadas
# ========================================================================
# Aqui começamos o processo de aplicação manual do LoRA nas camadas do YOLO.
# ========================================================================


for i, (name, module) in enumerate(model.named_modules(), start=1):
    if name in LORA_TARGET_MODULES:
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

# Configurações de treinamento com a API YOLO
epochs = 50
batch = 5  # Substituído 'batch_size' por 'batch'
learning_rate = 4e-4
imgsz = 640

print("Depois de aplicar LoRA:", type(model))

sys.exit()

# Iterar sobre os parâmetros do modelo para verificar quais foram ajustados pelo LoRA
print("Parâmetros ajustados pelo LoRA:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)


# Treinar com a API YOLO sem alterar o tipo do modelo

results = model.train(
    data=arquivo_yolo_yaml,  # Configuração dos dados do YOLO
    epochs=epochs,                   # Número de épocas
    imgsz=imgsz,                       # Tamanho da imagem
    batch=batch,                     # Tamanho do lote
    lr0=learning_rate,               # Taxa de aprendizado inicial
    device=device                    # Dispositivo para o treinamento
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

#salvando o modelo completo
print("Salvando o modelo completo via model.export()...")
#logging.info("Salvando o modelo completo via model.export()...")
model.export(format='torchscript', imgsz=640, save_dir=dir_modelo_ajustado) #comentei aqui para testar save via LORA
#model.save(f'{dir_modelo_ajustado}/ftYOLO_lora_teste.pt')

print("Salvando o modelo formato LORA...")
# Salvar corretamente o modelo LoRA ajustado
model.save_pretrained(dir_modelo_ajustado)

duration_time = (time.time() - start_time)/60
#print(f'\nDuration: {duration_time:.0f} minutes') # print the time elapsed  
#logging.info(f'\nDuration: {duration_time:.0f} minutes') # print the time elapsed  

print(f'\nDuration: {duration_time:.0f} minutes') # print the time elapsed  
