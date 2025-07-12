import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from peft import get_peft_model, LoraConfig
from sklearn.metrics import precision_score, recall_score, f1_score
#from yolov5 import YOLOv5  # YOLOv5 package ou custom loader de weights
from ultralyticsplus import YOLO, render_result
#pip install ultralyticsplus==0.0.28 ultralytics==8.0.43


# Diretórios da base COCO e os novos dados de tabelasp
diretorio_imagens_tabelas = '/caminho/para/imagens/tabelas'
diretorio_rotulos_tabelas = '/caminho/para/rotulos/tabelas.json'

# Verificar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Passo 1: Carregar modelo YOLO pré-treinado específico para detecção de tabelas
modelo_tabelas_pretreinado = YOLO('foduucom/table-detection-and-extraction').to(device)

# Passo 2: Configurar LoRA
lora_config = LoraConfig(
    r=8,  # Rank de atualização
    lora_alpha=16,  # Multiplicador de escala de LoRA
    target_modules=["q_proj", "v_proj"],  # Módulos específicos para aplicar LoRA
    lora_dropout=0.1,  # Taxa de dropout
)

# Adicionar LoRA ao modelo
modelo_com_lora = get_peft_model(modelo_tabelas_pretreinado, lora_config)
modelo_com_lora.to(device)

# Passo 3: Definir o DataLoader para carregar a base de dados e a nova classe "table"
def carregar_datasets(diretorio_imagens, diretorio_rotulos, batch_size=16):
    # Transformações nas imagens
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    # Dataset COCO para imagens ou custom dataset para tabelas
    coco_dataset = datasets.CocoDetection(
        root=diretorio_imagens,
        annFile=diretorio_rotulos,
        transform=transform
    )
    
    # Carregador de dados
    return DataLoader(coco_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# Carregar DataLoader com as novas classes
dataloader_tabelas = carregar_datasets(diretorio_imagens_tabelas, diretorio_rotulos_tabelas)

# Passo 4: Definir função de treinamento
def treinar_modelo(model, dataloader, num_epochs=5, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Zerar os gradientes
            optimizer.zero_grad()

            # Forward
            outputs = model(images)

            # Calcular loss
            loss = criterion(outputs, targets)
            loss.backward()

            # Otimizar
            optimizer.step()

            # Acumular loss
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Função de avaliação para calcular precisão, recall e F1-Score
def avaliar_modelo(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calcular métricas
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')

    return precision, recall, f1

# Passo 5: Treinar o modelo pré-ajuste fino e coletar métricas
print("Treinando modelo pré-ajuste fino...")
treinar_modelo(modelo_tabelas_pretreinado, dataloader_tabelas, num_epochs=5)

# Avaliar modelo antes do ajuste fino
precision_pre, recall_pre, f1_pre = avaliar_modelo(modelo_tabelas_pretreinado, dataloader_tabelas)
print(f"Antes do ajuste fino - Precision: {precision_pre:.4f}, Recall: {recall_pre:.4f}, F1-Score: {f1_pre:.4f}")

# Passo 6: Treinar modelo com ajuste fino e coletar métricas
print("Treinando modelo com ajuste fino (LoRA)...")
treinar_modelo(modelo_com_lora, dataloader_tabelas, num_epochs=5)

# Avaliar modelo após o ajuste fino
precision_post, recall_post, f1_post = avaliar_modelo(modelo_com_lora, dataloader_tabelas)
print(f"Após o ajuste fino - Precision: {precision_post:.4f}, Recall: {recall_post:.4f}, F1-Score: {f1_post:.4f}")
