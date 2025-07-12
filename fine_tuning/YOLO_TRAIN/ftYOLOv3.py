import torch
import json
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralyticsplus import YOLO
from sklearn.model_selection import train_test_split
import glob
import shutil
import os
#pip install pycocotools
import sys

def remover_arquivos(pasta):
    # Lista todos os arquivos no diretório
    for arquivo in os.listdir(pasta):
        # Cria o caminho completo do arquivo
        caminho_arquivo = os.path.join(pasta, arquivo)
        
        # Verifica se é um arquivo (ignora diretórios)
        if os.path.isfile(caminho_arquivo):
            #print(f"Removendo: {caminho_arquivo}")
            os.remove(caminho_arquivo)  # Remove o arquivo

# Diretórios do novo dataset de tabelas
arquivo_yolo_yaml = '/ROOT_PATH/yolo.yaml'
diretorio_imagens_tabelas = '/ROOT_PATH/DATASET/ALL_LABS'
diretorio_labels = '/ROOT_PATH/DATASET/ALL_YOLO'
diretorio_train_img = '/ROOT_PATH/DATASET/train/images'
diretorio_train_labels = '/ROOT_PATH/DATASET/train/labels'
diretorio_val_img = '/ROOT_PATH/DATASET/val/images'
diretorio_val_labels = '/ROOT_PATH/DATASET/val/labels'

# Coletando todas as imagens
images = glob.glob(diretorio_imagens_tabelas + '/*.jpg')

# Dividindo em treino e validação
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

#remover as pastas de treinamento e validacao (caso existam)
print("Removendo arquivos de imagem...")
remover_arquivos(diretorio_train_img)
remover_arquivos(diretorio_val_img)
print("Removendo arquivos de labels...")
remover_arquivos(diretorio_train_labels)
remover_arquivos(diretorio_val_labels)

print("Copiando imagens de treino e validação...")
# Copiando as imagens de treino e validação para suas respectivas pastas
for img in train_images:
    shutil.copy(img, f"{diretorio_train_img}/{os.path.basename(img)}")

for img in val_images:
    shutil.copy(img, f"{diretorio_val_img}/{os.path.basename(img)}")

# Obtendo a lista dos arquivos na pasta de treinamento e validaçao
train_images = glob.glob(f'{diretorio_train_img}/*.jpg')
val_images = glob.glob(f'{diretorio_val_img}/*.jpg')
#print("train_images", train_images)

print("Copiando os labels de treino e validacao...")

# Copiando as anotações (labels) para as pastas correspondentes
for label in glob.glob(f'{diretorio_labels}/*.txt'):
    img_name = os.path.basename(label).replace('.txt', '.jpg')

    if any(img_name in item for item in train_images):
        shutil.copy(label, f"{diretorio_train_labels}/{os.path.basename(label)}")
    else:
        shutil.copy(label, f"{diretorio_val_labels}/{os.path.basename(label)}")

# Verificar dispositivo (GPU ou CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Passo 1: Carregar o modelo YOLO pré-treinado específico para detecção de tabelas
try:
    modelo_tabelas_pretreinado = YOLO('foduucom/table-detection-and-extraction').to(device)
    print("modelo YOLO pre-treinado carregado!")

except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

# Passo 2: Definir o DataLoader para carregar o novo dataset
def carregar_datasets(diretorio_imagens, diretorio_rotulos, batch_size=16):
    # Transformações nas imagens
    transform = transforms.Compose([

        transforms.Resize((640, 640)),  # Ajustar a dimensão das imagens
        transforms.ToTensor(),         # Converter imagens em tensores
    ])

    # Carregar o dataset COCO (ou custom) de imagens e anotações de tabelas
    coco_dataset = datasets.CocoDetection(
        root=diretorio_imagens,
        annFile=diretorio_rotulos,
        transform=transform
    )
    
    # DataLoader para carregar as imagens e anotações
    return DataLoader(coco_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# Passo 3: Função para treinamento
def treinar_modelo(model, dataloader, num_epochs=10, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Otimizador
    criterion = torch.nn.CrossEntropyLoss()  # Função de perda (loss)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()  # Zerar os gradientes

            # Forward pass (predições do modelo)
            outputs = model(images)

            # Cálculo da perda (loss) com as predições e os rótulos
            loss = criterion(outputs, targets)
            loss.backward()  # Backpropagation  
            optimizer.step()  # Atualizar os pesos

            running_loss += loss.item()

        # Exibir o progresso
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Passo 4: Função de avaliação (opcional para coletar métricas)
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



# Função para exibir o conteúdo do DataLoader
def exibir_dataloader(dataloader, num_batches=3):
    # Lista para armazenar os dados
    data_list = []
    
    # Iterar sobre o DataLoader
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Armazenar imagens e targets em um dicionário
        batch_data = {
            'images': images,
            'targets': targets
        }
        data_list.append(batch_data)
        
        # Exibir o conteúdo do batch
        print(f"Batch {batch_idx + 1}:")
        print(f"Imagens: {images.shape}")  # Forma das imagens
        print(f"Rótulos (targets): {targets}")  # Exibir rótulos
        print()

        # Parar após exibir o número desejado de batches
        if batch_idx + 1 >= num_batches:
            break

    return data_list

modelo_tabelas_pretreinado.train(data=arquivo_yolo_yaml, epochs=50, imgsz=640)
