import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os

# Definir diretórios de imagens e anotações
diretorio_imagens = '/home/aluno-pbarroso/pytorch-pbarroso/ft_tatr/Certificados/img/COCO/ALL_LABS'
diretorio_anotacoes = '/home/aluno-pbarroso/pytorch-pbarroso/ft_tatr/Certificados/img/COCO/ALL_LABS/tabelas_cert.json'

# Verificar se a GPU está disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Transformações nas imagens (aqui aplicamos apenas redimensionamento e conversão para tensor)
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Carregar o dataset COCO customizado
dataset = datasets.CocoDetection(
    root=diretorio_imagens,
    annFile=diretorio_anotacoes,
    transform=transform
)

# Criar o DataLoader para carregar o dataset em batches
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

# Função para plotar uma imagem e suas anotações (opcional, para visualização)
def mostrar_imagem_com_anotacoes(image, target):
    plt.figure(figsize=(8, 8))
    plt.imshow(image.permute(1, 2, 0))  # Permutar as dimensões de [C, H, W] para [H, W, C]
    
    for obj in target:
        bbox = obj['bbox']
        x, y, w, h = bbox
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2))
    
    plt.show()

for batch_idx, (images, targets) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}")
    
    for i in range(len(images)):
        print(f"Imagem {i + 1}:")
        print(f"  - Tamanho da imagem: {images[i].shape}")
        
        print("targets ", targets)

        # Verifique se há anotações para a imagem
        if len(targets[i]) == 0:
            print(f"  - Anotações (targets): Nenhuma anotação encontrada para esta imagem.")
        else:
            print(f"  - Anotações (targets): {targets[i]}")
        
        # Opcional: visualizar a imagem com anotações, apenas se houver anotações
        if len(targets[i]) > 0:
            mostrar_imagem_com_anotacoes(images[i], targets[i])
    
    # Apenas mostrar o primeiro batch
    break

# Aqui você pode continuar com o treinamento ou avaliação usando o DataLoader
