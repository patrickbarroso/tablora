import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from datasets import Dataset, DatasetDict

# Caminhos dos diretórios
JSON_PATH = "/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/DATASET/"
IMAGENS_ALL = "/home/aluno-pbarroso/pytorch-pbarroso/DATASET/ALL_IMG"
XML_ALL = "/home/aluno-pbarroso/pytorch-pbarroso/DATASET/ALL_XML"
IMAGENS_LAB01 = "/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/dataset/train/images_lab01"
IMAGENS_LAB01_FASE2 = "/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/dataset/train/images_lab01_fase2"
XML_LAB01 = "/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/dataset/train/xml_lab01"
XML_LAB01_FASE2 = "/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/dataset/train/xml_lab01_fase2"

QTD_TEST = 30

JSON_PATH_FASE2 = "/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/dataset/train/"
IMAGENS_ALL_FASE2 = "/home/aluno-pbarroso/pytorch-pbarroso/DATASET/ALL_IMG/"
#XML_ALL_FASE2 = "/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/DATASET/train/xml_all_fase2"

output_json = "/home/aluno-pbarroso/pytorch-pbarroso/DATASET/annotations_coco_datasets_tratado.json"
output_json_fase2 = "/home/aluno-pbarroso/pytorch-pbarroso/DATASET/annotations_coco_datasets_fase2_v2.json"
output_json_lab01 = "/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/dataset/train/annotations_coco_datasets_lab01.json" 
output_json_lab01_fase2 = "/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/dataset/train/annotations_coco_datasets_lab01_fase2.json" 


# Categorias para COCO
categories_map = [
    {"id": 0, "name": "table"},
    {"id": 1, "name": "table column"},
    {"id": 2, "name": "table row"},
    {"id": 3, "name": "table column header"},
    {"id": 4, "name": "table projected row header"},
    {"id": 5, "name": "table spanning cell"},
]

# Função para carregar anotações de um arquivo XML PASCAL VOC
def parse_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_info = {
        "file_name": root.find("filename").text,
        "width": int(root.find("size/width").text),
        "height": int(root.find("size/height").text),
    }

    annotations = []
    invalid_bboxes_count = 0  # Contador de bounding boxes inválidos

    for obj in root.findall("object"):
        category_name = obj.find("name").text
        category_id = next((category["id"] for category in categories_map if category["name"] == category_name), None)
        
        if category_id is not None:
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Verificar se o bounding box é válido
            if xmax > xmin and ymax > ymin:            
                # Calcular a área
                area = (xmax - xmin) * (ymax - ymin)

                annotation = {
                    "category_id": category_id,
                    #"bbox": [xmin, ymin, xmax - xmin, ymax - ymin],  # COCO bbox format
                    "bbox": [xmin, ymin, xmax, ymax], #PASCAL VOC FORMAT
                    "area": area,
                    "iscrowd": 0,
                }
                annotations.append(annotation)
            else:
                invalid_bboxes_count += 1  # Incrementar contador de bbox inválidos

    # Exibir a quantidade de bounding boxes inválidos para o arquivo atual
    if invalid_bboxes_count > 0:
        print(f"Arquivo {xml_path} contém {invalid_bboxes_count} bounding boxes inválidos.")
            
    return image_info, annotations

# Função para criar o arquivo COCO JSON
def create_coco_json(image_dir, xml_dir, categories):
    coco_json = {
        "info": {"description": "Custom COCO dataset", "version": "1.0", "year": 2025},
        "licenses": [{"id": 1, "name": "CC0", "url": "http://creativecommons.org/licenses/by/4.0/"}],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    image_id = 0
    annotation_id = 0
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_dir, xml_file)
            image_info, annotations = parse_voc_annotation(xml_path)

            # Adicionar imagem
            image_info["id"] = image_id
            coco_json["images"].append(image_info)

            # Adicionar anotações
            for annotation in annotations:
                annotation["image_id"] = image_id
                annotation["id"] = annotation_id
                coco_json["annotations"].append(annotation)
                annotation_id += 1

            image_id += 1

    return coco_json

# Carregar o dataset COCO no formato DatasetDict
def load_coco_dataset_OLD(output_json, image_dir):
    with open(output_json, "r") as f:
        coco_data = json.load(f)

    # Preparar as listas para o formato DatasetDict
    images = coco_data['images']
    annotations = coco_data['annotations']

    image_data = []
    object_data = []

    for img in images:
        image_id = img['id']
        image_path = os.path.join(image_dir, img['file_name'])

        # Carregar a imagem
        image = Image.open(image_path)

        # Obter anotações para essa imagem
        objects = []
        for annotation in annotations:
            if annotation['image_id'] == image_id:
                objects.append({
                    "category_id": annotation['category_id'],
                    "bbox": annotation['bbox'],
                    "area": annotation['area']
                })

        image_data.append({
            'image_id': image_id,
            'image': image,
            'width': img['width'],
            'height': img['height'],
            'objects': objects,
        })

    # Criar o dataset
    dataset = Dataset.from_dict({
        'image_id': [entry['image_id'] for entry in image_data],
        'image': [entry['image'] for entry in image_data],
        'width': [entry['width'] for entry in image_data],
        'height': [entry['height'] for entry in image_data],
        'objects': [entry['objects'] for entry in image_data],
    })

    '''
    # Dividir em train, validation e test
    split = dataset.train_test_split(test_size=0.30, seed=1337)
    train_dataset = split['train']
    validation_dataset = split['test']

    # Criar DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
    })

    '''

    # Embaralhar o dataset para garantir aleatoriedade
    shuffled_dataset = dataset.shuffle(seed=1337)

    # Selecionar 30 imagens aleatórias para o conjunto de teste
    test_dataset = shuffled_dataset.select(range(QTD_TEST))  # Seleciona 30 imagens aleatórias
    remaining_dataset = shuffled_dataset.select(range(QTD_TEST, len(dataset)))  # O restante do dataset

    # Dividir o restante em treino e validação
    split = remaining_dataset.train_test_split(test_size=0.30, seed=1337)
    train_dataset = split['train']
    validation_dataset = split['test']

    # Criar DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })

    return dataset_dict

import json
import os
from PIL import Image
from datasets import Dataset, DatasetDict

import json
import os
from PIL import Image
from datasets import Dataset, DatasetDict

import json
import os
from PIL import Image
from datasets import Dataset, DatasetDict


import os
import json
from PIL import Image
from datasets import Dataset, DatasetDict

def load_coco_dataset_full(input_json, image_dir, seed=1337):
    # Carrega o JSON original em formato COCO
    with open(input_json, "r") as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']

    image_data = []
    for img in images:
        image_id = img['id']
        image_path = os.path.join(image_dir, img['file_name'])

        # Abre a imagem
        image = Image.open(image_path)

        # Coleta as anotações correspondentes a esta imagem
        objects = []
        for ann in annotations:
            if ann['image_id'] == image_id:
                objects.append({
                    "category_id": ann['category_id'],
                    "bbox": ann['bbox'],
                    "area": ann['area']
                })

        image_data.append({
            'image_id': image_id,
            'image': image,
            'width': img['width'],
            'height': img['height'],
            'objects': objects,
        })

    # Cria um dataset do HuggingFace a partir dos dados carregados
    dataset = Dataset.from_dict({
        'image_id': [entry['image_id'] for entry in image_data],
        'image': [entry['image'] for entry in image_data],
        'width': [entry['width'] for entry in image_data],
        'height': [entry['height'] for entry in image_data],
        'objects': [entry['objects'] for entry in image_data],
    })

    # Embaralha o dataset para garantir aleatoriedade
    shuffled_dataset = dataset.shuffle(seed=seed)
    total = len(shuffled_dataset)

    # Define os tamanhos de cada split
    train_count = int(total * 0.7)
    val_count = int(total * 0.15)
    # O teste receberá os restantes (aproximadamente 15%)
    test_count = total - train_count - val_count

    # Seleciona os subsets conforme os índices
    train_dataset = shuffled_dataset.select(range(train_count))
    validation_dataset = shuffled_dataset.select(range(train_count, train_count + val_count))
    test_dataset = shuffled_dataset.select(range(train_count + val_count, total))

    # Cria um DatasetDict com os três splits
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })

    # Prepara os dados de teste para salvar em formato COCO
    # (Filtra as imagens e anotações originais que estão no split de teste)
    test_image_ids = set(test_dataset['image_id'])
    test_images = [img for img in coco_data['images'] if img['id'] in test_image_ids]
    test_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in test_image_ids]
    test_categories = coco_data.get('categories', [])

    test_coco = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": test_categories
    }

    # Salva os dados de teste no arquivo "output_validation.json"
    #with open(output_test, "w") as f:
    #    json.dump(test_coco, f, indent=4)

    return dataset_dict


def load_coco_dataset(output_json, image_dir, train_size=None, QTD_VALIDATION=100, seed=1337):
    with open(output_json, "r") as f:
        coco_data = json.load(f)

    # Preparar as listas para o formato DatasetDict
    images = coco_data['images']
    annotations = coco_data['annotations']

    image_data = []

    for img in images:
        image_id = img['id']
        image_path = os.path.join(image_dir, img['file_name'])

        # Carregar a imagem
        image = Image.open(image_path)

        # Obter anotações para essa imagem
        objects = []
        for annotation in annotations:
            if annotation['image_id'] == image_id:
                objects.append({
                    "category_id": annotation['category_id'],
                    "bbox": annotation['bbox'],
                    "area": annotation['area']
                })

        image_data.append({
            'image_id': image_id,
            'image': image,
            'width': img['width'],
            'height': img['height'],
            'objects': objects,
        })

    # Criar o dataset
    dataset = Dataset.from_dict({
        'image_id': [entry['image_id'] for entry in image_data],
        'image': [entry['image'] for entry in image_data],
        'width': [entry['width'] for entry in image_data],
        'height': [entry['height'] for entry in image_data],
        'objects': [entry['objects'] for entry in image_data],
    })

    # Embaralhar o dataset para garantir aleatoriedade
    shuffled_dataset = dataset.shuffle(seed=seed)

    # Reservar QTD_VALIDATION imagens para validação
    QTD_VALIDATION = min(QTD_VALIDATION, len(shuffled_dataset))  # Garantir que não exceda o tamanho do dataset
    validation_dataset = shuffled_dataset.select(range(QTD_VALIDATION))  # Seleciona as primeiras `QTD_VALIDATION` imagens
    remaining_dataset = shuffled_dataset.select(range(QTD_VALIDATION, len(shuffled_dataset)))  # O restante do dataset

    # Determinar o número de imagens para o conjunto de treino
    if train_size is not None:
        train_size = min(train_size, len(remaining_dataset))  # Garantir que não exceda o tamanho do dataset restante
        train_dataset = remaining_dataset.select(range(train_size))  # Seleciona as primeiras `train_size` imagens
        test_size = int(0.3 * train_size)  # Teste será 30% do treino
        test_dataset = train_dataset.select(range(train_size - test_size, train_size))  # Seleciona os últimos 30% para teste
        train_dataset = train_dataset.select(range(train_size - test_size))  # Atualiza o treino para os 70% restantes
    else:
        # Usar todas as imagens restantes para treino e teste
        test_size = int(0.3 * len(remaining_dataset))  # Teste será 30% do restante
        test_dataset = remaining_dataset.select(range(len(remaining_dataset) - test_size, len(remaining_dataset)))  # Seleciona os últimos 30% para teste
        train_dataset = remaining_dataset.select(range(len(remaining_dataset) - test_size))  # Atualiza o treino para os 70% restantes

    # Criar DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })

    return dataset_dict



import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_valid_bbox(bbox):
    """
    Verifica se uma bounding box é válida.
    Retorna True se a bounding box for válida, False caso contrário.
    """
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    return x_max > x_min and y_max > y_min

def validate_and_filter_objects(objects):
    """
    Valida e filtra as bounding boxes inválidas de uma lista de objetos.
    Retorna uma lista contendo apenas objetos com bounding boxes válidas.
    """
    valid_objects = []
    for obj in objects:
        if is_valid_bbox(obj['bbox']):
            valid_objects.append(obj)
        else:
            print(f"Invalid bbox found and removed: {obj['bbox']}")
            logger.warning(f"Invalid bbox found and removed: {obj['bbox']}")
    return valid_objects

def load_coco_dataset2(output_json, image_dir, train_size=None, QTD_VALIDATION=100, seed=1337):
    """
    Carrega um dataset no formato COCO e valida as bounding boxes.
    """
    with open(output_json, "r") as f:
        coco_data = json.load(f)

    # Preparar as listas para o formato DatasetDict
    images = coco_data['images']
    annotations = coco_data['annotations']

    image_data = []

    for img in images:
        image_id = img['id']
        image_path = os.path.join(image_dir, img['file_name'])

        # Carregar a imagem
        image = Image.open(image_path)

        # Obter anotações para essa imagem
        objects = []
        for annotation in annotations:
            if annotation['image_id'] == image_id:
                objects.append({
                    "category_id": annotation['category_id'],
                    "bbox": annotation['bbox'],
                    "area": annotation['area']
                })

        # Validar e filtrar bounding boxes inválidas
        objects = validate_and_filter_objects(objects)

        image_data.append({
            'image_id': image_id,
            'image': image,
            'width': img['width'],
            'height': img['height'],
            'objects': objects,
        })

    # Criar o dataset
    dataset = Dataset.from_dict({
        'image_id': [entry['image_id'] for entry in image_data],
        'image': [entry['image'] for entry in image_data],
        'width': [entry['width'] for entry in image_data],
        'height': [entry['height'] for entry in image_data],
        'objects': [entry['objects'] for entry in image_data],
    })

    # Embaralhar o dataset para garantir aleatoriedade
    shuffled_dataset = dataset.shuffle(seed=seed)

    # Reservar QTD_VALIDATION imagens para validação
    QTD_VALIDATION = min(QTD_VALIDATION, len(shuffled_dataset))  # Garantir que não exceda o tamanho do dataset
    validation_dataset = shuffled_dataset.select(range(QTD_VALIDATION))  # Seleciona as primeiras `QTD_VALIDATION` imagens
    remaining_dataset = shuffled_dataset.select(range(QTD_VALIDATION, len(shuffled_dataset)))  # O restante do dataset

    # Determinar o número de imagens para o conjunto de treino
    if train_size is not None:
        train_size = min(train_size, len(remaining_dataset))  # Garantir que não exceda o tamanho do dataset restante
        train_dataset = remaining_dataset.select(range(train_size))  # Seleciona as primeiras `train_size` imagens
        test_size = int(0.3 * train_size)  # Teste será 30% do treino
        test_dataset = train_dataset.select(range(train_size - test_size, train_size))  # Seleciona os últimos 30% para teste
        train_dataset = train_dataset.select(range(train_size - test_size))  # Atualiza o treino para os 70% restantes
    else:
        # Usar todas as imagens restantes para treino e teste
        test_size = int(0.3 * len(remaining_dataset))  # Teste será 30% do restante
        test_dataset = remaining_dataset.select(range(len(remaining_dataset) - test_size, len(remaining_dataset)))  # Seleciona os últimos 30% para teste
        train_dataset = remaining_dataset.select(range(len(remaining_dataset) - test_size))  # Atualiza o treino para os 70% restantes

    # Criar DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })

    return dataset_dict


def indent_coco(arquivo_entrada, arquivo_saida):

    # Ler o arquivo COCO e identar
    with open(arquivo_entrada, "r", encoding="utf-8") as f:
        dados_coco = json.load(f)

    # Salvar o arquivo identado
    with open(arquivo_saida, "w", encoding="utf-8") as f:
        json.dump(dados_coco, f, indent=4, ensure_ascii=False)

    print(f"Arquivo identado salvo como {arquivo_saida}")

'''
# Criar o JSON COCO
coco_json = create_coco_json(IMAGENS_ALL, XML_ALL, categories_map)
coco_json = create_coco_json(IMAGENS_LAB01, XML_LAB01, categories_map)

# Salvar o arquivo JSON
with open(output_json_lab1, "w") as f:
    json.dump(coco_json, f)

#print(f"COCO JSON criado em {output_json}")

# Carregar o dataset
cppe5 = load_coco_dataset(output_json, IMAGENS_ALL, train_size=535, QTD_VALIDATION=30)
cppe5 = load_coco_dataset2(output_json_lab1, IMAGENS_LAB01, train_size=70, QTD_VALIDATION=20)


#novo formato!!!!! (train = 70% , test = 15%, val = 15%)
cppe5 = load_coco_dataset_full(output_json, IMAGENS_ALL)

# Visualizar o dataset
print(cppe5)
print(cppe5["train"][0]["image_id"])
print(cppe5["validation"][0]["image_id"])
print(cppe5["test"][0]["image_id"])


#PROCESSO DE CRIACAO DE ARQUIVO JSON COCO PARA UM DETERMINADO LABORATORIO
# Criar o JSON COCO
coco_json = create_coco_json(IMAGENS_LAB01_FASE2, XML_LAB01_FASE2, categories_map)

# Salvar o arquivo JSON
with open(output_json_lab01_fase2, "w") as f:
    json.dump(coco_json, f)

print(f"COCO JSON criado em {output_json}")

#CARREGANDO ARQUIVO COCO
cppe5 = load_coco_dataset_full(output_json_lab01_fase2, IMAGENS_LAB01_FASE2)
print(cppe5)

#CRIAR ARQUIVO IDENTADO
output_json_lab01_fase2_indent = "/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/dataset/train/annotations_coco_datasets_lab01_fase2_ident.json"
indent_coco(output_json_lab01_fase2, output_json_lab01_fase2_indent)


'''


