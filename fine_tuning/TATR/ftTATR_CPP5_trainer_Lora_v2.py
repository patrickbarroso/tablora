from compute_metrics import *
from coco_to_datasets import *
import os
import time
import torch
import numpy as np
import logging
from dataclasses import dataclass
from functools import partial
import albumentations as A
from PIL import Image, ExifTags
from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback  
)
from torch.optim import AdamW
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers.image_transforms import center_to_corners_format
import torch.multiprocessing as mp
from torch.amp import autocast
import sys
from datetime import datetime

'''

Referencia: https://huggingface.co/docs/transformers/tasks/object_detection#training-the-detr-model

'''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

print("Dispositivo em uso:", device)
fp16 = True if device == "cuda" else False
use_cpu = False if device == "cuda" else True
dataloader_pin_memory = True if device == "cuda" else False
dataloader_num_workers = 0 if device == "cuda" else 4

# ========================================================
# Configuração do Logger
# ========================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('ftTATR_CPP5.log')
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ========================================================
# Funções Auxiliares
# ========================================================

'''
class CustomTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Assegura que os rótulos estejam presentes para calcular a loss
        labels = inputs.get("labels")
        with torch.no_grad():
            outputs = model(**inputs)
            # outputs deve conter "loss" se os rótulos estiverem presentes
            loss = outputs.get("loss")
        # Retorna a loss para que o Trainer a agregue como "loss"
        return loss, None, None
'''
def training_step(self, model, inputs):
    with autocast():
        outputs = model(**inputs)
    return outputs

class CustomTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.get("labels")
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.get("loss")
            boxes = outputs.get("boxes")  # captura a chave 'boxes'
        # Retorne a loss, os logits (aqui boxes) e os labels
        return loss, boxes, labels
    
def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)
    return {"image_id": image_id, "annotations": annotations}

def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
    global invalid_bbox_count # contador global, se necessário
    invalid_bbox_count = 0
    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))
        category_ids = [obj['category_id'] for obj in objects]
        bboxes = [obj['bbox'] for obj in objects]
        areas = [obj.get('area', 0) for obj in objects]
        valid_bboxes, valid_categories, valid_areas = [], [], []
        for cat_id, bbox, area in zip(category_ids, bboxes, areas):

            x_min, y_min, x_max, y_max = bbox
            if x_max > x_min and y_max > y_min: # FORMATO PASCAL VOC
            #x_min, y_min, width, height = bbox # FORMATO COCO
            #if width > 0 and height > 0: 
                valid_bboxes.append(bbox)
                valid_categories.append(cat_id)
                valid_areas.append(area)
            else:
                invalid_bbox_count += 1
                logger.warning(f"Invalid bbox detected and skipped: {bbox} in image_id {image_id}")
        try:
            output = transform(image=image, bboxes=valid_bboxes, category=valid_categories)
        except Exception as e:
            logger.error(f"Error during transformation for image_id {image_id}: {e}")
            continue
        images.append(output["image"])
        formatted_annotations = format_image_annotations_as_coco(image_id, output["category"], valid_areas, output["bboxes"])
        annotations.append(formatted_annotations)
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    if not return_pixel_mask:
        result.pop("pixel_mask", None)
    return result

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch]).to(device)
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch]).to(device)
    return data


# ========================================================
# MAIN: Configuração do Modelo, Dataset, LoRA e Treinamento
# ========================================================

def main():
    start_time = time.time()

    # Carregue o dataset (exemplo: função load_coco_dataset deve ser definida por você)
    # cppe5 = load_coco_dataset(output_json, IMAGENS_ALL, train_size=535, QTD_VALIDATION=30)
    # Aqui assumimos que cppe5 é um dicionário com as chaves "train", "validation" e "test"
    #cppe5 = load_coco_dataset(output_json, IMAGENS_ALL, train_size=535, QTD_VALIDATION=30)
    #cppe5 = load_coco_dataset(output_json, IMAGENS_ALL, QTD_VALIDATION=20)

    LORA_TARGET_MODULES = [
    'model.encoder.layers.0.self_attn.k_proj',
    'model.encoder.layers.0.self_attn.v_proj',
    'model.encoder.layers.0.self_attn.q_proj',
    'model.encoder.layers.0.self_attn.out_proj',
    'model.encoder.layers.1.self_attn.k_proj',
    'model.encoder.layers.1.self_attn.v_proj',
    'model.encoder.layers.1.self_attn.q_proj',
    'model.encoder.layers.1.self_attn.out_proj',
    'model.encoder.layers.2.self_attn.k_proj',
    'model.encoder.layers.2.self_attn.v_proj',
    'model.encoder.layers.2.self_attn.q_proj',
    'model.encoder.layers.2.self_attn.out_proj',   
    'model.encoder.layers.3.self_attn.k_proj',
    'model.encoder.layers.3.self_attn.v_proj',
    'model.encoder.layers.3.self_attn.q_proj',
    'model.encoder.layers.3.self_attn.out_proj', 
    'model.encoder.layers.4.self_attn.k_proj',
    'model.encoder.layers.4.self_attn.v_proj',
    'model.encoder.layers.4.self_attn.q_proj',
    'model.encoder.layers.4.self_attn.out_proj', 
    'model.encoder.layers.5.self_attn.k_proj',
    'model.encoder.layers.5.self_attn.v_proj',
    'model.encoder.layers.5.self_attn.q_proj',
    'model.encoder.layers.5.self_attn.out_proj',
    'class_labels_classifier',
    'bbox_predictor.layers.0',
    'bbox_predictor.layers.1',
    'bbox_predictor.layers.2',
    ]  

    LORA_R = 8 
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LEARNING_RATE = 1e-3
    EPOCHS = 1000
    TRAIN_BATCH_SIZE = 16
    GRAD_NORM = 0.01

    data_formatada = datetime.now().strftime("%d%m%y")
    pacote = "P2"
    idcenario = f"{data_formatada}{pacote}"
    #desccenario = "FASE 2 - FINETUNING COM LORA NO MODELO TATR ORIGINAL - 50% DATASET - LORA - ENCODER/DECODER"

    desccenario = "FASE 1 - FINETUNING COM LORA EM MODELO TATR CLASSICO DATASET FASE 2  - LORA - ENCODER"
    dir_out_model = f"/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/model/LORA/{idcenario}"
    
    MODEL_NAME = "microsoft/table-transformer-structure-recognition"
    #MODEL_NAME = "/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/model/LORA/110225P1/checkpoint-585"

    print(f" # ========================================================")
    print(f" CENÁRIO {idcenario} - {desccenario}")
    print(f" HIPERPARÂMETROS \n LORA_R: {LORA_R} \n LORA_ALPHA: {LORA_ALPHA} \n LORA_DROPOUT: {LORA_DROPOUT} \n ")
    print(f" LEARNING_RATE: {LEARNING_RATE} \n EPOCHS: {EPOCHS} \n TRAIN_BATCH_SIZE: {TRAIN_BATCH_SIZE} \n ")
    print(f" GRAD_NORM: {LEARNING_RATE} \n GRAD_ACUM_STEPS: {EPOCHS} \n NUM_WORKERS: {TRAIN_BATCH_SIZE} \n ")
    print(f" MODELO TREINAR: {MODEL_NAME} \n ")
    print(f" MODELO A SALVAR EM: {dir_out_model} \n ")
    print(f" LORA_TARGET_MODULES: {LORA_TARGET_MODULES} \n ")

    #cppe5 = load_coco_dataset_full(output_json_lab01, IMAGENS_LAB01)
    cppe5 = load_coco_dataset_full(output_json_fase2, IMAGENS_ALL_FASE2)
    #cppe5 = load_coco_dataset_full(output_json, IMAGENS_ALL)

    tamDataset = len(cppe5["train"]) + len(cppe5["test"]) + len(cppe5["validation"])
    print(f"TAMANHO DATASET: {tamDataset} \n ")
    print(cppe5)
    #sys.exit()

    print(f" # ========================================================")


    # Exemplo de mapeamento de categorias
    #categories_map = [{"name": "cat0"}, {"name": "cat1"}]  # Substitua conforme seu dataset
    #id2label = {index: cat["name"] for index, cat in enumerate(categories_map)}
    #label2id = {v: k for k, v in id2label.items()}
    id2label = {index: category["name"] for index, category in enumerate(categories_map, start=0)}
    label2id = {v: k for k, v in id2label.items()}

    PROCESSOR_NAME = "microsoft/table-transformer-structure-recognition"
    IMAGE_SIZE = 800
    MAX_SIZE = IMAGE_SIZE

    image_processor = AutoImageProcessor.from_pretrained(
        PROCESSOR_NAME,
        do_resize=True,
        size={"max_height": MAX_SIZE, "max_width": MAX_SIZE},
        do_pad=True,
        pad_size={"height": MAX_SIZE, "width": MAX_SIZE},
    )

    # Definição das transformações usando Albumentations
    train_augment_and_transform = A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        #bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category"], clip=True, min_area=25),
    )
    validation_transform = A.Compose(
        [A.NoOp()],
        #bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category"], clip=True),
    )

    train_transform_batch = partial(augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor)
    validation_transform_batch = partial(augment_and_transform_batch, transform=validation_transform, image_processor=image_processor)

    cppe5["train"] = cppe5["train"].with_transform(train_transform_batch)
    cppe5["validation"] = cppe5["validation"].with_transform(validation_transform_batch)
    cppe5["test"] = cppe5["test"].with_transform(validation_transform_batch)

    # Exibe um exemplo do dataset
    print(cppe5["train"][0])
    #sys.exit()

    # Função para computar métricas (opcional)
    #eval_compute_metrics_fn = partial(compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0)

    # Diretório para salvar o modelo
    #dir_out_model = "/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/model/LORA/260225P2"

    # Carrega o modelo (substitua TableTransformerForObjectDetection pela sua classe de modelo)
    from transformers import TableTransformerForObjectDetection  # ajuste conforme necessário
    model = TableTransformerForObjectDetection.from_pretrained(
        MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # ========================================================
    # Configuração do LoRA
    # ========================================================
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )

    print("Antes de aplicar LoRA:", type(model))
    model = get_peft_model(model, lora_config)
    print("Após aplicar LoRA:", type(model))

    print("Parâmetros ajustados pelo LoRA:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # ========================================================
    # Configuração do TrainingArguments com Early Stopping
    # ========================================================
    training_args = TrainingArguments(
        output_dir=dir_out_model,
        num_train_epochs=EPOCHS,
        fp16=True,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        dataloader_num_workers=0,
        dataloader_pin_memory=True, # TRUE apenas para usar com GPU 
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        weight_decay=1e-5,
        max_grad_norm=GRAD_NORM,
        # Usando a loss de validação para monitorar o early stopping
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Menor loss é melhor
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        push_to_hub=False,
        report_to="none",
        use_cpu=use_cpu,
        logging_strategy="epoch",
        logging_dir="/home/aluno-pbarroso/pytorch-pbarroso/FT_TATR_STRUCTURE/logs",
        gradient_accumulation_steps=4,  # Ajuste para economizar memória,
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

     # Cria os callbacks: EarlyStoppingCallback e o seu callback customizado (TrainingMetricsCallback)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5), TrainingMetricsCallback()]

    # ========================================================
    # Criação do Trainer e Início do Treinamento
    # ========================================================

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=cppe5["train"],
        eval_dataset=cppe5["validation"],
        processing_class=image_processor,
        data_collator=collate_fn,
        optimizers=(optimizer, None),
        callbacks=callbacks
    )

    # Imprime os parâmetros treináveis
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    trainer.train()

    # Salva o modelo treinado
    trainer.save_model(dir_out_model)
    print(f"Treinamento concluído e modelo salvo em '{dir_out_model}'")
    
    # Avalia o modelo no conjunto de teste (se existir)
    if "test" in cppe5:
        metrics = trainer.evaluate(eval_dataset=cppe5["test"], metric_key_prefix="test")
        print("Resultados finais de avaliação:", metrics)
    
    duration_time = (time.time() - start_time) / 60
    print(f"\nDuration: {duration_time:.0f} minutes")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True # permite otimizações dinâmicas do cuDNN para sua arquitetura.
    mp.set_start_method("spawn", force=True) # comentar se for usar CPU (aceito pra GPU)
    torch.cuda.synchronize() # comentar se for usar CPU (aceito pra GPU)
    main()
