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

import optuna
from transformers import TableTransformerForObjectDetection
from peft import PeftModel, PeftConfig
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Safe alternative available for 'pytorch_model.bin'.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Missing keys.*")

# ========================================================
# Configuração do dispositivo
# ========================================================
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Dispositivo em uso:", device)
fp16 = True if device.type == "cuda" else False
use_cpu = False if device.type == "cuda" else True
dataloader_pin_memory = True if device.type == "cuda" else False
dataloader_num_workers = 0 if device.type == "cuda" else 4

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
    global invalid_bbox_count  # contador global, se necessário
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
            if x_max > x_min and y_max > y_min:  # FORMATO PASCAL VOC
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
    """Ajusta os dados para evitar erro com BatchFeature"""
    pixel_values = torch.stack([x["pixel_values"].clone().detach().requires_grad_(True) for x in batch])
    labels = [x["labels"] for x in batch]
    return {"pixel_values": pixel_values, "labels": labels}

# ========================================================
# Carregamento e Transformação do Dataset
# ========================================================
data_formatada = datetime.now().strftime("%d%m%y")

#cppe5 = load_coco_dataset_full(output_json_fase2, IMAGENS_ALL_FASE2)
cppe5 = load_coco_dataset_full(output_json, IMAGENS_ALL)

print(f"TAMANHO DATASET: {len(cppe5['train']) + len(cppe5['validation']) + len(cppe5['test'])}\n")
print(cppe5)
print("# ========================================================")

# Mapeamento de labels (supondo que 'categories_map' esteja definido no contexto)
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
    use_fast=True
)

# Definição das transformações usando Albumentations
train_augment_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category"], clip=True, min_area=25),
)
validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category"], clip=True),
)

train_transform_batch = partial(augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor)
validation_transform_batch = partial(augment_and_transform_batch, transform=validation_transform, image_processor=image_processor)

cppe5["train"] = cppe5["train"].with_transform(train_transform_batch)
cppe5["validation"] = cppe5["validation"].with_transform(validation_transform_batch)
cppe5["test"] = cppe5["test"].with_transform(validation_transform_batch)

# ========================================================
# Configuração dos Checkpoints e Custom Trainer
# ========================================================
CHECKPOINTS = [
#    !!!!! INPUT CHECKPOINTS PATH HERE !!!
]

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss.float()
        loss.requires_grad_()  # Garante que o tensor pode calcular gradientes

        # Aplicando pesos do DETR: 5 para class e 1 para bbox
        weighted_loss = (self.lambda_class * 5 * loss + self.lambda_bbox * loss) / 6.0

        return (weighted_loss, outputs) if return_outputs else weighted_loss
    def _load_best_model(self):
        try:
            super()._load_best_model()
        except FileNotFoundError:
            logger.warning("Best model checkpoint não encontrado. Ignorando carregamento do melhor modelo.")

# ========================================================
# Otimização com Optuna utilizando k-fold Cross-Validation
# ========================================================
def objective(trial):
    start_time = time.time()
    
    # Seleciona um checkpoint aleatório entre os disponíveis
    model_checkpoint = trial.suggest_categorical("model_checkpoint", CHECKPOINTS)
    print(f"Usando checkpoint: {model_checkpoint}")
    
    # Hiperparâmetros a serem otimizados
    lambda_class = trial.suggest_float("lambda_class", 0.1, 5.0)
    lambda_bbox = trial.suggest_float("lambda_bbox", 0.1, 2.0)
    
    # Configurações de treinamento
    EPOCHS = 1000
    TRAIN_BATCH_SIZE = 16
    GRAD_NORM = 0.01
    GRAD_ACUM_STEPS = 4
    NUM_WORKERS = 0
    LEARNING_RATE = 1e-3

    # Configura o K-Fold (ex.: 5 folds)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(cppe5["train"])):
        print(f"\nTreinando Fold {fold+1}/{kf.get_n_splits()}")
        # Cria os subconjuntos de treino e validação para este fold
        train_subset = torch.utils.data.Subset(cppe5["train"], train_idx)
        val_subset = torch.utils.data.Subset(cppe5["train"], val_idx)
        
        # Reinicializa o modelo para cada fold
        model = TableTransformerForObjectDetection.from_pretrained(model_checkpoint)
        if "070725P2" not in model_checkpoint and "070725P1" not in model_checkpoint and "microsoft" not in model_checkpoint:
            #if not hasattr(model, "peft_config"):    
            model = PeftModel.from_pretrained(model, model_checkpoint)  # inclui LoRA quando necessário
            print("Adaptador LORA carregado")
    
        model = model.to(device)
        
        # Diretório de saída para este fold
        dir_out_model = f"/PATH_TO_CROSS_VALIDATION_PROCESS/trial_{trial.number}_fold{fold}"
        
        training_args = TrainingArguments(
            output_dir=dir_out_model,
            num_train_epochs=EPOCHS,
            fp16=fp16,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            dataloader_num_workers=NUM_WORKERS,
            dataloader_pin_memory=True,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            weight_decay=1e-5,
            max_grad_norm=GRAD_NORM,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=5,
            remove_unused_columns=False,
            eval_do_concat_batches=False,
            push_to_hub=False,
            report_to="none",
            logging_strategy="epoch",
            logging_dir="/PATH_TO_LOGS/logs",
            gradient_accumulation_steps=GRAD_ACUM_STEPS,
            label_names=['labels'],
            save_safetensors=False, # Força a geração do pytorch_model.bin 
            use_cpu=use_cpu
        )
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10), TrainingMetricsCallback()]
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            optimizers=(optimizer, None),
            data_collator=collate_fn,
            callbacks=callbacks
        )
        
        # Define os valores de lambda no trainer
        trainer.lambda_class = lambda_class
        trainer.lambda_bbox = lambda_bbox
        
        trainer.train()
        trainer.save_model(dir_out_model)
        print("Modelo salvo em:", dir_out_model)

        metrics = trainer.evaluate()
        val_loss = metrics["eval_loss"]
        print(f"Fold {fold+1} - Loss de validação: {val_loss}")
        val_losses.append(val_loss)

        # Avalia o modelo no conjunto de teste (se existir)
        if "test" in cppe5:
            metrics = trainer.evaluate(eval_dataset=cppe5["test"], metric_key_prefix="test")
            print("Resultados de teste pós-treinamento:", metrics)
        
        # Libera memória da GPU após cada fold
        del trainer, model, optimizer
        torch.cuda.empty_cache()
    
    avg_val_loss = np.mean(val_losses)
    print(f"\nMédia da loss de validação nos {kf.get_n_splits()} folds: {avg_val_loss}")
    duration = (time.time() - start_time) / 60
    print(f"Duração total: {duration:.2f} minutos")
    return avg_val_loss

# Cria o estudo do Optuna e executa a otimização
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print(f"Melhor modelo: {study.best_params}")
print(f"Melhores valores de lambdas: lambda_class={study.best_params['lambda_class']}, lambda_bbox={study.best_params['lambda_bbox']}")
