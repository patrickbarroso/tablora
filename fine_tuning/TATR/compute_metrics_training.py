from ftTATR_functions import convert_bbox_yolo_to_pascal
from HungarianMatcher import *
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import generalized_box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from scipy.optimize import linear_sum_assignment  # para o matching húngaro
from dataclasses import dataclass
import numpy as np
import evaluate
import logging 
from sklearn.metrics import accuracy_score, precision_score

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_bbox(bbox):
    # Converte [x, y, w, h] em coordenadas normalizadas e retorna [x_min, y_min, x_max, y_max]
    x, y, w, h = bbox
    x_min = x / w
    y_min = y / h
    x_max = (x + w) / w
    y_max = (y + h) / h

    return [x_min, y_min, x_max, y_max]

def convert_bbox_xyxy_to_xywh(bbox):
    """
    Converte um bounding box do formato [x_min, y_min, x_max, y_max]
    para o formato [x, y, w, h].

    Args:
        bbox (list or tuple): Bounding box no formato [x_min, y_min, x_max, y_max].

    Returns:
        list: Bounding box no formato [x, y, w, h], onde:
            - x = x_min
            - y = y_min
            - w = x_max - x_min
            - h = y_max - y_min
    """
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min

    return [x_min, y_min, w, h]

def compute_detr_losses_for_image(
    dt_anns, 
    gt_anns, 
    cost_class_weight=1.0, 
    cost_bbox_weight=5.0, 
    cost_giou_weight=2.0
):
    """
    Calcula as perdas de classificação, L1 e GIoU (estilo DETR) para uma única imagem,
    resolvendo a correspondência dt_anns -> gt_anns via Hungarian.

    Args:
        dt_anns (list): Lista de anotações de detecções (COCO), cada item é um dicionário com:
                        {"category_id": int, "score": float, "bbox": [x, y, w, h], ...}
        gt_anns (list): Lista de anotações de ground truth (COCO), cada item é um dicionário com:
                        {"category_id": int, "bbox": [x, y, w, h], ...}
        cost_class_weight (float): Peso para o custo de classificação.
        cost_bbox_weight (float): Peso para o custo L1 de bounding box.
        cost_giou_weight (float): Peso para o custo GIoU.

    Returns:
        (total_cls_loss, total_bbox_loss, total_giou_loss, matched_pairs_count)
        - total_cls_loss (float): Soma das perdas de classificação
        - total_bbox_loss (float): Soma das perdas L1 das caixas
        - total_giou_loss (float): Soma das perdas GIoU
        - matched_pairs_count (int): Número de pares correspondidos
    """
    num_dt = len(dt_anns)
    num_gt = len(gt_anns)

    # Se não há detecções e não há ground truth, nenhuma perda a somar
    if num_dt == 0 and num_gt == 0:
        return 0.0, 0.0, 0.0, 0

    # Matriz de custos [num_dt, num_gt]
    cost_matrix = np.zeros((num_dt, num_gt), dtype=np.float32)

    for i, dt in enumerate(dt_anns):
        dt_cat = dt["category_id"]
        dt_score = max(min(dt["score"], 1 - 1e-6), 1e-6)  # clamp para evitar log(0)
        #dt_box_xyxy = coco_box_to_xyxy(dt["bbox"])     
        #dt_box_xyxy = dt["bbox"]
        dt_box_xyxy = convert_bbox_xyxy_to_xywh(dt["bbox"])
        dt_box_xyxy = normalize_bbox (dt_box_xyxy)

        for j, gt in enumerate(gt_anns):
            gt_cat = gt["category_id"]
            
            #gt_box_xyxy = coco_box_to_xyxy(gt["bbox"])
            #gt_box_xyxy = gt["bbox"]
            gt_box_xyxy = convert_bbox_xyxy_to_xywh(gt["bbox"])
            gt_box_xyxy = normalize_bbox (gt_box_xyxy)

            # --- Custo de Classificação ---
            if dt_cat == gt_cat:
                # Quanto maior dt_score, menor o custo
                cost_class = -np.log(dt_score)
            else:
                # Penaliza se a classe estiver errada
                cost_class = -np.log(1.0 - dt_score)

            # --- Custo L1 (x, y, w, h) ---
            dx1, dy1, dx2, dy2 = dt_box_xyxy
            gx1, gy1, gx2, gy2 = gt_box_xyxy
            dw = dx2 - dx1
            dh = dy2 - dy1
            gw = gx2 - gx1
            gh = gy2 - gy1
            cost_l1 = abs(dx1 - gx1) + abs(dy1 - gy1) + abs(dw - gw) + abs(dh - gh)

            # --- Custo GIoU (1 - GIoU) ---
            giou_val = generalized_box_iou_xyxy(dt_box_xyxy, gt_box_xyxy)
            cost_giou = 1.0 - giou_val

            # Combinar os custos com pesos (estilo DETR)
            cost = (cost_class_weight * cost_class 
                    + cost_bbox_weight * cost_l1 
                    + cost_giou_weight * cost_giou)
            cost_matrix[i, j] = cost

    # Resolver o problema de atribuição via Hungarian
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_giou_loss = 0.0
    matched_pairs_count = 0

    # Calcular perdas para cada par correspondido
    for r, c in zip(row_ind, col_ind):
        dt = dt_anns[r]
        gt = gt_anns[c]
        dt_cat = dt["category_id"]
        dt_score = max(min(dt["score"], 1 - 1e-6), 1e-6)
        #dt_box_xyxy = coco_box_to_xyxy(dt["bbox"])
        #dt_box_xyxy = dt["bbox"]
        dt_box_xyxy = convert_bbox_xyxy_to_xywh(dt["bbox"])
        dt_box_xyxy = normalize_bbox (dt_box_xyxy)

        gt_cat = gt["category_id"]
        #gt_box_xyxy = coco_box_to_xyxy(gt["bbox"])
        #gt_box_xyxy = gt["bbox"]
        gt_box_xyxy = convert_bbox_xyxy_to_xywh(gt["bbox"])
        gt_box_xyxy = normalize_bbox (gt_box_xyxy)

        # Classification
        if dt_cat == gt_cat:
            cls_loss = -np.log(dt_score)
        else:
            cls_loss = -np.log(1.0 - dt_score)

        # L1
        dx1, dy1, dx2, dy2 = dt_box_xyxy
        gx1, gy1, gx2, gy2 = gt_box_xyxy
        dw = dx2 - dx1
        dh = dy2 - dy1
        gw = gx2 - gx1
        gh = gy2 - gy1
        l1_loss = abs(dx1 - gx1) + abs(dy1 - gy1) + abs(dw - gw) + abs(dh - gh)

        # GIoU
        giou_val = generalized_box_iou_xyxy(dt_box_xyxy, gt_box_xyxy)
        giou_loss = 1.0 - giou_val

        total_cls_loss += cls_loss
        total_bbox_loss += l1_loss
        total_giou_loss += giou_loss
        matched_pairs_count += 1

    return total_cls_loss, total_bbox_loss, total_giou_loss, matched_pairs_count

# --- Função para computar os componentes da loss DETR ---
def compute_detr_losses(pred_logits, pred_boxes, target_label, target_box):
    """
    Calcula os três componentes da loss para um par predição-target.
    Os tensores pred_logits e target_label devem ser de dimensão [1, num_classes] e [1] respectivamente,
    e pred_boxes e target_box devem ser de dimensão [1, 4].
    """
    # Loss de classificação
    cls_loss = F.cross_entropy(pred_logits, target_label)

    # L1 loss para a caixa
    bbox_loss = F.l1_loss(pred_boxes, target_box)

    # Loss de Generalized IoU
    giou = generalized_box_iou(pred_boxes, target_box)  # retorna tensor de shape [1, 1]
    giou_loss = 1 - giou.squeeze()

    return cls_loss, bbox_loss, giou_loss

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor

@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.5, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids
    #print("Predictions:", evaluation_results.predictions)
    #print("Labels:", evaluation_results.label_ids)

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"], device="cpu") # Ensure CPU context
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"], device="cpu") # Ensure CPU contex
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits, device="cpu" ), pred_boxes=torch.tensor(batch_boxes, device="cpu" )
                            )
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()


    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


def compute_metrics_old(evaluation_results, image_processor, threshold=0.5, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            #print(f"orig_size: {image_target['orig_size']}, type: {type(image_target['orig_size'])}")


            orig_size = image_target.get("orig_size", None)
            if not isinstance(orig_size, (list, tuple)) or len(orig_size) != 2:
                print(f"[WARNING] Ignoring invalid orig_size: {orig_size}, type: {type(orig_size)}")
                continue  # Ignorar essa imagem e passar para a próxima  
            
            try:
                boxes = torch.tensor(image_target["boxes"])
                boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
                labels = torch.tensor(image_target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
                
                image_sizes.append(torch.tensor(orig_size))
            
            except Exception as e:
                print(f"[ERROR] Failed processing image_target: {image_target}. Error: {e}")
                continue  # Se der erro, ignora essa entrada 

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    #print(f"Número de predições: {len(post_processed_predictions)}")
    #print(f"Número de targets: {len(post_processed_targets)}")
    #print(f"Exemplo de post_processed_predictions[0]: {post_processed_predictions[0]}")
    #print(f"Exemplo de post_processed_targets[0]: {post_processed_targets[0]}")

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics

from transformers import TrainerCallback
import numpy as np

class TrainingMetricsCallback(TrainerCallback):
    """
    Callback personalizado para calcular e logar métricas de treinamento.
    """

    """
    Callback para logar métricas durante o treinamento.
    """
    def on_log(self, args, state, control, **kwargs):
        # Obtém o Trainer e checa os valores de loss
        trainer = kwargs.get('trainer', None)
        logs = kwargs.get('logs', {})
        
        if trainer and 'loss' in logs:
            current_loss = logs['loss']
            logger.info(f"Training loss: {current_loss:.4f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Calcula e loga as métricas de treinamento no final de cada época.
        """
        # Acessa o objeto Trainer
        trainer = kwargs.get('trainer')
        if trainer is None:
            return

        # Obtém as previsões e rótulos do treinamento
        train_preds = trainer.predict(trainer.train_dataset)
        train_labels = train_preds.label_ids
        train_preds = train_preds.predictions.argmax(-1)

        # Realiza previsões no conjunto de treinamento
        train_preds = trainer.predict(trainer.train_dataset)
        #logger.info(f"Type of predictions: {type(train_preds.predictions)}")
        #logger.info(f"Predictions shape: {np.array(train_preds.predictions).shape}")
        if hasattr(train_preds, 'predictions') and hasattr(train_preds, 'label_ids'):
            metrics = compute_train_metrics(train_preds)
            logger.info(f"Epoch {state.epoch} - Metrics: {metrics}")
        else:
            logger.warning(f"Predictions or labels missing at epoch {state.epoch}.")

        # Calcula as métricas de treinamento
        train_loss = train_preds.loss.item()  # Perda de treinamento
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_precision = precision_score(train_labels, train_preds, average='macro')

        # Loga as métricas de treinamento
        print(f"Epoch {state.epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Train Precision: {train_precision:.4f}")

        # Adiciona as métricas ao estado do Trainer (opcional)
        state.log_history.append({
            'epoch': state.epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
        })

def compute_train_metrics(pred):
    """
    Calcula as métricas de acurácia, precisão e perda.
    """

    # Verificar o tipo e tamanho das previsões
    logger.info(f"Type of predictions: {type(pred.predictions)}")
    #logger.info(f"Predictions example: {pred.predictions[:2]}")
    logger.info(f"Predictions length: {len(pred.predictions)}")

    # Padronizar previsões se forem listas de diferentes tamanhos
    try:
        max_length = max(len(p) for p in pred.predictions)
        predictions = np.array([np.pad(p, (0, max_length - len(p)), constant_values=0) for p in pred.predictions])
    except Exception as e:
        logger.error(f"Error processing predictions: {e}")
        return {}

    labels = pred.label_ids

    # Garantir que labels e preds sejam compatíveis
    if labels is None or predictions is None:
        logger.warning("Labels ou previsões ausentes durante o cálculo de métricas.")
        return {}
    
    if hasattr(pred.predictions, "shape"):
        logger.info(f"Predictions shape: {pred.predictions.shape}")

    if isinstance(pred.predictions, list):
        predictions = np.array(pred.predictions)
    else:
        predictions = pred.predictions
    
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    loss = pred.loss

    logger.info(f"Predictions received: {pred.predictions.shape if hasattr(pred, 'predictions') else 'None'}")
    logger.info(f"Labels received: {pred.label_ids.shape if hasattr(pred, 'label_ids') else 'None'}")

    # Verifica validade de rótulos e previsões
    if labels is None or preds is None:
        logger.warning("Labels ou previsões ausentes durante o cálculo de métricas.")
        return {}

    # Calcula as métricas
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')

    return {
        'train_loss': loss.item(),
        'train_accuracy': accuracy,
        'train_precision': precision,
    }

