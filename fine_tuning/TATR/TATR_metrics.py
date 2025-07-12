def calculate_metrics_TATR(validation_dataset, model, processor, device, threshold_param=0.5):
    """
    Calcula métricas COCO no conjunto de validação.

    Args:
        validation_dataset: Dataset de validação.
        model: Modelo carregado.
        processor: Processador de imagens.
        device: Dispositivo (CPU/GPU).

    Returns:
        dict: Métricas calculadas.
    """
    import numpy as np
    import time
    import json
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from tqdm import tqdm
    import torch

    # Inicializar contadores
    start_time = time.time()
    coco_ground_truth = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    coco_predictions = []

    # Configurar categorias
    categories = [
        {"id": 0, "name": "table"},
        {"id": 1, "name": "table column"},
        {"id": 2, "name": "table row"},
        {"id": 3, "name": "table column header"},
        {"id": 4, "name": "table projected row header"},
        {"id": 5, "name": "table spanning cell"}
    ]
    coco_ground_truth["categories"] = categories

    annotation_id = 1
    for idx, sample in enumerate(tqdm(validation_dataset, desc="Processing validation set")):
        image_id = sample["image_id"]
        image = sample["image"]
        width, height = image.size

        # Adicionar informações da imagem ao ground truth
        coco_ground_truth["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": f"image_{image_id}.jpg"
        })

        # Adicionar anotações ao ground truth
        for obj in sample["objects"]:
            coco_ground_truth["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": obj["category_id"],
                "bbox": obj["bbox"],
                "area": obj["area"],
                "iscrowd": 0
            })
            annotation_id += 1

        # Predições do modelo
        inputs = processor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[height, width]]).to(device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold_param)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            coco_predictions.append({
                "image_id": image_id,
                "category_id": label.item(),
                "bbox": [round(coord, 2) for coord in box.tolist()],
                "score": score.item()
            })

    # Salvar ground truth e predições em formato COCO
    ground_truth_path = "coco_gt.json"
    predictions_path = "coco_preds.json"
    with open(ground_truth_path, "w") as f:
        json.dump(coco_ground_truth, f)
    with open(predictions_path, "w") as f:
        json.dump(coco_predictions, f)

    # Avaliar métricas usando COCOeval
    coco_gt = COCO(ground_truth_path)
    coco_dt = coco_gt.loadRes(predictions_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # Ajuste seguro de iouThrs
    coco_eval.params.iouThrs = np.array([0.5, 0.75])

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Inicializar variáveis para métricas adicionais
    iou_thresholds = [0.5, 0.75]  # Limiares para AP50 e AP75
    ap_results = {}
    recall_results = {}

    # Calcular AP50, AP75 e AR
    for threshold in iou_thresholds:
        coco_eval.params.iouThrs = np.array([threshold])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if threshold == 0.5:
            ap_results["AP50"] = coco_eval.stats[0]  # AP@50%
        elif threshold == 0.75:
            ap_results["AP75"] = coco_eval.stats[0]  # AP@75%

    # Calcular a média de precisão (AP) considerando limiares de IoU de 0.5 a 0.95 em passos de 0.05
    iou_thresholds_full = np.arange(0.5, 1.0, 0.05)
    coco_eval.params.iouThrs = iou_thresholds_full
    coco_eval.evaluate()
    coco_eval.accumulate()
    ap_results["AP"] = np.mean([coco_eval.stats[i] for i in range(len(iou_thresholds_full))])

    # Calcular AR (Average Recall) - Média da taxa de recall em diferentes limiares de IoU
    coco_eval.params.maxDets = [1000]  # Considerando até 100 detecções por imagem
    coco_eval.evaluate()
    coco_eval.accumulate()
    recall_results["AR"] = np.mean([coco_eval.stats[i] for i in range(6, 11)])

    # Inicializar variáveis para métricas adicionais de precisão, recall e f1-score
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_threshold = threshold_param  # Ajuste conforme necessário

    # Processar TP, FP e FN diretamente
    for image_id in coco_gt.getImgIds():
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=image_id))

        for gt_ann in gt_anns:
            matched = False
            for dt_ann in dt_anns:
                iou = calculate_iou(gt_ann["bbox"], dt_ann["bbox"])
                if iou >= iou_threshold:
                    total_tp += 1
                    matched = True
                    break
            if not matched:
                total_fn += 1

        for dt_ann in dt_anns:
            matched = False
            for gt_ann in gt_anns:
                iou = calculate_iou(gt_ann["bbox"], dt_ann["bbox"])
                if iou >= iou_threshold:
                    matched = True
                    break
            if not matched:
                total_fp += 1

    # Calcular métricas adicionais
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Coletar métricas
    metrics = {
        "eval_map": coco_eval.stats[0],  # AP (média)
        "eval_map_50": ap_results.get("AP50", 0),  # AP50
        "eval_map_75": ap_results.get("AP75", 0),  # AP75
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1_score": f1_score,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "eval_ar": recall_results["AR"],  # AR
    }

    # Calcular tempo de execução
    metrics["eval_runtime"] = time.time() - start_time
    metrics["eval_samples_per_second"] = len(validation_dataset) / metrics["eval_runtime"]
    metrics["eval_steps_per_second"] = metrics["eval_samples_per_second"] / len(validation_dataset)

    # Adicionar métricas específicas por classe
    for cat in categories:
        cat_id = cat["id"]
        cat_name = cat["name"]
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics[f"eval_map_{cat_name}"] = coco_eval.stats[0]
        metrics[f"eval_mar_100_{cat_name}"] = coco_eval.stats[8]

    return metrics


def calculate_iou(bbox1, bbox2):
    """Calcula o IoU (Intersection over Union) entre dois bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

"""
# Exemplo de bounding boxes preditos e ground truth
pred_bboxes = [[10, 20, 50, 60], [30, 40, 49, 80]]
gt_bboxes = [[12, 28, 48, 58], [32, 32, 68, 78]]

# Calculando métricas
metrics = calculate_metrics_TATR(pred_bboxes, gt_bboxes)

# Exibindo resultados
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

"""

def calculate_grits_topology(pred_bboxes, gt_bboxes):
    """
    Calcula a métrica GriTS Topology baseada na sobreposição estrutural.

    Args:
        pred_bboxes (list): Lista de bounding boxes preditos [[x1, y1, x2, y2], ...].
        gt_bboxes (list): Lista de bounding boxes ground truth [[x1, y1, x2, y2], ...].

    Returns:
        float: Score da métrica GriTS Topology.
    """
    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    iou_scores = []
    for pred_box in pred_bboxes:
        best_iou = max([iou(pred_box, gt_box) for gt_box in gt_bboxes], default=0)
        iou_scores.append(best_iou)

    return sum(iou_scores) / len(iou_scores) if iou_scores else 0

from difflib import SequenceMatcher

def calculate_grits_content(pred_texts, gt_texts):
    """
    Calcula a métrica GriTS Content baseada na similaridade de texto.

    Args:
        pred_texts (list): Lista de textos preditos.
        gt_texts (list): Lista de textos ground truth.

    Returns:
        float: Score da métrica GriTS Content.
    """
    def text_similarity(text1, text2):
        return SequenceMatcher(None, text1, text2).ratio()

    similarity_scores = []
    for pred_text in pred_texts:
        best_match = max([text_similarity(pred_text, gt_text) for gt_text in gt_texts], default=0)
        similarity_scores.append(best_match)

    return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0


def calculate_grits_localization(pred_bboxes, gt_bboxes):
    """
    Calcula a métrica GriTS Localization baseada na sobreposição de bounding boxes.

    Args:
        pred_bboxes (list): Lista de bounding boxes preditos [[x1, y1, x2, y2], ...].
        gt_bboxes (list): Lista de bounding boxes ground truth [[x1, y1, x2, y2], ...].

    Returns:
        float: Score da métrica GriTS Localization.
    """
    return calculate_grits_topology(pred_bboxes, gt_bboxes)

def calculate_content_accuracy(pred_texts, gt_texts):
    """
    Calcula a métrica Content Accuracy.

    Args:
        pred_texts (list): Lista de textos preditos.
        gt_texts (list): Lista de textos ground truth.

    Returns:
        float: Score da métrica Content Accuracy.
    """
    correct_matches = sum(1 for pred_text in pred_texts if pred_text in gt_texts)
    return correct_matches / len(gt_texts) if gt_texts else 0

# Exemplo de entrada
pred_bboxes = [[10, 20, 50, 60], [30, 40, 70, 80]]
gt_bboxes = [[12, 22, 48, 58], [32, 42, 68, 78]]

pred_texts = ["Total", "2024"]
gt_texts = ["Total", "202aaaa4"]

# Calculando métricas
grits_top = calculate_grits_topology(pred_bboxes, gt_bboxes)
grits_con = calculate_grits_content(pred_texts, gt_texts)
grits_loc = calculate_grits_localization(pred_bboxes, gt_bboxes)
acc_con = calculate_content_accuracy(pred_texts, gt_texts)

# Exibindo resultados
print(f"GriTS Topology: {grits_top:.4f}")
print(f"GriTS Content: {grits_con:.4f}")
print(f"GriTS Localization: {grits_loc:.4f}")
print(f"Content Accuracy: {acc_con:.4f}")
