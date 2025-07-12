import numpy as np
from scipy.optimize import linear_sum_assignment

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

def normalize_bbox(bbox):
    # Converte [x, y, w, h] em coordenadas normalizadas e retorna [x_min, y_min, x_max, y_max]
    x, y, w, h = bbox
    x_min = x / w
    y_min = y / h
    x_max = (x + w) / w
    y_max = (y + h) / h

    return [x_min, y_min, x_max, y_max]

def coco_box_to_xyxy(box):
    """Converte [x, y, w, h] -> [x_min, y_min, x_max, y_max]."""
    x, y, w, h = box
    return [x, y, x + w, y + h]

def box_area_xyxy(box):
    """Calcula a área de uma box [x_min, y_min, x_max, y_max]."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def generalized_box_iou_xyxy(box1, box2):
    """
    Calcula GIoU entre duas boxes no formato [x_min, y_min, x_max, y_max].
    box1, box2: listas ou arrays de tamanho 4.
    Retorna um float com o valor de GIoU.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Área de cada box
    area1 = box_area_xyxy(box1)
    area2 = box_area_xyxy(box2)

    # Interseção
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # União
    union = area1 + area2 - inter_area
    iou = inter_area / union if union > 0 else 0.0

    # Área do "fechamento" (box mínima que engloba as duas)
    enclose_x1 = min(x1_1, x1_2)
    enclose_y1 = min(y1_1, y1_2)
    enclose_x2 = max(x2_1, x2_2)
    enclose_y2 = max(y2_1, y2_2)
    enclose_w = max(0, enclose_x2 - enclose_x1)
    enclose_h = max(0, enclose_y2 - enclose_y1)
    enclose_area = enclose_w * enclose_h if (enclose_w > 0 and enclose_h > 0) else 0.0

    if enclose_area == 0:
        return iou  # Se não for possível calcular GIoU, retorna IoU

    giou = iou - (enclose_area - union) / enclose_area
    return giou

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

def calculate_metrics_V3(validation_dataset, model, processor, device, threshold_param=0.5):
    """
    Calcula métricas COCO no conjunto de validação e as perdas no estilo DETR
    (classificação, L1 e GIoU), usando um matcher húngaro simplificado.
    """
    import numpy as np
    import time
    import json
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from tqdm import tqdm
    import torch

    start_time = time.time()

    coco_ground_truth = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    coco_predictions = []

    # Definir categorias (ajuste conforme sua necessidade)
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

        # Adicionar anotações (GT) no formato COCO
        for obj in sample["objects"]:
            coco_ground_truth["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": obj["category_id"],
                "bbox": obj["bbox"],  # [x, y, w, h]
                "area": obj["area"],
                "iscrowd": 0
            })
            annotation_id += 1

        # Obter predições do modelo
        inputs = processor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[height, width]]).to(device)

        results = processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=threshold_param
        )[0]

        # Adicionar predições ao array coco_predictions (formato COCO)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x_min, y_min, x_max, y_max = box.tolist()
            w = x_max - x_min
            h = y_max - y_min

            coco_predictions.append({
                "image_id": image_id,
                "category_id": label.item(),
                "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                "score": score.item()
            })

    # Salvar e avaliar usando COCOeval
    ground_truth_path = "coco_gt.json"
    predictions_path = "coco_preds.json"
    with open(ground_truth_path, "w") as f:
        json.dump(coco_ground_truth, f)
    with open(predictions_path, "w") as f:
        json.dump(coco_predictions, f)

    coco_gt = COCO(ground_truth_path)
    coco_dt = coco_gt.loadRes(predictions_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Cálculo de TP, FP, FN para precisão/recall/f1 (opcional)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_threshold = threshold_param

    def calculate_iou_boxes(boxA, boxB):
        xA, yA, wA, hA = boxA
        xB, yB, wB, hB = boxB
        xA2 = xA + wA
        yA2 = yA + hA
        xB2 = xB + wB
        yB2 = yB + hB
        interX1 = max(xA, xB)
        interY1 = max(yA, yB)
        interX2 = min(xA2, xB2)
        interY2 = min(yA2, yB2)
        interW = max(0, interX2 - interX1)
        interH = max(0, interY2 - interY1)
        interArea = interW * interH
        areaA = wA * hA
        areaB = wB * hB
        union = areaA + areaB - interArea
        if union <= 0:
            return 0.0
        return interArea / union

    for image_id in coco_gt.getImgIds():
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=image_id))

        # Contar TP e FN
        for gt_ann in gt_anns:
            matched = False
            for dt_ann in dt_anns:
                iou_val = calculate_iou_boxes(gt_ann["bbox"], dt_ann["bbox"])
                if iou_val >= iou_threshold:
                    total_tp += 1
                    matched = True
                    break
            if not matched:
                total_fn += 1

        # Contar FP
        for dt_ann in dt_anns:
            matched = False
            for gt_ann in gt_anns:
                iou_val = calculate_iou_boxes(gt_ann["bbox"], dt_ann["bbox"])
                if iou_val >= iou_threshold:
                    matched = True
                    break
            if not matched:
                total_fp += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # ----------------------
    # CHAMADA DA FUNÇÃO DETR
    # ----------------------
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_giou_loss = 0.0
    matched_pairs_count = 0

    for image_id in coco_gt.getImgIds():
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=image_id))
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))

        cls_loss_img, bbox_loss_img, giou_loss_img, pairs_count_img = compute_detr_losses_for_image(
            dt_anns, gt_anns,
            cost_class_weight=1.0,
            cost_bbox_weight=5.0,
            cost_giou_weight=2.0
        )
        total_cls_loss += cls_loss_img
        total_bbox_loss += bbox_loss_img
        total_giou_loss += giou_loss_img
        matched_pairs_count += pairs_count_img

    if matched_pairs_count > 0:
        avg_cls_loss = total_cls_loss / matched_pairs_count
        avg_bbox_loss = total_bbox_loss / matched_pairs_count
        avg_giou_loss = total_giou_loss / matched_pairs_count
    else:
        avg_cls_loss = 0.0
        avg_bbox_loss = 0.0
        avg_giou_loss = 0.0

    # Reunir métricas em um dicionário
    metrics = {
        "eval_map": coco_eval.stats[0],
        "eval_map_50": coco_eval.stats[1],
        "eval_map_75": coco_eval.stats[2],
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1_score": f1_score,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        # Perdas DETR-like
        "loss_cls": round(avg_cls_loss, 4),
        "loss_bbox": round(avg_bbox_loss, 4),
        "loss_giou": round(avg_giou_loss, 4),
        "loss_total": round(avg_cls_loss + avg_bbox_loss + avg_giou_loss, 4),
    }

    # Tempo de execução
    metrics["eval_runtime"] = time.time() - start_time
    metrics["eval_samples_per_second"] = len(validation_dataset) / metrics["eval_runtime"]
    metrics["eval_steps_per_second"] = metrics["eval_samples_per_second"] / max(len(validation_dataset), 1)

    # Métricas específicas por classe (se desejar manter a lógica de sumário por categoria)
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

def calculate_metrics(validation_dataset, model, processor, device):
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
    # Inicializar contadores
    start_time = time.time()
    coco_ground_truth = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    coco_predictions = []

    # Configurar categorias (ajustar IDs conforme necessário)
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
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

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
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Coletar métricas
    metrics = {
        "eval_loss": 0.0,  # Substitua se houver cálculo de perda
        "eval_map": coco_eval.stats[0],
        "eval_map_50": coco_eval.stats[1],
        "eval_map_75": coco_eval.stats[2],
        "eval_map_small": coco_eval.stats[3],
        "eval_map_medium": coco_eval.stats[4],
        "eval_map_large": coco_eval.stats[5],
        "eval_mar_1": coco_eval.stats[6],
        "eval_mar_10": coco_eval.stats[7],
        "eval_mar_100": coco_eval.stats[8],
        "eval_mar_small": coco_eval.stats[9],
        "eval_mar_medium": coco_eval.stats[10],
        "eval_mar_large": coco_eval.stats[11],
    }

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

    # Calcular tempo de execução
    metrics["eval_runtime"] = time.time() - start_time
    metrics["eval_samples_per_second"] = len(validation_dataset) / metrics["eval_runtime"]
    metrics["eval_steps_per_second"] = metrics["eval_samples_per_second"] / len(validation_dataset)

    return metrics

def calculate_metrics_full(validation_dataset, model, processor, device):
    """
    Calcula métricas COCO, incluindo F1-score, Recall, Precision e IoU.

    Args:
        validation_dataset: Dataset de validação.
        model: Modelo carregado.
        processor: Processador de imagens.
        device: Dispositivo (CPU/GPU).

    Returns:
        dict: Métricas calculadas.
    """
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support

    # Inicializar contadores
    start_time = time.time()
    coco_ground_truth = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    coco_predictions = []

    # Configurar categorias (ajustar IDs conforme necessário)
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
    gt_labels, pred_labels, gt_boxes, pred_boxes = [], [], [], []

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
            gt_labels.append(obj["category_id"])
            gt_boxes.append(obj["bbox"])

        # Predições do modelo
        inputs = processor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[height, width]]).to(device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            coco_predictions.append({
                "image_id": image_id,
                "category_id": label.item(),
                "bbox": [round(coord, 2) for coord in box.tolist()],
                "score": score.item()
            })
            pred_labels.append(label.item())
            pred_boxes.append(box.tolist())

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
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Calcular métricas F1-score, Recall e Precision por classe
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_labels, pred_labels, average="macro", zero_division=0
    )

    # Calcular IoU médio entre GT e predições
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    ious = [
        calculate_iou(gt_box, pred_box)
        for gt_box, pred_box in zip(gt_boxes, pred_boxes)
        if gt_box and pred_box
    ]
    mean_iou = np.mean(ious) if ious else 0.0

    # Coletar métricas
    metrics = {
        "eval_loss": 0.0,  # Substitua se houver cálculo de perda
        "eval_map": coco_eval.stats[0],
        "eval_map_50": coco_eval.stats[1],
        "eval_map_75": coco_eval.stats[2],
        "eval_map_small": coco_eval.stats[3],
        "eval_map_medium": coco_eval.stats[4],
        "eval_map_large": coco_eval.stats[5],
        "eval_mar_1": coco_eval.stats[6],
        "eval_mar_10": coco_eval.stats[7],
        "eval_mar_100": coco_eval.stats[8],
        "eval_mar_small": coco_eval.stats[9],
        "eval_mar_medium": coco_eval.stats[10],
        "eval_mar_large": coco_eval.stats[11],
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_mean_iou": mean_iou,
    }

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

    # Calcular tempo de execução
    metrics["eval_runtime"] = time.time() - start_time
    metrics["eval_samples_per_second"] = len(validation_dataset) / metrics["eval_runtime"]
    metrics["eval_steps_per_second"] = metrics["eval_samples_per_second"] / len(validation_dataset)

    return metrics

def calculate_metrics_V2_1(validation_dataset, model, processor, device, threshold_param=0.5):
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
        "info": {
            "description": "Ground truth COCO format for evaluation",
            "version": "1.0",
            "year": 2025
        },
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

            bbox_normalized = normalize_bbox(convert_bbox_xyxy_to_xywh(obj["bbox"]))
            coco_ground_truth["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": obj["category_id"],
                "bbox": obj["bbox"],
                #"bbox": bbox_normalized,
                "area": obj["area"],
                "iscrowd": 0
            })
            annotation_id += 1

        #entrar no modo de inferência
        #model.eval()

        # Predições do modelo
        inputs = processor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[height, width]]).to(device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold_param)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):

            bbox_normalized = normalize_bbox(convert_bbox_xyxy_to_xywh([round(coord, 2) for coord in box.tolist()]))
            coco_predictions.append({
                "image_id": image_id,
                "category_id": label.item(),
                "bbox": [round(coord, 2) for coord in box.tolist()],
                #"bbox": bbox_normalized,
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
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Inicializar variáveis para métricas adicionais
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

    # ----------------------
    # CHAMADA DA FUNÇÃO DETR
    # ----------------------
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_giou_loss = 0.0
    matched_pairs_count = 0

    for image_id in coco_gt.getImgIds():
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=image_id))
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))

        cls_loss_img, bbox_loss_img, giou_loss_img, pairs_count_img = compute_detr_losses_for_image(
            dt_anns, gt_anns,
            cost_class_weight=1.0,
            cost_bbox_weight=5.0,
            cost_giou_weight=2.0
        )
        total_cls_loss += cls_loss_img
        total_bbox_loss += bbox_loss_img
        total_giou_loss += giou_loss_img
        matched_pairs_count += pairs_count_img

    if matched_pairs_count > 0:
        avg_cls_loss = total_cls_loss / matched_pairs_count
        avg_bbox_loss = total_bbox_loss / matched_pairs_count
        avg_giou_loss = total_giou_loss / matched_pairs_count
    else:
        avg_cls_loss = 0.0
        avg_bbox_loss = 0.0
        avg_giou_loss = 0.0

    # Coletar métricas
    metrics = {
        #"eval_loss": eval_loss,  # Substitua se houver cálculo de perda
        "eval_map": coco_eval.stats[0],
        "eval_map_50": coco_eval.stats[1],
        "eval_map_75": coco_eval.stats[2],
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1_score": f1_score,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        # Perdas DETR-like
        "loss_cls": round(avg_cls_loss, 4),
        "loss_bbox": round(avg_bbox_loss, 4),
        "loss_giou": round(avg_giou_loss, 4),
        "loss_total": round(avg_cls_loss + avg_bbox_loss + avg_giou_loss, 4),
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



######  METRICAS ANTERIORES ##########

import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

def compute_eval_loss(model, validation_dataset, processor, device, threshold):
    """
    Calcula a loss de validação para o modelo Table Transformer usando o dataset formatado.
    
    Parâmetros:
    - model: modelo TableTransformerForObjectDetection
    - validation_dataset: lista de dicionários contendo 'image_id', 'image', 'width', 'height' e 'objects'
    - processor: processador de imagens do modelo
    - device: dispositivo de execução (CPU/GPU)
    
    Retorna:
    - loss média sobre o dataset
    """
    
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for data in tqdm(validation_dataset, desc="Computing eval loss"):
            image = data["image"].convert("RGB")
            width, height = data["width"], data["height"]
            
            # Preparar input para o modelo
            inputs = processor(images=[image], return_tensors="pt").to(device)

            #target_sizes = torch.tensor([[image.size[1], image.size[0]]], device=device)
            #results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
            
            # Obter bounding boxes e labels
            target_bboxes = [obj["bbox"] for obj in data["objects"]]
            target_labels = [obj["category_id"] for obj in data["objects"]]
            
            # Garantir que há anotações para processar
            if not target_bboxes:
                continue
            
            # Converter bounding boxes para formato tensor e mover para o dispositivo
            target_bboxes = torch.tensor(target_bboxes, dtype=torch.float32).to(device)
            target_labels = torch.tensor(target_labels, dtype=torch.long).to(device)

            # Forward pass
            outputs = model(**inputs)
            
            # Processar predições
            #target_sizes = torch.tensor([[height, width]]).to(device)
            target_sizes = torch.tensor([[image.size[1], image.size[0]]], device=device)
            results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
           
            pred_bboxes = results["boxes"]  # Bounding boxes preditas
            pred_labels = results["labels"]  # Labels preditos
            
            # Ajustar tamanhos para garantir correspondência entre predições e ground truth
            min_len = min(len(pred_bboxes), len(target_bboxes))
            if min_len == 0:
                continue  # Pular se não houver predições ou ground truth
            
            pred_bboxes = pred_bboxes[:min_len]
            pred_labels = pred_labels[:min_len]
            target_bboxes = target_bboxes[:min_len]
            target_labels = target_labels[:min_len]
            
            # Calcular loss
            bbox_loss = F.smooth_l1_loss(pred_bboxes, target_bboxes)
            #class_loss = F.cross_entropy(pred_labels, target_labels.float())
            class_loss = F.cross_entropy(pred_labels.float(), target_labels.float())
            loss = bbox_loss + class_loss
            
            total_loss += loss.item()
            num_samples += 1
    
    return total_loss / num_samples if num_samples > 0 else 0.0



def calculate_metrics_V2(validation_dataset, model, processor, device, threshold_param=0.5):
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
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Inicializar variáveis para métricas adicionais
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

    #loss
    #eval_loss = compute_eval_loss(model, validation_dataset, processor, device, threshold_param)

    # Coletar métricas
    metrics = {
        #"eval_loss": eval_loss,  # Substitua se houver cálculo de perda
        "eval_map": coco_eval.stats[0],
        "eval_map_50": coco_eval.stats[1],
        "eval_map_75": coco_eval.stats[2],
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1_score": f1_score,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
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


################################