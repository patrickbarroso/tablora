from flask import Flask, request, render_template, redirect, url_for
import os
import torch
import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralyticsplus import render_result
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from peft import PeftModel
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'static/saida'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Caminhos fixos dos modelos
YOLO_MODEL_PATH = 'models/yolo_model.pt'
TATR_CHECKPOINT_PATH = 'models/tatr_checkpoint'
TATR_BASE_MODEL = 'microsoft/table-transformer-structure-recognition'

# Carregamento dos modelos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_yolo = YOLO(YOLO_MODEL_PATH)
base_tatr = TableTransformerForObjectDetection.from_pretrained(TATR_BASE_MODEL).to(device)
model_tatr = PeftModel.from_pretrained(base_tatr, TATR_CHECKPOINT_PATH).to(device)
model_tatr.eval()
image_processor = AutoImageProcessor.from_pretrained(TATR_BASE_MODEL)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            # --- YOLO INFERÊNCIA ---
            img_cv2 = cv2.imread(image_path)
            results_yolo = model_yolo(img_cv2)
            yolo_output_path = os.path.join(UPLOAD_FOLDER, f"yolo_{file.filename}")
            img_yolo = results_yolo[0].plot()
            cv2.imwrite(yolo_output_path, img_yolo)

            # --- TATR INFERÊNCIA ---
            image_pil = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                inputs = image_processor(images=[image_pil], return_tensors="pt")
                outputs = model_tatr(**inputs.to(device))
                target_sizes = torch.tensor([[image_pil.size[1], image_pil.size[0]]])
                results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

            image_with_boxes = image_pil.copy()
            draw = ImageDraw.Draw(image_with_boxes)

            colors = {
                "table": (0, 0, 255),
                "table column": (255, 0, 0),
                "table row": (0, 255, 0),
                "table column header": (255, 165, 0),
                "table projected row header": (128, 0, 128),
                "table spanning cell": (0, 255, 255)
            }

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = model_tatr.config.id2label[label.item()]
                box = [round(i, 2) for i in box.tolist()]
                color = colors.get(label_name, (255, 255, 255))
                hex_color = "#{:02x}{:02x}{:02x}".format(*color)
                draw.rectangle(box, outline=hex_color, width=3)
                draw.text((box[0], box[1]), f"{label_name} ({round(score.item(), 2)})", fill=hex_color)

            tatr_output_path = os.path.join(UPLOAD_FOLDER, f"tatr_{file.filename}")
            image_with_boxes.save(tatr_output_path)

            return render_template("index.html",
                                   yolo_image=yolo_output_path,
                                   tatr_image=tatr_output_path)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
