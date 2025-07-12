# TabLoRA: Table Detection and Structural Recognition via YOLO + TATR with Flask

TabLoRA is an integrated AI pipeline that combines two state-of-the-art models — **YOLOv8 (with LoRA fine-tuning)** for table detection, and **TATR (Table Transformer, also LoRA-tuned)** for structural parsing — into a unified web application using **Flask**.

## 🔍 Overview

We developed an integrated pipeline process using the best checkpoints generated from **YOLO** and **TATR** to create the AI-based application **TabLoRA**.

To deploy this pipeline in an operational environment, both models are encapsulated within a Python-based web application framework (**Flask**). The models are loaded at runtime and exposed through modular inference functions, allowing for **scalable** and **reusable integration**.

### Key Features

- ✅ **YOLO Model**: Performs bounding box detection of tables on full images.
- ✅ **TATR Model**: Processes each table region to infer fine-grained structure (rows, columns, headers, etc.).
- ✅ **Flask API**: Provides a simple UI for uploading images and visualizing predictions.
- ✅ **GPU & CPU support**: Automatically selects the best backend available for real-time inference.
- ✅ **Efficient memory use**: Models are loaded only once and reused across requests.

---

## 🚀 Getting Started

### 1. Clone the repository

bash
git clone https://github.com/your-username/TabLoRA.git
cd TabLoRA


### 2. Setup the Python environment

pip install -r requirements.txt


### 3. Project Structure

```text
TabLoRA/
├── app.py                  # Flask web application
├── models/
│   ├── yolo_model.pt       # YOLO checkpoint (LoRA fine-tuned)
│   └── tatr_checkpoint/    # TATR LoRA checkpoint directory
├── static/
│   └── saida/              # Output directory for processed images
├── templates/
│   └── index.html          # Simple HTML interface
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

### 4.Start the Flask App

python app.py

Open your browser at http://127.0.0.1:5000 and upload an image of a document containing tables.

The application will:
- ✅ Run the YOLO model to detect all table bounding boxes.
- ✅ Crop each detected table region.
- ✅ Feed each cropped region into TATR for fine-grained structural analysis.
- ✅ Display the results with colored bounding boxes and class labels.

Output

Processed images are saved in the static/saida/ directory with:
yolo_<filename>.jpg → image with detected table regions
tatr_<filename>.jpg → image with detailed structure annotations

🧠 Models
YOLOv8: Finetuned with LoRA for detecting table boundaries.
TATR (Table Transformer): Finetuned with LoRA for table structural recognition.
Models are loaded once and reused across all inferences, minimizing loading overhead and enabling near real-time processing.

📌 Requirements
Python 3.8+
PyTorch
Transformers
Flask
PEFT (LoRA support)
PIL, OpenCV, Matplotlib
ultralytics

📜 License
This project is licensed under the MIT License. See LICENSE for details.

🙌 Acknowledgements
Ultralytics YOLOv8

Table Transformer by Microsoft

LoRA: Low-Rank Adaptation


