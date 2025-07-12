from ultralyticsplus import YOLO
import cv2
import ast
# Load a pretrained YOLO model (recommended for training)

model_detect = YOLO('foduucom/table-detection-and-extraction')


#imgpath = "/home/aluno-pbarroso/pytorch-pbarroso/ft_tatr/Certificados/In/LAB_01_CTM/img/LAB_01_CTM_001_table_0.png"
imgpath = "/home/aluno-pbarroso/pytorch-pbarroso/ft_tatr/Certificados/Out/LAB_01_CTM/png/LAB01_001.png"
           
#imgpath = "/home/aluno-pbarroso/pytorch-pbarroso/ft_tatr/Certificados/Out/BCS/bcs_a_original.jpg"
image = cv2.imread(imgpath)

# Perform object detection on an image using the model
results = model_detect.predict(imgpath)

for result in results:
  print("result com dados carregados....")
  #boxes = result.boxes.xyxyn
  boxes = result.boxes.xyxy.tolist()

  string = "[{'label': 'table', 'score': 0.9998989105224609, 'bbox': " + str(boxes[0]) + "}]"
  #key, value = string.split(": ", 1)
  #value = ast.literal_eval(string)
  #dicionario = {key: value}
  #print(value[0]["bbox"][0])
  #result.save(filename='result.jpg')
  print(string)

'''
import ast

# String fornecida

# Dividir a string na chave e valor
≈

# Converter o valor para uma lista de dicionários usando ast.literal_eval
value = ast.literal_eval(value)

# Montar o dicionário
dicionario = {key: value}

print(dicionario)

'''