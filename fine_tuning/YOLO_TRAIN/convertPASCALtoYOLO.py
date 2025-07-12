import os
import xml.etree.ElementTree as ET
import glob

def pascal_voc_para_yolo(xml_file):
    
    yolo_data = []

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Erro ao processar {xml_file}: {e}")
        yolo_data.append(xml_file)
        return xml_file

    #tree = ET.parse(xml_file)
    #root = tree.getroot()


    # Obtém as dimensões da imagem
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    # Para cada objeto no XML
    for obj in root.findall("object"):
        class_name = obj.find("name").text

        # Verifica se a classe é "table"
        if class_name != "table":
            continue

        # Obtém as coordenadas da caixa delimitadora
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Converte as coordenadas para o formato YOLO (normalizadas)
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Formato YOLO: class_id x_center y_center width height (class_id é 0 para "table")
        yolo_data.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_data

def converter_pasta_pascal_para_yolo(diretorio, destino):
  
    listError = []
    # Cria o diretório de destino, se não existir
    if not os.path.exists(destino):
        os.makedirs(destino)
    
    cont = 1
    # Varre a pasta em busca de arquivos .xml
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith(".xml"):
            xml_file = os.path.join(diretorio, arquivo)
            #print("caminho xml", xml_file)

            # Converte o arquivo PASCAL VOC para YOLO (somente para a classe "table")
            yolo_data = pascal_voc_para_yolo(xml_file)

            # Nome do arquivo YOLO (mesmo nome, mas com extensão .txt)
            if type(yolo_data) == str:
                print(f"Arquivo de XML com erro: {yolo_data}")
            elif yolo_data:
                yolo_file = os.path.join(destino, arquivo.replace(".xml", ".txt"))

                # Salva os dados no formato YOLO
                with open(yolo_file, "w") as f:
                    f.write("\n".join(yolo_data))

                print(f"Arquivo YOLO gerado: {yolo_file} / contador:{cont}")
            cont +=1
    

diretorio_PASCAL = '/home/aluno-pbarroso/pytorch-pbarroso/DATASET/ALL_PASCAL_VOC'
diretorio_YOLO = '/home/aluno-pbarroso/pytorch-pbarroso/DATASET/ALL_YOLO'

#images = glob.glob(diretorio_PASCAL + '/*.xml')
#print("passou...")

converter_pasta_pascal_para_yolo(diretorio_PASCAL, diretorio_YOLO)

#xml_file = os.path.join(diretorio_PASCAL, "LAB_08_011.xml")
            #print("caminho xml", xml_file)

            # Converte o arquivo PASCAL VOC para YOLO (somente para a classe "table")
#pascal_voc_para_yolo(xml_file)