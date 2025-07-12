import pandas as pd
from pdf2image import convert_from_path
import os
import configparser
import fitz
from PIL import Image
import numpy as np

########### CARREGAR VARIAVEIS CONFIG ###############
config = configparser.ConfigParser()
config.read('/ROOT_PATH/config.ini')
ROOT_CERT_IN = config['DEFAULT']['caminho_absoluto_certificados_in']
ROOT_CERT_OUT = config['DEFAULT']['caminho_absoluto_certificados_out']
#ARQ1_PATH = config['DEFAULT']['caminho_arq_1']
PATH_LAB1 = config['DEFAULT']['PATH_LAB1']

print(os.path.abspath("config.ini"))

########### CARREGAR VARIAVEIS DE INICIALIZACAO ###############

#fileIN = ROOT_CERT_IN + ARQ1_PATH
imgOUT = ROOT_CERT_OUT + 'teste.png'
page_number = 3 

################### FUNÇÕES ################

def InfoPDF(pathPDF):

  lenPages = 0
  doc = fitz.open(pathPDF)

  lenPages = len(doc)
  for pagina in doc:
    isText = bool(pagina.get_text())
    break

  typeArq = ["TEXT" if isText else "IMAGE"]

  return typeArq, lenPages

def listFiles(CERT_PATH, LAB_PATH):

  dfArq = pd.DataFrame()

  if LAB_PATH is None:
    dirs = [nome for nome in os.listdir(CERT_PATH) if os.path.isdir(os.path.join(CERT_PATH, nome))]
  else:
    dirs = [LAB_PATH]

  i = 0
  for dir in dirs:

    #coletando dados do diretorio (id, laboratorio)
    lstDir = dir.split("_")
    print("lstDir ", lstDir);

    if(len(lstDir)==3):

      for file in os.listdir(os.path.join(CERT_PATH, dir)):

        #é arquivo PDF
        if file.lower().endswith(".pdf"):

          pathFile = CERT_PATH + dir + "/" + file
          print("pathFile ", pathFile);
          dfArq.at[i,"LAB"] = lstDir[2]
          dfArq.at[i,'PATH'] = pathFile

          typeArq, qtdPages = InfoPDF(pathFile)
          dfArq.at[i,'TYPE'] = typeArq
          dfArq.at[i,'QTDPAGES'] = qtdPages

          i = i + 1

  return dfArq

#CONVERTER PARA PNG
def pdf_page_to_png(pdf_path, page_number, output_path):
    # Convertendo a página do PDF para uma lista de imagens
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)

    # Salvando a imagem como PNG
    images[0].save(output_path, 'PNG')


def converter_png_para_jpg(caminho_entrada, caminho_saida):
    # Abre a imagem PNG
    imagem = Image.open(caminho_entrada)
    
    # Converte para RGB (necessário para salvar em JPG, pois PNG pode ter transparência)
    imagem = imagem.convert("RGB")
    
    # Salva a imagem no formato JPG
    imagem.save(caminho_saida, "JPEG")
    
    print(f"Imagem convertida e salva em {caminho_saida}")


def renomear_arquivos(pasta_origem, pasta_dest, prefixo):
    contador = 1

    # Lista todos os arquivos na pasta
    arquivos = os.listdir(pasta_origem)
    
    # Ordena os arquivos para garantir a ordem correta
    arquivos.sort()

    for arquivo in arquivos:
        # Gera o novo nome usando o contador
        novo_nome = f"{prefixo}{contador:03d}"

        # Obtém a extensão do arquivo
        extensao = os.path.splitext(arquivo)[1]

        # Monta o caminho completo para o arquivo atual e o novo nome
        caminho_antigo = os.path.join(pasta_origem, arquivo)
        caminho_novo = os.path.join(pasta_dest, novo_nome + extensao)

        # Renomeia o arquivo
        os.rename(caminho_antigo, caminho_novo)
        print(f"Renomeado: {caminho_antigo} -> {caminho_novo}")

        # Incrementa o contador
        contador += 1

# Exemplo de uso
lab = "LAB_02/"

#pasta_entrada = ROOT_CERT_IN + lab + "img/"
pasta_entrada = ROOT_CERT_OUT + lab + "png/"
pasta_saida = ROOT_CERT_OUT + lab + "jpg/"
pasta_jpg = ROOT_CERT_OUT + lab + "jpg/"

renomear_arquivos (pasta_entrada, pasta_saida, "LAB_02_")

for file in os.listdir(pasta_saida):
  #print(file)
  converter_png_para_jpg(pasta_saida + file, pasta_jpg + file.replace("png", "jpg"))

