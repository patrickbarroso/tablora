import cv2
from pdf2image import convert_from_path

#CONVERTER PARA PNG
def pdf_page_to_png(pdf_path, page_number, output_path):
    # Convertendo a página do PDF para uma lista de imagens
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)

    # Salvando a imagem como PNG
    images[0].save(output_path, 'PNG')

def aumentar_qualidade_e_contraste(imagem_path, fator_contraste, fator_brilho):
    # Carregar a imagem
    imagem = cv2.imread(imagem_path)

    # Converter a imagem para o espaço de cores LAB (Luminância, Azul, Vermelho)
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)

    # Separar os canais L, A, B
    l, a, b = cv2.split(lab)

    # Aplicar o aumento de contraste na imagem L (luminância)
    l = cv2.add(l, fator_brilho)
    l = cv2.multiply(l, fator_contraste)

    # Mesclar novamente os canais LAB
    lab = cv2.merge((l, a, b))

    # Converter a imagem de volta para o espaço de cores BGR
    imagem_contraste = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return imagem_contraste


