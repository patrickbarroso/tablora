from ultralytics import YOLO
import torch
import optuna
import yaml
import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import KFold
import os
import glob
import shutil
import random
import sys 
import time 

#PARAMETROS PARA UTILIZAR
EPOCH_MIN = 150 # epoca maxima
EPOCH_MAX = 200 #epoca minima
QTD_TRIALS = 15 # quantidade de tentativas
QTD_SPLITS = 5 # quantidade de folds para usar

# ========================================================
# Configuração do dispositivo
# ========================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Dispositivo em uso:", device)

# ========================================================
# Configuração do Logger
# ========================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('yolov8_cv.log')
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Dispositivo em uso:", device)
# ========================================================
# Caminhos importantes
# ========================================================
DATASET_DIR = '/home/aluno-pbarroso/pytorch-pbarroso/DATASET'
BASE_DIR = '/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_LORA'
CV_DIR = os.path.join(BASE_DIR, 'CV_YOLO11')
arquivo_yolo_yaml = os.path.join(BASE_DIR, "main", 'yolo.yaml')

# Cria o diretório CV se não existir
os.makedirs(CV_DIR, exist_ok=True)

# Modelos YOLO que serão testados

MODELS = {
    'YOLO-CEN01': '/home/aluno-pbarroso/yolo11n.pt',
    'YOLO-CEN02': '/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_TRAIN/Model/YOLO11n_18062025SP2_SEMLORA.pt',
    'YOLO-CEN03': '/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_LORA/Model/YOLO11n_LORA_23062025SP4.pt',
    'YOLO-CEN04': '/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_LORA/Model/YOLO11n_LORA_18062025SP2.pt',
    'YOLO-CEN05': '/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_LORA/Model/YOLO11n_LORA_19062025SP1.pt',
    'YOLO-CEN06': '/home/aluno-pbarroso/pytorch-pbarroso/FT_YOLO_LORA/Model/YOLO11n_LORA_19062025SP1.pt'
}

# ========================================================
# Funções auxiliares para K-Fold
# ========================================================
def create_temp_labels(image_paths, labels_dir):
    """Cria labels temporários para imagens sem labels válidos"""
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        
        if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
            with open(label_path, 'w') as f:
                f.write("0 0.5 0.5 0.8 0.8")  # Label padrão para tabela central
            logger.warning(f"Criado label temporário: {label_path}")

def load_dataset_info(yaml_path):
    """Carrega informações do dataset conforme sua estrutura"""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    
    # Caminhos absolutos para imagens
    train_img_dir = os.path.join(DATASET_DIR, 'train', 'images')
    val_img_dir = os.path.join(DATASET_DIR, 'val', 'images')
    
    # Lista de imagens
    train_images = glob.glob(os.path.join(train_img_dir, '*.[jp][pn]g'))
    val_images = glob.glob(os.path.join(val_img_dir, '*.[jp][pn]g'))
    
    # Caminhos para labels
    train_labels_dir = os.path.join(DATASET_DIR, 'train', 'labels')
    val_labels_dir = os.path.join(DATASET_DIR, 'val', 'labels')
    
    # Garante que os diretórios de labels existam
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Cria labels temporários se necessário
    create_temp_labels(train_images, train_labels_dir)
    create_temp_labels(val_images, val_labels_dir)
    
    return {
        'train_images': train_images,
        'val_images': val_images,
        'train_labels_dir': train_labels_dir,
        'val_labels_dir': val_labels_dir
    }

def verify_label_content(images, labels_dir):
    """Verifica se os labels têm conteúdo válido"""
    valid_count = 0
    for img_path in images:
        img_name = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            valid_count += 1
        else:
            logger.warning(f"Label inválido/ausente: {label_path}")
    
    print(f"Labels válidos: {valid_count}/{len(images)}")
    logger.info("Labels válidos: %s/%s", valid_count, len(images))
    return valid_count > 0  # Pelo menos um label válido

def create_fold_structure(train_images, val_images, train_labels_dir, val_labels_dir, fold_dir):
    """Cria estrutura para um fold específico"""
    # Cria diretórios para o fold
    fold_train_dir = os.path.join(fold_dir, 'train')
    fold_val_dir = os.path.join(fold_dir, 'val')
    
    os.makedirs(os.path.join(fold_train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(fold_train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(fold_val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(fold_val_dir, 'labels'), exist_ok=True)

    def copy_files(file_list, src_labels, dest_dir):
        """Copia imagens e labels correspondentes"""
        for img_path in file_list:
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            
            # Copia imagem
            shutil.copy2(img_path, os.path.join(dest_dir, 'images', img_name))
            
            # Copia label se existir
            src_label = os.path.join(src_labels, label_name)
            if os.path.exists(src_label):
                shutil.copy2(src_label, os.path.join(dest_dir, 'labels', label_name))

    # Copia arquivos para o fold
    copy_files(train_images, train_labels_dir, fold_train_dir)
    copy_files(val_images, val_labels_dir, fold_val_dir)
    
    # Cria dataset.yaml para o fold
    fold_yaml = {
        'path': os.path.abspath(fold_dir),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'table'},
        'nc': 1
    }
    
    yaml_path = os.path.join(fold_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(fold_yaml, f)
    
    return yaml_path

# ========================================================
# Função de Treinamento para YOLOv8 com K-Fold
# ========================================================
def train_yolo_model(trial, model_name, fold_yaml_path, fold_num):
    """Treina o modelo YOLOv8 com os hiperparâmetros do trial para um fold específico"""
    # Parâmetros de loss
    loss_params = {
        'box': trial.suggest_float("box", 0.05, 0.5),
        'cls': 0.0,
        'dfl': trial.suggest_float("dfl", 0.5, 2.0)
    }
    
    train_params = {
        'epochs': trial.suggest_int("epochs", EPOCH_MIN, EPOCH_MAX),
        #'epochs': trial.suggest_int("epochs", 1, 1),
        'batch': trial.suggest_categorical("batch_size", [8, 16]),
        'lr0': trial.suggest_float("lr0", 1e-5, 1e-3, log=True),
        'lrf': trial.suggest_float("lrf", 0.01, 0.2),
        'momentum': trial.suggest_float("momentum", 0.9, 0.98),
        'weight_decay': trial.suggest_float("weight_decay", 0.0001, 0.001),
        'warmup_epochs': trial.suggest_int("warmup_epochs", 1, 3),
        'warmup_momentum': trial.suggest_float("warmup_momentum", 0.5, 0.8)
    }

    params = {**loss_params, **train_params}

    # Carrega o modelo específico
    model = YOLO(MODELS[model_name]).to(device)

    # Desativa verificação de atualização
    os.environ['YOLO_NO_UPDATE_CHECK'] = '1'
    
    results = model.train(
        data=fold_yaml_path,
        **params,
        imgsz=640,
        patience=7,
        device=str(device),
        workers=3,
        single_cls=True,
        optimizer='AdamW',
        project=os.path.join(CV_DIR, 'results'),
        name=f'{model_name}_trial_{trial.number}_fold{fold_num}',
        exist_ok=True
    )
    '''
    if model.trainer.losses:
        print("entrou model.trainer.losses")
        last_loss = model.trainer.losses[-1]
        box_loss = last_loss[0]  # box loss
        dfl_loss = last_loss[2]  # dfl loss (cls loss está no índice 1)
        print("box_loss ", box_loss)
        print("dfl_loss ", dfl_loss)
    '''
    #metrics = model.val()
    
    #print("metrics train/box_loss ", )
    # Caminho para o arquivo de resultados
    results_file = os.path.join(CV_DIR, 'results', f'{model_name}_trial_{trial.number}_fold{fold_num}', 'results.csv')
    print("results_file ", results_file)
    logger.info("results_file %s", results_file)

    # Lê o arquivo CSV
    try:
        import pandas as pd
        df = pd.read_csv(results_file)
        #print("df ", df.columns)
        # Pega os valores do último epoch
        n_epochs = len(df)
        #target_idx = max(0, n_epochs - 11)
        best_box_loss_idx = df['val/box_loss'].idxmin()

        box_loss = df['val/box_loss'].iloc[best_box_loss_idx]
        dfl_loss = df['val/dfl_loss'].iloc[best_box_loss_idx]

        #print("box_loss2 ", box_loss)
        #print("dfl_loss2 ", dfl_loss)

        #sys.exit()
    except Exception as e:
        logger.error(f"Erro ao ler arquivo de resultados: {str(e)}")
        box_loss = float('inf')
        dfl_loss = float('inf')

    return box_loss, dfl_loss

# ========================================================
# Função Objetivo para Optuna com K-Fold
# ========================================================
def objective(trial, model_name):
    start_time = time.time()
    
    # Seleciona um modelo aleatório para este trial
    #model_name = random.choice(list(MODELS.keys()))
    #model_name = trial.params['model_name']

    #trial.set_user_attr('model_name', model_name)  # Add this line
    print("="*60)
    print(f"\nIniciando NOVO OBJECTIVE - TRIAL: {trial.number} ")
    print(f"\nMODELO: {model_name}")
    print("="*60) 
    
    n_splits = QTD_SPLITS  # Número de folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    dataset_info = load_dataset_info(arquivo_yolo_yaml)
    train_images = dataset_info['train_images']
    train_labels_dir = dataset_info['train_labels_dir']
    
    #print("train_images ", train_images)
    #print("train_labels_dir ", train_labels_dir)

    # Verifica labels de treino
    if not verify_label_content(train_images, train_labels_dir):
        logger.error("Labels de treino inválidos. Corrija antes de continuar.")
        return float('inf')  # Retorna um valor muito alto para indicar falha
    
    box_losses = []
    dfl_losses = []
    temp_fold_dir = os.path.join(CV_DIR, 'kfold_temp')
    os.makedirs(temp_fold_dir, exist_ok=True)
    
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(train_images)):
        
        fold_train_images = [train_images[i] for i in train_idx]
        fold_val_images = [train_images[i] for i in val_idx]
        
        fold_dir = os.path.join(temp_fold_dir, f'fold_{fold_num}')
        fold_yaml_path = create_fold_structure(
            fold_train_images, 
            fold_val_images,
            train_labels_dir,
            train_labels_dir,  # Usamos os mesmos labels para validação
            fold_dir
        )
             
        print("="*60)
        print(f"\nTRIAL {trial.number}/{QTD_TRIALS} / MODELO {model_name} / FOLD {fold_num + 1}/{n_splits} ")
        logger.info("TRIAL %s/%s / MODELO  / %s FOLD %s/%s ", str(trial.number), str(QTD_TRIALS), model_name, str(fold_num + 1), str(n_splits))
        
        try:
            
            #print("="*60)
            #print(f"Iniciando treinamento model = {model_name} / fold = {fold_num}")
            #logger.info("Iniciando treinamento model = %s / fold = %s", model_name, str(fold_num))

            print("INICIANDO TREINAMENTO ...")
            logger.info("INICIANDO TREINAMENTO ...")
            print("="*60)

            box_loss, dfl_loss = train_yolo_model(trial, model_name, fold_yaml_path, fold_num)

            print("="*60)
            print("Resultados de loss:")
            print(f"box_loss {box_loss} / dfl_loss = {dfl_loss}")
            logger.info("box_loss %s / dfl_loss = %s", str(box_loss),  str(dfl_loss))
            

            #sys.exit()
            box_losses.append(box_loss)
            dfl_losses.append(dfl_loss)
            
            # Calcula a média ponderada atual
            current_weighted_avg = (5 * box_loss + 1 * dfl_loss) / 6
            
            # Imprime os resultados do fold
            print(f"Fold {fold_num + 1} concluído para modelo {model_name}:")
            print(f"Box Loss: {box_loss} (peso 5)")
            print(f"DFL Loss: {dfl_loss} (peso 1)")
            print(f"Média ponderada atual: {current_weighted_avg}")
            print("="*60)

            logger.info("Fold %s concluído para modelo %s:", str(fold_num + 1), model_name)
            logger.info("Box Loss: %s (peso 5)", str(box_loss))
            logger.info("DFL Loss: %s (peso 1)", str(dfl_loss))
            logger.info("Média ponderada atual: %s", str(current_weighted_avg))
            
        except Exception as e:
            print(f"Erro no fold {fold_num + 1} para modelo {model_name}: {str(e)}")
            logger.error("Erro no fold %s para modelo %s", str(fold_num + 1), str(e))
            box_losses.append(float('inf'))
            dfl_losses.append(float('inf'))
            sys.exit()
            
        shutil.rmtree(fold_dir, ignore_errors=True)
    
    # Calcula as médias finais
    avg_box_loss = np.mean(box_losses)
    avg_dfl_loss = np.mean(dfl_losses)
    weighted_avg_loss = (5 * avg_box_loss + 1 * avg_dfl_loss) / 6  # Média ponderada
    
    duration = (time.time() - start_time) / 60
    print(f"\nTrial {trial.number} concluído para modelo {model_name}")
    print(f"Médias finais:")
    print(f"Box Loss: {avg_box_loss:.4f} (peso 5)")
    print(f"DFL Loss: {avg_dfl_loss:.4f} (peso 1)")
    print(f"Média ponderada final (objetivo): {weighted_avg_loss:.4f}")
    print(f"Duração total: {duration:.2f} minutos")

    logger.info("\nTrial %s concluído para modelo %s ", str(trial.number), model_name)
    logger.info("Médias finais:")
    logger.info("Box Loss: %s (peso 5)", str(avg_box_loss))
    logger.info("DFL Loss: %s (peso 1)", str(avg_dfl_loss))
    logger.info("Média ponderada final (objetivo): %s",  str(weighted_avg_loss))
    logger.info("Duração total: %s minutos", str(duration))

    try:
        weighted_avg_loss = (5 * avg_box_loss + 1 * avg_dfl_loss) / 6
        print("retornando media do OBJECTIVE = ", weighted_avg_loss)
        return float(weighted_avg_loss)
    except Exception as e:
        print("Exception weighted_avg_loss ", weighted_avg_loss)
        print("OBJECTIVE = Error calculating weighted average: ", str(e))
        logger.error("OBJECTIVE = Error calculating weighted average: %s", str(e))
        sys.exit()
        return float('inf')


# Função para armazenar os resultados por modelo
def callback(study, trial):
    # Pega o nome do modelo dos parâmetros do trial
    model_name = trial.params['model_name']
    value = trial.value
    model_results[model_name]['trials'].append({
        'params': trial.params,
        'value': value
    })
    
    # Atualiza o melhor resultado para este modelo
    if value < model_results[model_name]['best_value']:
        model_results[model_name]['best_value'] = value
        model_results[model_name]['best_params'] = trial.params
    
# ========================================================
# Execução do Estudo Optuna
# ========================================================
# ========================================================
# Execução do Estudo Optuna
# ========================================================
if __name__ == "__main__":
    # Dicionário para armazenar resultados por modelo
    model_results = {model_name: {'trials': [], 'best_value': float('inf'), 'best_params': None} 
                    for model_name in MODELS.keys()}
    
    # Cria um estudo Optuna único
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    print(f"Iniciando otimização com {QTD_TRIALS} trials / {QTD_SPLITS} folds / por modelos aleatórios...")
    print("Objetivo: minimizar a média ponderada dos losses (5*box + 1*dfl)/6")

    logger.info("Iniciando otimização com %s trials / %s folds / por modelos aleatórios...", str(QTD_TRIALS), str(QTD_SPLITS))
    logger.info("Objetivo: minimizar a média ponderada dos losses (5*box + 1*dfl)/6")
    
    # 1. Prepara a lista de modelos para os trials
    models_for_trials = [random.choice(list(MODELS.keys())) for _ in range(QTD_TRIALS)]

    # 2. Executa CADA TRIAL MANUALMENTE com seu modelo atribuído
    for model_name in models_for_trials:
        trial = study.ask()  # Cria um novo trial
        trial.set_user_attr('model_name', model_name)  # Atribui o modelo
        value = objective(trial, model_name)  # Executa o trial
        study.tell(trial, value)  # Registra os resultados

        # Atualiza model_results manualmente
        model_results[model_name]['trials'].append({
            'params': trial.params,
            'value': value
        })
        if value < model_results[model_name]['best_value']:
            model_results[model_name]['best_value'] = value
            model_results[model_name]['best_params'] = trial.params
    
    # Executa os trials - APENAS UMA VEZ
    #study.optimize(lambda trial: objective(trial), n_trials=QTD_TRIALS, callbacks=[callback])
    
    # Resultados finais
    print("\n" + "="*60)
    print("Resumo dos resultados por modelo:")

    logger.info("\n" + "="*60)
    logger.info("Resumo dos resultados por modelo:")

    for model_name, results in model_results.items():
        if results['trials']:  # Só mostra modelos que foram testados
            print(f"\nModelo: {model_name}")
            print(f"Número de trials: {len(results['trials'])}")
            print(f"Melhor média ponderada: {results['best_value']}")

            logger.info("Modelo: %s", model_name)
            logger.info("Número de trials: %s", str(len(results['trials'])))
            logger.info("Melhor média ponderada: %s", str(results['best_value']))

            if results['best_params']:
                print("Melhores parâmetros:")
                logger.info("Melhores parâmetros:")
                for key, value in results['best_params'].items():
                    print(f"  {key}: {value}")
                    logger.info("%s: %s", key, value)
    
    print("="*60)
    logger.info("="*60)

    print("model_results ",model_results)
    print("model_results.items() ", model_results.items())    
    # Encontra o melhor modelo global
    best_model_name = min(
        [(name, res) for name, res in model_results.items() if res['trials']],
        key=lambda x: x[1]['best_value']
    )[0]
    best_value = model_results[best_model_name]['best_value']
    best_params = model_results[best_model_name]['best_params']
    
    print(f"\nMelhor modelo global: {best_model_name}")
    print(f"Melhor média ponderada: {best_value}")
    print(f"Melhores parâmetros: {best_params}")

    logger.info("Melhor modelo global: %s", best_model_name)
    logger.info("Melhor média ponderada: %s", str(best_value))
    logger.info("Melhores parâmetros: %s", str(best_params))
    
    # Salva todos os resultados em um arquivo YAML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(CV_DIR, f'optuna_results_random_models_{timestamp}.yaml')
    with open(results_file, 'w') as f:
        yaml.dump({
            'best_model': {
                'name': best_model_name,
                'value': float(best_value),
                'params': best_params
            },
            'all_models': model_results,
            'weights': {'box_loss': 5, 'dfl_loss': 1}
        }, f)
    print(f"\nResultados salvos em: {results_file}")
    logger.info("Resultados salvos em %s", results_file)