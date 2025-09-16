import os
import shutil
import kagglehub

def import_datasets():
    """
    Baixa datasets do Kaggle e salva na pasta 'data/' da raiz do projeto.
    Renomeia os arquivos para evitar sobrescrever arquivos com mesmo nome.
    """
    
    BASE_DIR = os.getcwd()  
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR))
    
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"✅ Data directory: {DATA_DIR}\n")
    
    datasets = {
        "mustafatz": "iammustafatz/diabetes-prediction-dataset",
        "uciml": "uciml/pima-indians-diabetes-database",
        "mathchi": "mathchi/diabetes-data-set"
    }
    
    download_paths = {}
    for prefix, kaggle_id in datasets.items():
        print(f"📥 Baixando dataset: {kaggle_id} ...")
        path = kagglehub.dataset_download(kaggle_id)
        download_paths[prefix] = path
    
    def copy_to_data(download_path, prefix, data_dir=DATA_DIR):
        for file_name in os.listdir(download_path):
            src = os.path.join(download_path, file_name)
            if os.path.isfile(src):
                dst_name = f"{prefix}.csv"
                dst = os.path.join(data_dir, dst_name)
                shutil.copy(src, dst)
                print(f"✅ Copiado: {dst_name}")
    
    for prefix, path in download_paths.items():
        copy_to_data(path, prefix)
    
    print("\n🎉 Todos os datasets foram importados para a pasta 'data/'.")

# --- Permite rodar direto ---
if __name__ == "__main__":
    import_datasets()
