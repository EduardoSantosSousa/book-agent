import os
import re
from datetime import datetime
from google.cloud import storage
import tempfile
import logging

logger = logging.getLogger(__name__)

class GCSFileLoader:
    """Carrega arquivos mais recentes do Google Cloud Storage"""
    
    def __init__(self, bucket_name="book-agent-embeddings-403941621548"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def get_latest_files(self, prefix_pattern="embeddings/"):
        """Encontra os arquivos mais recentes baseados no timestamp"""
        try:
            blobs = list(self.client.list_blobs(self.bucket_name, prefix=prefix_pattern))
            
            # Filtrar apenas arquivos .npy e .faiss
            npy_files = [b for b in blobs if b.name.endswith('.npy')]
            faiss_files = [b for b in blobs if b.name.endswith('.faiss')]
            
            # Extrair timestamps dos nomes dos arquivos
            dated_files = []
            
            for blob in npy_files + faiss_files:
                # Procurar padrão de data no nome do arquivo
                match = re.search(r'(\d{8}_\d{6})', blob.name)
                if match:
                    timestamp = match.group(1)
                    try:
                        date_obj = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                        dated_files.append((date_obj, blob.name, blob))
                    except ValueError:
                        continue
            
            # Ordenar por data (mais recente primeiro)
            dated_files.sort(key=lambda x: x[0], reverse=True)
            
            if not dated_files:
                logger.warning("Nenhum arquivo com timestamp encontrado")
                # Tentar arquivos sem timestamp
                all_files = []
                for blob in npy_files + faiss_files:
                    all_files.append((datetime.min, blob.name, blob))
                all_files.sort(key=lambda x: x[1])
                dated_files = all_files
            
            return dated_files
            
        except Exception as e:
            logger.error(f"Erro ao listar arquivos do GCS: {e}")
            return []
    
    def download_latest_embeddings(self, local_dir="embeddings/latest"):
        """Baixa os arquivos de embeddings mais recentes"""
        try:
            os.makedirs(local_dir, exist_ok=True)
            
            # Encontrar arquivos mais recentes
            dated_files = self.get_latest_files()
            
            if not dated_files:
                raise FileNotFoundError("Nenhum arquivo encontrado no bucket")
            
            # Separar arquivos .npy e .faiss mais recentes
            latest_npy = None
            latest_faiss = None
            
            for date_obj, filename, blob in dated_files:
                if filename.endswith('.npy') and not latest_npy:
                    latest_npy = (filename, blob)
                elif filename.endswith('.faiss') and not latest_faiss:
                    latest_faiss = (filename, blob)
                
                if latest_npy and latest_faiss:
                    break
            
            # Baixar arquivos
            downloaded_files = []
            
            for file_type, file_info in [('embeddings', latest_npy), ('index', latest_faiss)]:
                if file_info:
                    filename, blob = file_info
                    local_path = os.path.join(local_dir, os.path.basename(filename))
                    
                    logger.info(f"Baixando {filename} para {local_path}")
                    blob.download_to_filename(local_path)
                    downloaded_files.append(local_path)
                    
                    logger.info(f"✅ {file_type} baixado: {local_path}")
            
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Erro ao baixar embeddings: {e}")
            raise
    
    def get_file_by_pattern(self, pattern):
        """Busca arquivo por padrão específico"""
        blobs = list(self.client.list_blobs(self.bucket_name))
        
        for blob in blobs:
            if re.search(pattern, blob.name):
                return blob
        
        return None

# Singleton instance
_gcs_loader = None

def get_gcs_loader():
    """Obtém instância singleton do GCS loader"""
    global _gcs_loader
    if _gcs_loader is None:
        _gcs_loader = GCSFileLoader()
    return _gcs_loader