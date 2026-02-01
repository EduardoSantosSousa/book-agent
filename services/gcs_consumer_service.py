# D:\Django\book_agent\services\gcs_consumer_service.py

import io
import numpy as np
import faiss
import logging
from google.cloud import storage
from typing import Tuple, Optional, Dict
import tempfile
import re
from datetime import datetime
import os
import shutil

logger = logging.getLogger(__name__)

class GCSEmbeddingConsumer:
    """
    Consumidor puro de embeddings do GCS.
    Não gera, não atualiza - apenas carrega a versão mais recente.
    """
    
    def __init__(self, bucket_name: str = "book-agent-embeddings-bucket",
                 embeddings_prefix: str = "embeddings/"):
        self.bucket_name = bucket_name
        self.embeddings_prefix = embeddings_prefix
        
        # Inicializar cliente GCS
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Cache em memória
        self.current_embeddings = None
        self.current_index = None
        self.version_info = None
        self.loaded_at = None

        # Cria diretório temp no projeto (opcional, para debug)
        self.temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_data")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"📦 GCS Consumer inicializado")
        logger.info(f"   Bucket: {bucket_name}")
        logger.info(f"   Prefixo: {embeddings_prefix}")
    
    def _extract_timestamp(self, filename: str) -> Optional[datetime]:
        """Extrai timestamp do nome do arquivo"""
        try:
            # Procura padrão YYYYMMDD_HHMMSS
            match = re.search(r'(\d{8}_\d{6})', filename)
            if match:
                return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except:
            pass
        return None
    
    def find_latest_embeddings_pair(self) -> Dict:
        """
        Encontra o par mais recente de embeddings no bucket.
        Retorna caminhos dos arquivos .npy e .faiss mais recentes.
        """
        try:
            logger.info(f"🔍 Buscando embeddings mais recentes em {self.bucket_name}/{self.embeddings_prefix}")
            
            # Listar todos os blobs no prefixo
            blobs = list(self.client.list_blobs(self.bucket_name, prefix=self.embeddings_prefix))
            
            if not blobs:
                raise Exception(f"Nenhum arquivo encontrado em {self.embeddings_prefix}")
            
            # Separar por tipo e extrair timestamps
            embeddings_files = []
            index_files = []
            
            for blob in blobs:
                filename = os.path.basename(blob.name)
                
                if filename.endswith('.npy'):
                    timestamp = self._extract_timestamp(filename)
                    if timestamp:
                        embeddings_files.append((timestamp, blob.name, blob))
                elif filename.endswith('.faiss'):
                    timestamp = self._extract_timestamp(filename)
                    if timestamp:
                        index_files.append((timestamp, blob.name, blob))
            
            if not embeddings_files or not index_files:
                raise Exception("Arquivos .npy ou .faiss não encontrados")
            
            # Ordenar por timestamp (mais recente primeiro)
            embeddings_files.sort(key=lambda x: x[0], reverse=True)
            index_files.sort(key=lambda x: x[0], reverse=True)
            
            # Encontrar par com mesmo timestamp
            latest_embeddings = embeddings_files[0]
            latest_index = index_files[0]
            
            # Tentar encontrar par com mesmo timestamp
            for emb_timestamp, emb_path, emb_blob in embeddings_files:
                for idx_timestamp, idx_path, idx_blob in index_files:
                    if emb_timestamp == idx_timestamp:
                        latest_embeddings = (emb_timestamp, emb_path, emb_blob)
                        latest_index = (idx_timestamp, idx_path, idx_blob)
                        break
                if latest_embeddings[0] == latest_index[0]:
                    break
            
            timestamp_str = latest_embeddings[0].strftime('%Y%m%d_%H%M%S')
            
            result = {
                'timestamp': timestamp_str,
                'timestamp_dt': latest_embeddings[0],
                'embeddings_path': latest_embeddings[1],
                'index_path': latest_index[1],
                'embeddings_size_mb': latest_embeddings[2].size / (1024 * 1024),
                'index_size_mb': latest_index[2].size / (1024 * 1024),
                'embeddings_filename': os.path.basename(latest_embeddings[1]),
                'index_filename': os.path.basename(latest_index[1])
            }
            
            logger.info(f"✅ Par mais recente encontrado:")
            logger.info(f"   Timestamp: {timestamp_str}")
            logger.info(f"   Embeddings: {result['embeddings_filename']}")
            logger.info(f"   Índice: {result['index_filename']}")
            logger.info(f"   Tamanho: {result['embeddings_size_mb']:.1f}MB + {result['index_size_mb']:.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao buscar embeddings: {e}")
            raise
    
    def get_latest_version(self) -> Optional[str]:
        """Obtém a versão mais recente disponível"""
        try:
            files = self.find_latest_embeddings_pair()
            return files['timestamp']
        except Exception as e:
            logger.error(f"Erro ao obter versão mais recente: {e}")
            return None
    
    def get_embeddings_blob(self, version: str) -> Optional[storage.Blob]:
        """Obtém blob dos embeddings para uma versão"""
        try:
            files = self.find_latest_embeddings_pair()
            return self.bucket.blob(files['embeddings_path'])
        except Exception as e:
            logger.error(f"Erro ao obter embeddings blob: {e}")
            return None
    
    def get_index_blob(self, version: str) -> Optional[storage.Blob]:
        """Obtém blob do índice para uma versão"""
        try:
            files = self.find_latest_embeddings_pair()
            return self.bucket.blob(files['index_path'])
        except Exception as e:
            logger.error(f"Erro ao obter índice blob: {e}")
            return None
    
    def load_latest_embeddings(self) -> bool:
        """
        Carrega os embeddings mais recentes do GCS para memória.
        Retorna True se bem-sucedido.
        """
        try:
            # 1. Encontrar arquivos mais recentes
            files = self.find_latest_embeddings_pair()
            
            logger.info(f"📥 Carregando embeddings: {files['embeddings_filename']}")
            
            # 2. Carregar embeddings (.npy) via streaming
            start_time = datetime.now()
            
            embeddings_blob = self.bucket.blob(files['embeddings_path'])
            embeddings_data = embeddings_blob.download_as_bytes()
            
            # Carregar do buffer de memória
            with io.BytesIO(embeddings_data) as buffer:
                self.current_embeddings = np.load(buffer, allow_pickle=True)
            
            embeddings_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Embeddings carregados: {self.current_embeddings.shape} ({embeddings_time:.2f}s)")
            
            # 3. Carregar índice FAISS (arquivo temporário deletável)
            logger.info(f"📊 Carregando índice: {files['index_filename']}")
            start_time = datetime.now()
            
            # NOVO: Usar diretório temporário específico para Windows
            with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False, dir=self.temp_dir) as tmp_file:
                index_path = tmp_file.name
                
            try:
                index_blob = self.bucket.blob(files['index_path'])
                index_blob.download_to_filename(index_path)
                
                self.current_index = faiss.read_index(index_path)
                logger.info(f"   ✅ Índice carregado do arquivo: {index_path}")
            finally:
                # Limpa o arquivo temporário após carregar
                if os.path.exists(index_path):
                    os.remove(index_path)
                    logger.info(f"   🗑️  Arquivo temporário removido: {index_path}")
            
            index_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Índice carregado: {self.current_index.ntotal} vetores ({index_time:.2f}s)")
            
            # 4. Armazenar metadados
            self.version_info = files
            self.loaded_at = datetime.now()
            
            total_time = embeddings_time + index_time
            logger.info(f"🎉 Embeddings carregados com sucesso em {total_time:.2f} segundos")
            logger.info(f"   Versão: {files['timestamp']}")
            logger.info(f"   Memória: ~{files['embeddings_size_mb']:.1f}MB")
            logger.info(f"   Modo: Consumidor GCS Puro (sem persistência local)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar embeddings: {e}")
            # Tenta limpar em caso de erro
            self.cleanup_temp_files()
            return False
    
    def cleanup_temp_files(self):
        """Limpa os arquivos temporários"""
        try:
            if os.path.exists(self.temp_dir):
                for filename in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logger.debug(f"Arquivo temporário removido: {file_path}")
                    except Exception as e:
                        logger.warning(f"Não foi possível remover {file_path}: {e}")
        except Exception as e:
            logger.error(f"⚠️ Erro ao limpar arquivos temporários: {e}")
    
    def semantic_search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Busca semântica usando embeddings carregados"""
        if self.current_index is None:
            raise ValueError("Índice não carregado")
        
        k = min(k, self.current_index.ntotal)
        distances, indices = self.current_index.search(query_embedding.astype('float32'), k)
        
        return indices[0], distances[0]
    
    def get_embedding_by_index(self, idx: int) -> Optional[np.ndarray]:
        """Obtém embedding por índice"""
        if self.current_embeddings is None or idx >= len(self.current_embeddings):
            return None
        return self.current_embeddings[idx]
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do consumidor"""
        stats = {
            'status': 'loaded' if self.current_index is not None else 'not_loaded',
            'bucket': self.bucket_name,
            'prefix': self.embeddings_prefix,
            'mode': 'gcs_consumer',
            'persistence': 'memory_only',
            'loaded_at': self.loaded_at.isoformat() if self.loaded_at else None
        }
        
        if self.version_info:
            # CORREÇÃO: Verifica se current_embeddings não é None usando 'is not None'
            embeddings_shape = None
            if self.current_embeddings is not None:
                embeddings_shape = self.current_embeddings.shape
            
            stats.update({
                'version': self.version_info['timestamp'],
                'embeddings': {
                    'shape': embeddings_shape,  # Usa a variável corrigida
                    'filename': self.version_info['embeddings_filename'],
                    'size_mb': self.version_info['embeddings_size_mb']
                },
                'index': {
                    'size': self.current_index.ntotal if self.current_index else 0,
                    'filename': self.version_info['index_filename'],
                    'size_mb': self.version_info['index_size_mb']
                }
            })
        
        return stats
    
    def check_for_new_version(self) -> bool:
        """Verifica se há uma versão mais nova no bucket"""
        if not self.version_info:
            return True
        
        try:
            current_timestamp = self.version_info['timestamp_dt']
            files = self.find_latest_embeddings_pair()
            
            if files['timestamp_dt'] > current_timestamp:
                logger.info(f"🔄 Nova versão disponível!")
                logger.info(f"   Atual: {current_timestamp.strftime('%Y%m%d_%H%M%S')}")
                logger.info(f"   Nova: {files['timestamp']}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao verificar nova versão: {e}")
            return False
    
    def reload_if_new_version(self) -> bool:
        """Recarrega apenas se houver versão mais nova"""
        if self.check_for_new_version():
            logger.info("📥 Recarregando nova versão...")
            return self.load_latest_embeddings()
        return False