import io
import numpy as np
import faiss
import logging
from google.cloud import storage
from typing import Tuple, Optional
import tempfile
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class GCSEmbeddingService:
    """Serviço de embeddings que acessa diretamente do GCS (sem download permanente)"""
    
    def __init__(self, bucket_name: str = "book-agent-embeddings-403941621548"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.index = None
        self.embeddings = None
        self.current_files = {}
        
    def _extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """Extrai timestamp do nome do arquivo"""
        try:
            match = re.search(r'(\d{8}_\d{6})', filename)
            if match:
                return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except:
            pass
        return None
    
    def get_latest_files(self) -> Tuple[Optional[str], Optional[str]]:
        """Encontra os arquivos mais recentes no bucket"""
        try:
            logger.info(f"🔍 Procurando arquivos mais recentes no bucket: {self.bucket_name}")
            
            # Listar todos os blobs
            blobs = list(self.client.list_blobs(self.bucket_name))
            
            if not blobs:
                logger.warning("Bucket vazio ou sem acesso")
                return None, None
            
            # Filtrar por tipo e extrair timestamps
            npy_files = []
            faiss_files = []
            
            for blob in blobs:
                if blob.name.endswith('.npy'):
                    timestamp = self._extract_timestamp_from_filename(blob.name)
                    if timestamp:
                        npy_files.append((timestamp, blob.name, blob))
                    else:
                        # Se não tem timestamp, usar data de criação
                        npy_files.append((datetime.min, blob.name, blob))
                        
                elif blob.name.endswith('.faiss'):
                    timestamp = self._extract_timestamp_from_filename(blob.name)
                    if timestamp:
                        faiss_files.append((timestamp, blob.name, blob))
                    else:
                        faiss_files.append((datetime.min, blob.name, blob))
            
            # Ordenar por timestamp (mais recente primeiro)
            npy_files.sort(key=lambda x: x[0], reverse=True)
            faiss_files.sort(key=lambda x: x[0], reverse=True)
            
            latest_npy = npy_files[0][1] if npy_files else None
            latest_faiss = faiss_files[0][1] if faiss_files else None
            
            if latest_npy and latest_faiss:
                logger.info(f"📅 Arquivos mais recentes encontrados:")
                logger.info(f"   Embeddings: {latest_npy}")
                logger.info(f"   Índice: {latest_faiss}")
            else:
                logger.warning("Não foram encontrados arquivos .npy e .faiss no bucket")
            
            return latest_npy, latest_faiss
            
        except Exception as e:
            logger.error(f"Erro ao buscar arquivos no GCS: {e}")
            return None, None
    
    def load_from_gcs(self) -> bool:
        """Carrega embeddings e índice diretamente do GCS (sem salvar localmente)"""
        try:
            # Encontrar arquivos mais recentes
            embeddings_file, index_file = self.get_latest_files()
            
            if not embeddings_file or not index_file:
                logger.error("Arquivos não encontrados no GCS")
                return False
            
            logger.info(f"📥 Carregando embeddings do GCS...")
            logger.info(f"   Embeddings: {embeddings_file}")
            logger.info(f"   Índice: {index_file}")
            
            # 1. Carregar embeddings (.npy) diretamente para memória
            embeddings_blob = self.bucket.blob(embeddings_file)
            embeddings_data = embeddings_blob.download_as_bytes()
            
            # Carregar para numpy a partir dos bytes
            with io.BytesIO(embeddings_data) as f:
                self.embeddings = np.load(f, allow_pickle=True)
            
            logger.info(f"✅ Embeddings carregados: {self.embeddings.shape}")
            
            # 2. Carregar índice FAISS (precisa de arquivo temporário)
            logger.info(f"📊 Carregando índice FAISS...")
            
            # FAISS precisa de arquivo temporário para carregar
            with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                index_blob = self.bucket.blob(index_file)
                index_blob.download_to_filename(tmp_path)
                
                self.index = faiss.read_index(tmp_path)
            
            logger.info(f"✅ Índice carregado: {self.index.ntotal} vetores")
            
            self.current_files = {
                'embeddings': embeddings_file,
                'index': index_file,
                'embeddings_shape': self.embeddings.shape,
                'index_size': self.index.ntotal,
                'bucket': self.bucket_name,
                'loaded_at': datetime.now().isoformat()
            }
            
            logger.info(f"🎉 Embeddings carregados com sucesso do GCS!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar do GCS: {e}")
            # Tentar fallback para arquivos padrão se os com timestamp falharem
            return self._try_fallback_files()
    
    def _try_fallback_files(self) -> bool:
        """Tenta carregar arquivos sem timestamp (fallback)"""
        try:
            logger.info("🔄 Tentando fallback para arquivos sem timestamp...")
            
            # Nomes de arquivos padrão
            standard_files = [
                ("book_index_gpu_embeddings.npy", "book_index_gpu_index.faiss"),
                ("embeddings.npy", "index.faiss"),
                ("book_embeddings.npy", "book_index.faiss")
            ]
            
            for emb_file, idx_file in standard_files:
                try:
                    # Verificar se os arquivos existem
                    emb_blob = self.bucket.blob(emb_file)
                    idx_blob = self.bucket.blob(idx_file)
                    
                    if emb_blob.exists() and idx_blob.exists():
                        logger.info(f"📁 Encontrados arquivos padrão: {emb_file}, {idx_file}")
                        
                        # Carregar embeddings
                        embeddings_data = emb_blob.download_as_bytes()
                        with io.BytesIO(embeddings_data) as f:
                            self.embeddings = np.load(f, allow_pickle=True)
                        
                        # Carregar índice
                        with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                            idx_blob.download_to_filename(tmp_path)
                            self.index = faiss.read_index(tmp_path)
                        
                        self.current_files = {
                            'embeddings': emb_file,
                            'index': idx_file,
                            'embeddings_shape': self.embeddings.shape,
                            'index_size': self.index.ntotal,
                            'bucket': self.bucket_name,
                            'loaded_at': datetime.now().isoformat(),
                            'is_fallback': True
                        }
                        
                        logger.info(f"✅ Fallback carregado: {self.embeddings.shape}")
                        return True
                        
                except Exception as e:
                    logger.debug(f"Fallback falhou para {emb_file}: {e}")
                    continue
            
            logger.error("❌ Nenhum arquivo de fallback encontrado")
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro no fallback: {e}")
            return False
    
    def semantic_search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Busca semântica usando índice carregado"""
        if self.index is None:
            raise ValueError("Índice não carregado")
        
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        logger.debug(f"Busca GCS: {len(indices[0])} resultados")
        return indices[0], distances[0]
    
    def get_embedding_by_index(self, idx: int) -> Optional[np.ndarray]:
        """Obtém embedding por índice"""
        if self.embeddings is None or idx >= len(self.embeddings):
            return None
        return self.embeddings[idx]
    
    def get_stats(self) -> dict:
        """Retorna estatísticas"""
        return {
            'loaded': self.index is not None and self.embeddings is not None,
            'index_size': self.index.ntotal if self.index else 0,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None,
            'current_files': self.current_files,
            'bucket': self.bucket_name,
            'mode': 'gcs_direct'
        }
    
    def refresh_if_needed(self) -> bool:
        """Atualiza se encontrar arquivos mais recentes"""
        try:
            current_emb_file = self.current_files.get('embeddings')
            current_idx_file = self.current_files.get('index')
            
            if not current_emb_file or not current_idx_file:
                return False
            
            # Verificar se há arquivos mais recentes
            latest_emb, latest_idx = self.get_latest_files()
            
            if latest_emb != current_emb_file or latest_idx != current_idx_file:
                logger.info("🔄 Encontrados arquivos mais recentes, atualizando...")
                return self.load_from_gcs()
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao verificar atualizações: {e}")
            return False