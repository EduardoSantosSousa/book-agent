import torch
import numpy as np
import faiss
import pickle
import os
import time
import logging
import tempfile
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import pandas as pd
from config import config, EmbeddingLoader

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Serviço de embeddings - modo consumidor puro do GCS"""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', 
                 use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = None
        self.embedding_model = None
        self.gcs_consumer = None  # Consumidor GCS
        self.index = None
        self.book_embeddings = None
        self.index_built = False
        
    def initialize(self) -> bool:
        """Inicializa apenas como consumidor do GCS"""
        try:
            logger.info(f"Inicializando modelo de embeddings: {self.model_name}")
            
            # 1. Inicializar modelo SentenceTransformer
            if self.use_gpu and torch.cuda.is_available():
                self.device = 'cuda'
                logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
                self.embedding_model = SentenceTransformer(self.model_name, device='cuda')
            else:
                self.device = 'cpu'
                logger.info("Usando CPU")
                self.embedding_model = SentenceTransformer(self.model_name)
            
            logger.info(f"✅ Modelo de embeddings inicializado")
            
            # 2. Inicializar consumidor GCS
            logger.info("🔗 Conectando ao bucket GCS...")
            
            # Importar aqui para evitar dependência circular
            from services.gcs_consumer_service import GCSEmbeddingConsumer
            
            self.gcs_consumer = GCSEmbeddingConsumer(
                bucket_name=config.GCS_BUCKET_NAME,
                embeddings_prefix=config.GCS_EMBEDDINGS_PREFIX
            )
            
            # 3. Carregar embeddings mais recentes
            if not self.gcs_consumer.load_latest_embeddings():
                logger.error("❌ Falha ao carregar embeddings do GCS")
                return False
            
            # 4. Para compatibilidade com código existente
            self.index = self.gcs_consumer.current_index
            self.book_embeddings = self.gcs_consumer.current_embeddings
            self.index_built = True
            
            # 5. Log de sucesso
            stats = self.gcs_consumer.get_stats()
            logger.info(f"🎉 Sistema inicializado como consumidor GCS")
            logger.info(f"   Versão: {stats.get('version', 'N/A')}")
            logger.info(f"   Embeddings: {stats.get('embeddings', {}).get('shape', 'N/A')}")
            logger.info(f"   Índice: {stats.get('index', {}).get('size', 0)} vetores")
            logger.info(f"   Bucket: {config.GCS_BUCKET_NAME}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar EmbeddingService: {e}")
            return False
    
    def semantic_search(self, query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Busca semântica usando embeddings do GCS"""
        if not self.index_built or not self.gcs_consumer:
            logger.warning("Índice não carregado")
            return np.array([]), np.array([])
        
        try:
            # 1. Gerar embedding da query
            query_embedding = self.embedding_model.encode(
                [query], 
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # 2. Buscar usando consumidor GCS
            indices, distances = self.gcs_consumer.semantic_search(query_embedding, k)
            
            logger.debug(f"Busca: '{query[:50]}...' -> {len(indices)} resultados")
            return indices, distances
            
        except Exception as e:
            logger.error(f"Erro na busca semântica: {e}")
            return np.array([]), np.array([])
    
    def get_embedding_by_index(self, idx: int) -> Optional[np.ndarray]:
        """Obtém embedding por índice"""
        if self.gcs_consumer:
            return self.gcs_consumer.get_embedding_by_index(idx)
        return None
    
    def get_stats(self) -> dict:
        """Retorna estatísticas"""
        if self.gcs_consumer:
            stats = self.gcs_consumer.get_stats()
            stats['embedding_model'] = self.model_name
            stats['device'] = self.device
            return stats
        else:
            return {
                'status': 'not_initialized',
                'model': self.model_name,
                'device': self.device
            }

    def prepare_texts_batch(self, data: pd.DataFrame, batch_size: int = 128) -> List[str]:
        """Método mantido para compatibilidade"""
        logger.warning("⚠️  Este é um consumidor puro - método prepare_texts_batch não faz nada")
        return []
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Método mantido para compatibilidade"""
        logger.warning("⚠️  Este é um consumidor puro - não gera embeddings")
        return np.array([])
    
    
    
    def is_initialized(self) -> bool:
        """Verifica se está inicializado"""
        return self.index_built and self.gcs_consumer is not None    

    def get_index_stats(self) -> dict:
        """Retorna estatísticas do índice - MANTENDO PARA COMPATIBILIDADE"""
        if not self.index_built and not self.gcs_service:
            return {
                'index_built': False,
                'message': 'Índice não carregado'
            }
        
        if self.gcs_service:
            return {
                'index_built': True,
                'index_size': self.gcs_service.index.ntotal if self.gcs_service.index else 0,
                'embeddings_shape': self.gcs_service.embeddings.shape if self.gcs_service.embeddings is not None else None,
                'model_name': self.model_name,
                'device': self.device,
                'use_gpu': self.use_gpu,
                'mode': 'gcs_direct'
            }
        else:
            return {
                'index_built': True,
                'index_size': self.index.ntotal if self.index else 0,
                'embeddings_shape': self.book_embeddings.shape if self.book_embeddings is not None else None,
                'model_name': self.model_name,
                'device': self.device,
                'use_gpu': self.use_gpu,
                'mode': 'traditional'
            }

    def search_similar_books(self, book_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """Busca livros similares a um livro específico"""
        if (not self.index_built and not self.gcs_service) or (self.book_embeddings is None and not self.gcs_service):
            return []
        
        try:
            if self.gcs_service:
                # Modo GCS direto
                if book_id < 0 or book_id >= len(self.gcs_service.embeddings):
                    logger.warning(f"book_id {book_id} fora do range")
                    return []
                
                book_embedding = self.gcs_service.embeddings[book_id:book_id+1]
                distances, indices = self.gcs_service.index.search(book_embedding.astype('float32'), k + 1)
            else:
                # Modo tradicional
                if book_id < 0 or book_id >= len(self.book_embeddings):
                    logger.warning(f"book_id {book_id} fora do range (0-{len(self.book_embeddings)-1})")
                    return []
                
                book_embedding = self.book_embeddings[book_id:book_id+1]
                distances, indices = self.index.search(book_embedding.astype('float32'), k + 1)
            
            # Remover o próprio livro dos resultados
            results = []
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if idx == book_id:  # Pular o próprio livro
                    continue
                if idx != -1:  # -1 indica resultado inválido no FAISS
                    similarity = 1.0 / (1.0 + dist) if dist > 0 else 1.0
                    results.append((int(idx), float(similarity)))
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Erro ao buscar livros similares: {e}")
            return []

    def load_existing_index(self, index_path: str = None, embeddings_path: str = None) -> bool:
        """Carrega um índice já existente de caminhos específicos"""
        try:
            if index_path is None or embeddings_path is None:
                # Usar caminhos padrão
                index_path = 'embeddings/local/book_index_gpu_index.faiss'
                embeddings_path = 'embeddings/local/book_index_gpu_embeddings.npy'
            
            logger.info(f"Carregando índice de {index_path}")
            
            if not os.path.exists(index_path):
                logger.error(f"Arquivo de índice não encontrado: {index_path}")
                return False
            
            if not os.path.exists(embeddings_path):
                logger.error(f"Arquivo de embeddings não encontrado: {embeddings_path}")
                return False
            
            self.index = faiss.read_index(index_path)
            self.book_embeddings = np.load(embeddings_path)
            self.index_built = True
            
            logger.info(f"✅ Índice carregado: {self.book_embeddings.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar índice existente: {e}")
            return False

    def encode_query(self, query: str) -> np.ndarray:
        """Codifica uma query para embedding"""
        try:
            if self.embedding_model is None:
                raise ValueError("Modelo de embeddings não inicializado")
            
            embedding = self.embedding_model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embedding[0]  # Retorna apenas o vetor, não a lista
            
        except Exception as e:
            logger.error(f"Erro ao codificar query: {e}")
            return np.array([])

    def is_initialized(self) -> bool:
        """Verifica se o serviço está inicializado"""
        return self.embedding_model is not None and (self.index_built or self.gcs_service is not None)

    def get_embedding_dimension(self) -> int:
        """Retorna a dimensão dos embeddings"""
        if self.gcs_service and self.gcs_service.embeddings is not None:
            return self.gcs_service.embeddings.shape[1]
        elif self.book_embeddings is not None:
            return self.book_embeddings.shape[1]
        elif self.embedding_model is not None:
            # Tenta obter a dimensão do modelo
            try:
                # Cria um embedding dummy para obter a dimensão
                dummy_embedding = self.embedding_model.encode(
                    ["test"],
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return dummy_embedding.shape[1]
            except:
                return 384  # Dimensão padrão do MiniLM
        else:
            return 0