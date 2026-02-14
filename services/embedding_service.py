import torch
import numpy as np
import faiss
import os
import logging
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import pandas as pd
from config import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Serviço de embeddings - modo consumidor puro do GCS"""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', 
                 use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = None
        self.embedding_model = None
        self.gcs_consumer = None
        self.index = None
        self.book_embeddings = None
        self.index_built = False
        
    def initialize(self) -> bool:
        """Inicializa como consumidor do GCS com embeddings, índice e metadados"""
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
            
            # Import CORRETO - usando a classe GCSEmbeddingService
            from services.gcs_embedding_service import GCSEmbeddingService
            
            self.gcs_consumer = GCSEmbeddingService(
                bucket_name=config.GCS_BUCKET_NAME
            )
            
            # 3. Carregar embeddings, índice E METADADOS mais recentes
            if not self.gcs_consumer.load_latest_embeddings_with_metadata():
                logger.error("❌ Falha ao carregar embeddings do GCS")
                return False
            
            # 4. Para compatibilidade com código existente
            self.index = self.gcs_consumer.index
            self.book_embeddings = self.gcs_consumer.embeddings
            self.index_built = True
            
            # 5. Log de sucesso com estatísticas completas
            stats = self.gcs_consumer.get_stats()
            logger.info(f"🎉 Sistema inicializado como consumidor GCS")
            logger.info(f"   Embeddings: {stats.get('embeddings_shape', 'N/A')}")
            logger.info(f"   Índice: {stats.get('index_size', 0)} vetores")
            if stats.get('metadata_loaded'):
                logger.info(f"   Metadados: {stats.get('metadata_count', 0)} registros")
                logger.info(f"   Book IDs mapeados: {stats.get('book_id_mapping_count', 0)}")
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
            query_embedding = self.embedding_model.encode(
                [query], 
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
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
        """Retorna estatísticas completas"""
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
        logger.warning("⚠️ Este é um consumidor puro - método prepare_texts_batch não faz nada")
        return []
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Método mantido para compatibilidade"""
        logger.warning("⚠️ Este é um consumidor puro - não gera embeddings")
        return np.array([])
    
    def is_initialized(self) -> bool:
        """Verifica se está inicializado"""
        return self.index_built and self.gcs_consumer is not None
    
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
            
            return embedding[0]
            
        except Exception as e:
            logger.error(f"Erro ao codificar query: {e}")
            return np.array([])
    
    def get_embedding_dimension(self) -> int:
        """Retorna a dimensão dos embeddings"""
        if self.gcs_consumer and self.gcs_consumer.embeddings is not None:
            return self.gcs_consumer.embeddings.shape[1]
        elif self.book_embeddings is not None:
            return self.book_embeddings.shape[1]
        elif self.embedding_model is not None:
            try:
                dummy_embedding = self.embedding_model.encode(
                    ["test"],
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return dummy_embedding.shape[1]
            except:
                return 384
        else:
            return 0

    # ============= NOVOS MÉTODOS PARA VERIFICAÇÃO DE COBERTURA =============
    
    def verificar_livros_sem_embedding(self, csv_path: str = None) -> dict:
        """
        Verifica quantos livros da base NÃO têm embedding.
        
        Args:
            csv_path: Caminho no GCS para o arquivo CSV 
                     (ex: "exports/20260119_231738_EDU_books.csv")
        
        Returns:
            Dict com estatísticas de cobertura
        """
        if not self.gcs_consumer:
            logger.error("❌ GCS Consumer não inicializado")
            return None
        
        # Se não forneceu caminho, tenta inferir do nome dos embeddings
        if csv_path is None:
            embeddings_file = self.gcs_consumer.current_files.get('embeddings', '')
            
            # Extrai timestamp do formato: YYYYMMDD_HHMMSS_..._embeddings.npy
            import re
            match = re.search(r'(\d{8}_\d{6})', embeddings_file)
            if match:
                timestamp = match.group(1)
                csv_path = f"exports/{timestamp}_EDU_books.csv"
                logger.info(f"📄 Caminho do CSV inferido: {csv_path}")
            else:
                # Fallback para o padrão que você mostrou
                csv_path = "exports/20260119_231738_EDU_books.csv"
                logger.warning(f"⚠️ Usando caminho padrão: {csv_path}")
        
        # Chama o método de verificação
        return self.gcs_consumer.verificar_cobertura_com_metadados(csv_path)
    
    def get_book_id_by_index(self, idx: int) -> Optional[str]:
        """Retorna o book_id para um determinado índice do embedding"""
        if self.gcs_consumer:
            return self.gcs_consumer.get_book_id_by_index(idx)
        return None
    
    def get_index_by_book_id(self, book_id: str) -> Optional[int]:
        """Retorna o índice do embedding para um determinado book_id"""
        if self.gcs_consumer:
            return self.gcs_consumer.get_index_by_book_id(book_id)
        return None
    
    def listar_livros_sem_embedding(self, csv_path: str = None, max_ids: int = 100) -> List[str]:
        """
        Retorna a lista de IDs dos livros sem embedding.
        
        Args:
            csv_path: Caminho do CSV no GCS
            max_ids: Número máximo de IDs para retornar
        
        Returns:
            Lista com os IDs dos livros sem embedding
        """
        resultado = self.verificar_livros_sem_embedding(csv_path)
        if resultado:
            return resultado.get('ids_sem_embedding', [])[:max_ids]
        return []
    
    def gerar_relatorio_cobertura(self, csv_path: str = None) -> str:
        """
        Gera um relatório formatado sobre a cobertura de embeddings.
        
        Returns:
            String com relatório formatado
        """
        resultado = self.verificar_livros_sem_embedding(csv_path)
        
        if not resultado:
            return "❌ Não foi possível gerar relatório de cobertura."
        
        relatorio = f"""
╔══════════════════════════════════════════════════════════════╗
║              RELATÓRIO DE COBERTURA DE EMBEDDINGS            ║
╠══════════════════════════════════════════════════════════════╣
║  📊 Estatísticas Gerais:                                     ║
║     • Total de livros na base: {resultado['total_livros_csv']:>10}        ║
║     • Livros COM embedding:   {resultado['total_com_embedding']:>10}        ║
║     • Livros SEM embedding:   {resultado['total_sem_embedding']:>10}        ║
║     • Cobertura:              {resultado['cobertura_percentual']:>9.2f}%     ║
║                                                              ║
║  🕒 Timestamp dos embeddings:                               ║
║     {resultado.get('timestamp', 'N/A')[:50]}║
║                                                              ║
║  📋 Primeiros 10 IDs sem embedding:                         ║
"""
        
        # Adiciona os primeiros 10 IDs sem embedding
        ids_amostra = resultado.get('ids_sem_embedding', [])[:10]
        for i, book_id in enumerate(ids_amostra, 1):
            relatorio += f"║     {i:2d}. {book_id:<45} ║\n"
        
        if resultado['total_sem_embedding'] > 10:
            relatorio += f"║     ... e mais {resultado['total_sem_embedding'] - 10} IDs       ║\n"
        
        relatorio += "╚══════════════════════════════════════════════════════════════╝"
        
        return relatorio

    # ============= MÉTODOS DE COMPATIBILIDADE (NÃO REMOVER) =============
    
    def search_similar_books(self, book_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """Busca livros similares a um livro específico (MODO LEGADO)"""
        logger.warning("⚠️ search_similar_books está obsoleto - use semantic_search com o book_id")
        
        # Tenta converter book_id para índice se for string
        if isinstance(book_id, str):
            idx = self.get_index_by_book_id(book_id)
            if idx is None:
                logger.warning(f"Book ID {book_id} não encontrado nos metadados")
                return []
            book_id = idx
        
        if (not self.index_built) or (self.book_embeddings is None):
            return []
        
        try:
            if book_id < 0 or book_id >= len(self.book_embeddings):
                logger.warning(f"book_id {book_id} fora do range")
                return []
            
            book_embedding = self.book_embeddings[book_id:book_id+1]
            distances, indices = self.index.search(book_embedding.astype('float32'), k + 1)
            
            results = []
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if idx == book_id:
                    continue
                if idx != -1:
                    similarity = 1.0 / (1.0 + dist) if dist > 0 else 1.0
                    results.append((int(idx), float(similarity)))
            
            return results[:k]
            
        except Exception as e:
            logger.error(f"Erro ao buscar livros similares: {e}")
            return []
    
    def load_existing_index(self, index_path: str = None, embeddings_path: str = None) -> bool:
        """Método mantido para compatibilidade"""
        logger.warning("⚠️ load_existing_index está obsoleto - use initialize() com GCS")
        return False
    
    def get_index_stats(self) -> dict:
        """Retorna estatísticas do índice - compatibilidade"""
        return self.get_stats()
