import io
import numpy as np
import faiss
import logging
from google.cloud import storage
from typing import Tuple, Optional
import tempfile
import re
from datetime import datetime
import pandas as pd
import json

logger = logging.getLogger(__name__)

class GCSEmbeddingService:
    """Serviço de embeddings que acessa diretamente do GCS (sem download permanente)"""
    
    def __init__(self, bucket_name: str = "book-agent-embeddings-bucket"):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.index = None
        self.embeddings = None
        self.current_files = {}
        self.metadata = None
        self.book_id_to_index = {}
        
        # Contadores para estatísticas de processamento
        self.stats = {
            'total_embeddings_carregados': 0,
            'total_metadados_carregados': 0,
            'total_book_ids_mapeados': 0,
            'ultima_verificacao': None,
            'total_verificacoes_realizadas': 0,
            'total_csv_processados': 0,
            'ultimo_csv_processado': None
        }
        
    def _extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """Extrai timestamp do nome do arquivo"""
        try:
            match = re.search(r'(\d{8}_\d{6})', filename)
            if match:
                return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except Exception as e:
            logger.debug(f"Erro ao extrair timestamp de {filename}: {e}")
        return None
    
    def get_latest_files(self) -> Tuple[Optional[str], Optional[str]]:
        """Encontra os arquivos mais recentes no bucket"""
        try:
            logger.info(f"🔍 [GCS] Procurando arquivos mais recentes no bucket: {self.bucket_name}")
            
            blobs = list(self.client.list_blobs(self.bucket_name))
            logger.info(f"📁 [GCS] Total de arquivos encontrados no bucket: {len(blobs)}")
            
            # LISTAR TODOS OS ARQUIVOS PARA DEBUG
            logger.info("📋 [GCS] Lista completa de arquivos no bucket:")
            for blob in blobs:
                logger.info(f"   - {blob.name}")
            
            if not blobs:
                logger.warning("⚠️ [GCS] Bucket vazio ou sem acesso")
                return None, None
            
            npy_files = []
            faiss_files = []
            metadata_files = []  # Adicionado para debug
            
            for blob in blobs:
                if blob.name.endswith('.npy'):
                    timestamp = self._extract_timestamp_from_filename(blob.name)
                    npy_files.append((timestamp or datetime.min, blob.name))
                    logger.debug(f"   📄 Encontrado .npy: {blob.name} (timestamp: {timestamp})")
                    
                elif blob.name.endswith('.faiss'):
                    timestamp = self._extract_timestamp_from_filename(blob.name)
                    faiss_files.append((timestamp or datetime.min, blob.name))
                    logger.debug(f"   📊 Encontrado .faiss: {blob.name} (timestamp: {timestamp})")
                    
                elif blob.name.endswith('.json'):
                    timestamp = self._extract_timestamp_from_filename(blob.name)
                    metadata_files.append((timestamp or datetime.min, blob.name))
                    logger.debug(f"   📋 Encontrado .json: {blob.name} (timestamp: {timestamp})")
            
            npy_files.sort(key=lambda x: x[0], reverse=True)
            faiss_files.sort(key=lambda x: x[0], reverse=True)
            metadata_files.sort(key=lambda x: x[0], reverse=True)
            
            # DEBUG: Mostrar arquivos de metadados encontrados
            if metadata_files:
                logger.info(f"📋 [GCS] Arquivos de metadados (.json) encontrados:")
                for ts, name in metadata_files[:5]:  # Mostrar apenas os 5 mais recentes
                    logger.info(f"   - {name} (timestamp: {ts})")
            else:
                logger.warning("⚠️ [GCS] NENHUM arquivo .json encontrado no bucket!")
            
            latest_npy = npy_files[0][1] if npy_files else None
            latest_faiss = faiss_files[0][1] if faiss_files else None
            
            if latest_npy and latest_faiss:
                logger.info(f"✅ [GCS] Arquivos mais recentes selecionados:")
                logger.info(f"   📄 Embeddings: {latest_npy}")
                logger.info(f"   📊 Índice: {latest_faiss}")
                logger.info(f"   🕒 Timestamp: {npy_files[0][0] if npy_files else 'N/A'}")
            else:
                logger.warning("⚠️ [GCS] Não foram encontrados pares de arquivos .npy e .faiss")
                if not latest_npy:
                    logger.warning("   ❌ Nenhum arquivo .npy encontrado")
                if not latest_faiss:
                    logger.warning("   ❌ Nenhum arquivo .faiss encontrado")
            
            return latest_npy, latest_faiss
            
        except Exception as e:
            logger.error(f"❌ [GCS] Erro ao buscar arquivos: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def load_from_gcs(self) -> bool:
        """Carrega embeddings e índice diretamente do GCS"""
        try:
            logger.info("=" * 80)
            logger.info("🚀 INICIANDO CARREGAMENTO DE EMBEDDINGS DO GCS")
            logger.info("=" * 80)
            
            embeddings_file, index_file = self.get_latest_files()
            
            if not embeddings_file or not index_file:
                logger.error("❌ [LOAD] Arquivos não encontrados no GCS")
                return False
            
            logger.info(f"📥 [LOAD] Carregando embeddings: {embeddings_file}")
            
            # Carregar embeddings
            embeddings_blob = self.bucket.blob(embeddings_file)
            embeddings_data = embeddings_blob.download_as_bytes()
            logger.info(f"   📦 Tamanho do arquivo: {len(embeddings_data) / 1024 / 1024:.2f} MB")
            
            with io.BytesIO(embeddings_data) as f:
                self.embeddings = np.load(f, allow_pickle=True)
            
            self.stats['total_embeddings_carregados'] = self.embeddings.shape[0]
            logger.info(f"✅ [LOAD] Embeddings carregados com sucesso!")
            logger.info(f"   📊 Shape: {self.embeddings.shape}")
            logger.info(f"   📈 Total de vetores: {self.embeddings.shape[0]}")
            logger.info(f"   📐 Dimensão: {self.embeddings.shape[1]}")
            logger.info(f"   💾 Memória: {self.embeddings.nbytes / 1024 / 1024:.2f} MB")
            
            # Carregar índice FAISS
            logger.info(f"📊 [LOAD] Carregando índice FAISS: {index_file}")
            
            with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                index_blob = self.bucket.blob(index_file)
                index_blob.download_to_filename(tmp_path)
                logger.info(f"   📦 Arquivo temporário criado: {tmp_path}")
                
                self.index = faiss.read_index(tmp_path)
            
            logger.info(f"✅ [LOAD] Índice FAISS carregado com sucesso!")
            logger.info(f"   📊 Total de vetores no índice: {self.index.ntotal}")
            logger.info(f"   🏷️  Tipo do índice: {type(self.index).__name__}")
            
            self.current_files = {
                'embeddings': embeddings_file,
                'index': index_file,
                'embeddings_shape': self.embeddings.shape,
                'index_size': self.index.ntotal,
                'bucket': self.bucket_name,
                'loaded_at': datetime.now().isoformat()
            }
            
            logger.info("=" * 80)
            logger.info("🎉 EMBEDDINGS CARREGADOS COM SUCESSO DO GCS!")
            logger.info(f"   🕒 Carregado em: {self.current_files['loaded_at']}")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [LOAD] Erro ao carregar do GCS: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._try_fallback_files()
    
    def _try_fallback_files(self) -> bool:
        """Tenta carregar arquivos sem timestamp (fallback)"""
        try:
            logger.warning("🔄 [FALLBACK] Tentando carregar arquivos padrão...")
            
            standard_files = [
                ("book_index_gpu_embeddings.npy", "book_index_gpu_index.faiss"),
                ("embeddings.npy", "index.faiss"),
                ("book_embeddings.npy", "book_index.faiss")
            ]
            
            for emb_file, idx_file in standard_files:
                try:
                    logger.info(f"   🔍 Tentando: {emb_file} + {idx_file}")
                    
                    emb_blob = self.bucket.blob(emb_file)
                    idx_blob = self.bucket.blob(idx_file)
                    
                    if emb_blob.exists() and idx_blob.exists():
                        logger.info(f"   ✅ Arquivos encontrados!")
                        
                        # Carregar embeddings
                        embeddings_data = emb_blob.download_as_bytes()
                        with io.BytesIO(embeddings_data) as f:
                            self.embeddings = np.load(f, allow_pickle=True)
                        
                        # Carregar índice
                        with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp_file:
                            tmp_path = tmp_file.name
                            idx_blob.download_to_filename(tmp_path)
                            self.index = faiss.read_index(tmp_path)
                        
                        logger.info(f"   ✅ Fallback carregado: {self.embeddings.shape}")
                        return True
                        
                except Exception as e:
                    logger.debug(f"   ❌ Fallback falhou para {emb_file}: {e}")
                    continue
            
            logger.error("❌ [FALLBACK] Nenhum arquivo de fallback encontrado")
            return False
            
        except Exception as e:
            logger.error(f"❌ [FALLBACK] Erro no fallback: {e}")
            return False
    
    def load_latest_embeddings_with_metadata(self) -> bool:
        """
        Carrega embeddings, índice E TENTA carregar metadados (se existirem)
        """
        # 1. Carregar embeddings e índice
        if not self.load_from_gcs():
            return False
        
        # 2. TENTAR carregar metadados - mas NÃO falhar se não existir
        metadata_loaded = self.load_metadata()
        
        if metadata_loaded:
            logger.info("✅ Metadados carregados com sucesso!")
        else:
            logger.warning("⚠️ Metadados NÃO foram carregados - verificação de cobertura NÃO disponível")
            logger.warning("   Isso NÃO afeta a busca semântica, apenas a verificação de cobertura")
        
        return True
    
    def load_metadata(self, metadata_filename: str = None) -> bool:
        """
        Carrega o arquivo de metadados que mapeia índices FAISS para IDs dos livros.
        CORRIGIDO: Agora encontra o metadata.json correto no bucket!
        """
        try:
            if metadata_filename is None:
                # Tenta encontrar o metadata baseado no timestamp dos embeddings
                latest_emb, _ = self.get_latest_files()
                if latest_emb:
                    # EXTRAIR O TIMESTAMP do nome do arquivo
                    match = re.search(r'(\d{8}_\d{6})', latest_emb)
                    if match:
                        timestamp = match.group(1)
                        
                        # LISTA DE POSSÍVEIS CAMINHOS - PRIORIDADE CORRETA!
                        possible_names = [
                            # PRIORIDADE 1: Mesmo diretório que os embeddings (SEU ARQUIVO EXISTE AQUI!)
                            f"embeddings/{timestamp}_EDU_books_metadata.json",
                            
                            # PRIORIDADE 2: Sem o timestamp extra (como está no seu bucket)
                            f"embeddings/{timestamp}_EDU_books_metadata.json",
                            
                            # PRIORIDADE 3: Com o nome completo (fallback)
                            latest_emb.replace('_embeddings.npy', '_metadata.json'),
                            
                            # PRIORIDADE 4: Apenas o nome base
                            f"{timestamp}_EDU_books_metadata.json",
                            
                            # PRIORIDADE 5: Na pasta exports
                            f"exports/{timestamp}_EDU_books_metadata.json",
                        ]
                        
                        # Remover duplicatas mantendo a ordem
                        possible_names = list(dict.fromkeys(possible_names))
                        
                        logger.info(f"🔍 [METADATA] Procurando arquivo de metadados...")
                        logger.info(f"   📌 Timestamp extraído: {timestamp}")
                        
                        for test_name in possible_names:
                            try:
                                logger.info(f"   🔎 Verificando: {test_name}")
                                metadata_blob = self.bucket.blob(test_name)
                                
                                if metadata_blob.exists():
                                    logger.info(f"   ✅ METADADOS ENCONTRADOS: {test_name}")
                                    metadata_filename = test_name
                                    break
                                else:
                                    logger.debug(f"   ❌ Não encontrado: {test_name}")
                            except Exception as e:
                                logger.debug(f"   ⚠️ Erro ao verificar {test_name}: {e}")
                    else:
                        logger.error(f"❌ [METADATA] Não foi possível extrair timestamp de: {latest_emb}")
            
            if not metadata_filename:
                # ÚLTIMA TENTATIVA: Procurar qualquer arquivo metadata.json no bucket
                logger.info("🔍 [METADATA] Buscando qualquer arquivo metadata.json no bucket...")
                blobs = list(self.client.list_blobs(self.bucket_name))
                metadata_blobs = [b for b in blobs if b.name.endswith('metadata.json')]
                
                if metadata_blobs:
                    # Pega o mais recente
                    metadata_blobs.sort(key=lambda b: b.updated or datetime.min, reverse=True)
                    metadata_filename = metadata_blobs[0].name
                    logger.info(f"   ✅ Encontrado metadata.json alternativo: {metadata_filename}")
                else:
                    logger.warning("❌ [METADATA] Nenhum arquivo metadata.json encontrado no bucket")
                    return False
            
            logger.info(f"📚 [METADATA] Carregando metadados: {metadata_filename}")
            
            # Download do metadata.json do GCS
            metadata_blob = self.bucket.blob(metadata_filename)
            metadata_data = metadata_blob.download_as_string()
            
            self.metadata = json.loads(metadata_data)
            
            logger.info(f"✅ [METADATA] Metadados carregados: {len(self.metadata)} registros")
            
            # Verificar a estrutura dos metadados
            if len(self.metadata) > 0:
                logger.info(f"   📋 Estrutura do primeiro registro: {list(self.metadata[0].keys())}")
            
            # Criar um índice reverso para busca por book_id
            self.book_id_to_index = {}
            registros_com_id = 0
            registros_sem_id = 0
            
            for idx, meta in enumerate(self.metadata):
                # Tentar diferentes campos de ID
                book_id = None
                for id_field in ['book_id', 'id', 'livro_id', 'codigo', 'isbn']:
                    if id_field in meta and meta[id_field]:
                        book_id = str(meta[id_field])
                        break
                
                if book_id:
                    self.book_id_to_index[book_id] = idx
                    registros_com_id += 1
                else:
                    registros_sem_id += 1
            
            self.stats['total_metadados_carregados'] = len(self.metadata)
            self.stats['total_book_ids_mapeados'] = registros_com_id
            self.stats['total_registros_sem_id'] = registros_sem_id
            
            logger.info(f"🔗 [METADATA] Mapeamento criado:")
            logger.info(f"   ✅ Registros com ID: {registros_com_id}")
            logger.info(f"   ⚠️ Registros sem ID: {registros_sem_id}")
            logger.info(f"   📊 Total book_ids mapeados: {len(self.book_id_to_index)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [METADATA] Erro ao carregar metadados: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.metadata = None
            self.book_id_to_index = {}
            return False
    
    def verificar_cobertura_com_metadados(self, csv_gcs_path: str) -> dict:
        """
        VERIFICAÇÃO CORRETA: Usa os metadados para comparar IDs reais dos livros.
        CORRIGIDO: Agora usa o caminho correto do CSV!
        """
        logger.info("=" * 80)
        logger.info("🔍 INICIANDO VERIFICAÇÃO DE COBERTURA DE EMBEDDINGS")
        logger.info("=" * 80)
        
        # Se o caminho do CSV for None ou vazio, usa o padrão
        if not csv_gcs_path:
            # Extrai timestamp do nome dos embeddings
            embeddings_file = self.current_files.get('embeddings', '')
            match = re.search(r'(\d{8}_\d{6})', embeddings_file)
            if match:
                timestamp = match.group(1)
                csv_gcs_path = f"exports/{timestamp}_EDU_books.csv"
                logger.info(f"📄 [COBERTURA] Caminho do CSV inferido: {csv_gcs_path}")
    
        # Verificar se metadados existem
        if not hasattr(self, 'metadata') or not self.metadata:
            logger.warning("⚠️ [COBERTURA] Metadados não carregados. Tentando carregar...")
            if not self.load_metadata():
                logger.error("❌ [COBERTURA] NÃO É POSSÍVEL VERIFICAR COBERTURA - Metadados não disponíveis")
                logger.error("   Para resolver, faça upload do arquivo metadata.json para o bucket")
                return None
            
        # 1. Carregar CSV de livros do GCS
        try:
            logger.info(f"📄 [COBERTURA] Carregando CSV: {csv_gcs_path}")
            
            csv_blob = self.bucket.blob(csv_gcs_path)
            
            # Verificar se o CSV existe
            if not csv_blob.exists():
                logger.error(f"❌ [COBERTURA] CSV não encontrado: {csv_gcs_path}")
                logger.error(f"   Arquivos disponíveis no bucket:")
                for blob in list(self.client.list_blobs(self.bucket_name))[:10]:  # Mostrar primeiros 10
                    logger.error(f"   - {blob.name}")
                return None
            
            csv_data = csv_blob.download_as_string()
            logger.info(f"   📦 Tamanho do CSV: {len(csv_data) / 1024:.2f} KB")
            
            df_books = pd.read_csv(io.BytesIO(csv_data))
            logger.info(f"   📊 CSV carregado: {len(df_books)} linhas, {len(df_books.columns)} colunas")
            logger.info(f"   🏷️  Colunas disponíveis: {list(df_books.columns)}")
            
            # Identificar coluna de ID do livro
            id_column = None
            for col in ['id', 'book_id', 'livro_id', 'codigo']:
                if col in df_books.columns:
                    id_column = col
                    break
            
            if not id_column:
                logger.error("❌ [COBERTURA] Não foi possível identificar coluna de ID no CSV")
                return None
            
            logger.info(f"   ✅ Coluna de ID identificada: '{id_column}'")
            
            # Mostrar amostra dos IDs
            ids_amostra = df_books[id_column].astype(str).head(5).tolist()
            logger.info(f"   📋 Amostra de IDs do CSV: {ids_amostra}")
            
            todos_ids_csv = set(df_books[id_column].astype(str).values)
            logger.info(f"   📊 Total de IDs únicos no CSV: {len(todos_ids_csv)}")
            
        except Exception as e:
            logger.error(f"❌ [COBERTURA] Erro ao carregar CSV: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
        # 2. IDs que estão nos metadados (têm embedding)
        ids_com_embedding = set(self.book_id_to_index.keys())
        logger.info(f"   📊 Total de IDs com embedding: {len(ids_com_embedding)}")
        
        # Mostrar amostra dos IDs com embedding
        ids_embedding_amostra = list(ids_com_embedding)[:5]
        logger.info(f"   📋 Amostra de IDs COM embedding: {ids_embedding_amostra}")
        
        # 3. IDs sem embedding
        ids_sem_embedding = todos_ids_csv - ids_com_embedding
        logger.info(f"   📊 Total de IDs SEM embedding: {len(ids_sem_embedding)}")
        
        # Mostrar amostra dos IDs sem embedding
        ids_sem_amostra = list(ids_sem_embedding)[:5]
        logger.info(f"   📋 Amostra de IDs SEM embedding: {ids_sem_amostra}")
        
        # 4. Estatísticas
        stats = {
            'total_livros_csv': len(todos_ids_csv),
            'total_com_embedding': len(ids_com_embedding),
            'total_sem_embedding': len(ids_sem_embedding),
            'cobertura_percentual': (len(ids_com_embedding) / len(todos_ids_csv)) * 100 if todos_ids_csv else 0,
            'ids_sem_embedding': sorted(list(ids_sem_embedding))[:50],
            'timestamp': self.current_files.get('embeddings', 'N/A'),
            'csv_path': csv_gcs_path,
            'id_column': id_column
        }
        
        # 5. LOG DETALHADO DOS RESULTADOS
        logger.info("=" * 80)
        logger.info("📊 RESULTADO DA VERIFICAÇÃO DE COBERTURA:")
        logger.info("=" * 80)
        logger.info(f"   📁 CSV processado: {csv_gcs_path}")
        logger.info(f"   🆔 Coluna de ID: {id_column}")
        logger.info(f"   📚 Total de livros na base: {stats['total_livros_csv']}")
        logger.info(f"   ✅ Livros COM embedding: {stats['total_com_embedding']}")
        logger.info(f"   ❌ Livros SEM embedding: {stats['total_sem_embedding']}")
        logger.info(f"   📈 Cobertura: {stats['cobertura_percentual']:.2f}%")
        logger.info(f"   🕒 Timestamp embeddings: {stats['timestamp']}")
        logger.info("=" * 80)
        
        if stats['total_sem_embedding'] > 0:
            logger.warning(f"⚠️ ATENÇÃO: {stats['total_sem_embedding']} livros SEM embedding!")
            logger.warning(f"   Primeiros 10 IDs: {sorted(list(ids_sem_embedding))[:10]}")
        else:
            logger.info("🎉 PARABÉNS! Todos os livros têm embedding!")
        
        # Atualizar estatísticas do serviço
        self.stats['ultima_verificacao'] = datetime.now().isoformat()
        self.stats['total_verificacoes_realizadas'] += 1
        self.stats['total_csv_processados'] += 1
        self.stats['ultimo_csv_processado'] = csv_gcs_path
        
        return stats
    
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
        stats = {
            'loaded': self.index is not None and self.embeddings is not None,
            'index_size': self.index.ntotal if self.index else 0,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None,
            'current_files': self.current_files,
            'bucket': self.bucket_name,
            'mode': 'gcs_direct'
        }
        
        # Adicionar info de metadados se disponível
        if hasattr(self, 'metadata') and self.metadata:
            stats['metadata_loaded'] = True
            stats['metadata_count'] = len(self.metadata)
            stats['book_id_mapping_count'] = len(self.book_id_to_index)
        
        return stats
    
    def refresh_if_needed(self) -> bool:
        """Atualiza se encontrar arquivos mais recentes"""
        try:
            current_emb_file = self.current_files.get('embeddings')
            current_idx_file = self.current_files.get('index')
            
            if not current_emb_file or not current_idx_file:
                return False
            
            latest_emb, latest_idx = self.get_latest_files()
            
            if latest_emb != current_emb_file or latest_idx != current_idx_file:
                logger.info("🔄 Encontrados arquivos mais recentes, atualizando...")
                return self.load_latest_embeddings_with_metadata()
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao verificar atualizações: {e}")
            return False

    def get_book_id_by_index(self, idx: int) -> Optional[str]:
        """Retorna o book_id para um determinado índice do embedding"""
        if hasattr(self, 'metadata') and self.metadata and idx < len(self.metadata):
            meta = self.metadata[idx]
            return str(meta.get('book_id') or meta.get('id'))
        return None
    
    def get_index_by_book_id(self, book_id: str) -> Optional[int]:
        """Retorna o índice do embedding para um determinado book_id"""
        if hasattr(self, 'book_id_to_index'):
            return self.book_id_to_index.get(str(book_id))
        return None
    
    # Adicione este método ao seu GCSEmbeddingService

    def get_latest_complete_version(self) -> dict:
        """
        Busca a versão mais recente e COMPLETA (embeddings + índice + metadados + CSV)
        """
        try:
            # Tenta ler o active_knowledge_base.json primeiro
            active_blob = self.bucket.blob('embeddings/active_knowledge_base.json')
            
            if active_blob.exists():
                active_data = json.loads(active_blob.download_as_string())
                logger.info(f"✅ Versão ativa encontrada: {active_data.get('timestamp')}")
                return active_data
            
        except Exception as e:
            logger.warning(f"Não foi possível ler active_knowledge_base.json: {e}")
        
        return None