import io
import numpy as np
import faiss
import logging
import json
import pandas as pd
from google.cloud import storage
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import tempfile
import re
from datetime import datetime
from tqdm import tqdm
import torch

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Gerador de embeddings a partir de CSV do GCS.
    Baixa o CSV, gera embeddings para TODOS os livros e faz upload dos resultados.
    """
    
    def __init__(self, 
                 bucket_name: str = "book-agent-embeddings-bucket",
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 use_gpu: bool = True):
        
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Modelo de embeddings
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = None
        self.model = None
        
        # Dados
        self.df = None
        self.texts = None
        self.book_ids = None
        self.embeddings = None
        self.index = None
        self.metadata = []
        
    def initialize_model(self) -> bool:
        """Inicializa o modelo SentenceTransformer"""
        try:
            logger.info(f"🚀 Inicializando modelo: {self.model_name}")
            
            if self.use_gpu and torch.cuda.is_available():
                self.device = 'cuda'
                logger.info(f"✅ Usando GPU: {torch.cuda.get_device_name(0)}")
                self.model = SentenceTransformer(self.model_name, device='cuda')
            else:
                self.device = 'cpu'
                logger.info("✅ Usando CPU")
                self.model = SentenceTransformer(self.model_name)
            
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar modelo: {e}")
            return False
    
    def download_csv_from_gcs(self, csv_path: str = None) -> bool:
        """
        Baixa o arquivo CSV do GCS.
        
        Args:
            csv_path: Caminho do CSV no GCS (ex: "exports/20260119_231738_EDU_books.csv")
        """
        try:
            if csv_path is None:
                # Tenta encontrar o CSV mais recente
                csv_path = self._get_latest_csv()
            
            logger.info(f"📥 Baixando CSV do GCS: {csv_path}")
            
            blob = self.bucket.blob(csv_path)
            
            if not blob.exists():
                logger.error(f"❌ CSV não encontrado: {csv_path}")
                
                # Lista CSVs disponíveis
                self._list_available_csvs()
                return False
            
            # Download e leitura do CSV
            csv_data = blob.download_as_string()
            self.df = pd.read_csv(io.BytesIO(csv_data))
            
            logger.info(f"✅ CSV carregado com sucesso!")
            logger.info(f"   📊 Shape: {self.df.shape}")
            logger.info(f"   📋 Colunas: {list(self.df.columns)}")
            logger.info(f"   📝 Total de registros: {len(self.df)}")
            
            # Identificar colunas importantes
            self._identify_columns()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao baixar CSV: {e}")
            return False
    
    def _get_latest_csv(self) -> str:
        """Encontra o CSV mais recente no bucket"""
        try:
            blobs = list(self.client.list_blobs(self.bucket_name))
            csv_files = []
            
            for blob in blobs:
                if blob.name.endswith('.csv') and 'export' in blob.name.lower():
                    timestamp = self._extract_timestamp(blob.name)
                    csv_files.append((timestamp or datetime.min, blob.name))
            
            if csv_files:
                csv_files.sort(key=lambda x: x[0], reverse=True)
                latest_csv = csv_files[0][1]
                logger.info(f"📄 CSV mais recente encontrado: {latest_csv}")
                return latest_csv
            
            return "exports/20260119_231738_EDU_books.csv"  # Fallback
            
        except Exception as e:
            logger.error(f"Erro ao buscar CSV: {e}")
            return "exports/20260119_231738_EDU_books.csv"
    
    def _list_available_csvs(self):
        """Lista todos os CSVs disponíveis no bucket"""
        try:
            logger.info("📋 CSVs disponíveis no bucket:")
            blobs = list(self.client.list_blobs(self.bucket_name))
            for blob in blobs:
                if blob.name.endswith('.csv'):
                    logger.info(f"   - {blob.name}")
        except:
            pass
    
    def _extract_timestamp(self, filename: str) -> Optional[datetime]:
        """Extrai timestamp do nome do arquivo"""
        try:
            match = re.search(r'(\d{8}_\d{6})', filename)
            if match:
                return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except:
            pass
        return None
    
    def _identify_columns(self):
        """Identifica colunas importantes no DataFrame"""
        # Coluna de ID - SEU CSV USA 'bookId', NÃO 'book_id'!
        for col in ['bookId', 'id', 'book_id', 'livro_id', 'codigo', 'isbn']:
            if col in self.df.columns:
                self.id_column = col
                self.book_ids = self.df[col].astype(str).tolist()  # <-- DEFINE AQUI!
                logger.info(f"   🆔 Coluna de ID: {col}")
                break
        
        if not hasattr(self, 'id_column'):
            self.id_column = self.df.columns[0]
            self.book_ids = self.df[self.id_column].astype(str).tolist()
            logger.warning(f"   ⚠️ Usando primeira coluna como ID: {self.id_column}")
        
        # Colunas de texto para embedding
        text_columns = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['title', 'titulo', 'description', 'descricao', 'text', 'conteudo', 'sinopse']):
                text_columns.append(col)
        
        if text_columns:
            self.text_columns = text_columns
            logger.info(f"   📝 Colunas de texto: {text_columns}")
        else:
            self.text_columns = self.df.select_dtypes(include=['object']).columns.tolist()
            logger.info(f"   📝 Usando colunas de string: {self.text_columns[:5]}...")

    def prepare_texts(self) -> List[str]:
        """
        Prepara os textos para gerar embeddings.
        Combina múltiplas colunas em um único texto.
        """
        if self.df is None:
            logger.error("❌ DataFrame não carregado")
            return []
        
        logger.info("🔧 Preparando textos para embeddings...")
        
        texts = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Preparando textos"):
            # Combina todas as colunas de texto relevantes
            text_parts = []
            
            # Título (prioridade máxima)
            title = None
            for col in ['titulo', 'title', 'nome', 'name']:
                if col in self.df.columns and pd.notna(row[col]):
                    title = str(row[col]).strip()
                    text_parts.append(f"Título: {title}")
                    break
            
            # Descrição/Sinopse
            for col in ['descricao', 'description', 'sinopse', 'synopsis', 'resumo']:
                if col in self.df.columns and pd.notna(row[col]):
                    desc = str(row[col]).strip()
                    if len(desc) > 10:  # Ignora descrições muito curtas
                        text_parts.append(f"Descrição: {desc}")
                    break
            
            # Autor
            for col in ['autor', 'author', 'autores']:
                if col in self.df.columns and pd.notna(row[col]):
                    text_parts.append(f"Autor: {str(row[col]).strip()}")
                    break
            
            # Assunto/Categoria
            for col in ['assunto', 'subject', 'categoria', 'category', 'area']:
                if col in self.df.columns and pd.notna(row[col]):
                    text_parts.append(f"Assunto: {str(row[col]).strip()}")
                    break
            
            # Editora
            for col in ['editora', 'publisher']:
                if col in self.df.columns and pd.notna(row[col]):
                    text_parts.append(f"Editora: {str(row[col]).strip()}")
                    break
            
            # Ano
            for col in ['ano', 'year', 'publicacao']:
                if col in self.df.columns and pd.notna(row[col]):
                    text_parts.append(f"Ano: {str(row[col]).strip()}")
                    break
            
            # Se não encontrou nada, usa todas as colunas
            if not text_parts:
                for col in self.df.columns:
                    if pd.notna(row[col]) and isinstance(row[col], str):
                        text_parts.append(f"{col}: {str(row[col]).strip()[:200]}")
            
            combined_text = " | ".join(text_parts)
            texts.append(combined_text)
        
        self.texts = texts
        logger.info(f"✅ Textos preparados: {len(texts)} registros")
        logger.info(f"   📝 Exemplo: {texts[0][:200]}...")
        
        return texts
    
    def generate_embeddings(self, batch_size: int = 64) -> bool:
        """
        Gera embeddings para todos os textos.
        """
        if self.model is None:
            logger.error("❌ Modelo não inicializado")
            return False
        
        if not self.texts:
            logger.error("❌ Textos não preparados")
            return False
        
        logger.info("🧠 Gerando embeddings...")
        logger.info(f"   📊 Total de textos: {len(self.texts)}")
        logger.info(f"   📦 Batch size: {batch_size}")
        logger.info(f"   💻 Device: {self.device}")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(self.texts), batch_size), desc="Gerando embeddings"):
            batch_texts = self.texts[i:i + batch_size]
            
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            all_embeddings.append(batch_embeddings)
            
            if i % (batch_size * 10) == 0:
                logger.info(f"   Progresso: {i}/{len(self.texts)} embeddings gerados")
        
        self.embeddings = np.vstack(all_embeddings)
        
        logger.info(f"✅ Embeddings gerados com sucesso!")
        logger.info(f"   📊 Shape: {self.embeddings.shape}")
        logger.info(f"   💾 Memória: {self.embeddings.nbytes / 1024 / 1024:.2f} MB")
        
        return True
    
    def create_faiss_index(self) -> bool:
        """
        Cria índice FAISS para busca semântica.
        """
        if self.embeddings is None:
            logger.error("❌ Embeddings não gerados")
            return False
        
        logger.info("🔧 Criando índice FAISS...")
        
        # Cria índice plano (mais preciso)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        
        # Adiciona embeddings ao índice
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"✅ Índice FAISS criado com sucesso!")
        logger.info(f"   📊 Total de vetores: {self.index.ntotal}")
        logger.info(f"   🏷️  Tipo do índice: {type(self.index).__name__}")
        
        return True
    
    def create_metadata(self) -> dict:
        """
        Cria metadados completos com mapeamento book_id -> índice.
        """
        if self.df is None or self.book_ids is None:
            logger.error("❌ Dados não carregados")
            return {}
        
        logger.info("📋 Criando metadados...")
        
        metadata = []
        book_id_to_index = {}
        
        for idx, (_, row) in enumerate(self.df.iterrows()):
            # Converte row para dict, tratando NaN
            record = {}
            for col in self.df.columns:
                value = row[col]
                if pd.isna(value):
                    record[col] = None
                elif isinstance(value, (np.integer, np.floating)):
                    record[col] = value.item()
                else:
                    record[col] = value
            
            metadata.append(record)
            
            # Mapeamento book_id -> índice
            book_id = str(row.get(self.id_column)) if self.id_column else str(idx)
            book_id_to_index[book_id] = idx
        
        self.metadata = metadata
        self.book_id_to_index = book_id_to_index
        
        logger.info(f"✅ Metadados criados: {len(metadata)} registros")
        logger.info(f"🔗 Book IDs mapeados: {len(book_id_to_index)}")
        
        return {
            'metadata': metadata,
            'book_id_to_index': book_id_to_index
        }
    
    def upload_to_gcs(self) -> bool:
        """
        Faz upload dos embeddings, índice e metadados para o GCS.
        CORRIGIDO: Timeout aumentado para 600 segundos!
        """
        if self.embeddings is None or self.index is None or not self.metadata:
            logger.error("❌ Dados incompletos para upload")
            return False
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prefix = f"embeddings/{timestamp}_EDU_books"
            
            logger.info("=" * 80)
            logger.info(f"📤 FAZENDO UPLOAD PARA GCS: {prefix}")
            logger.info("=" * 80)
            
            # 1. Upload dos embeddings (.npy) - 76MB
            embeddings_filename = f"{prefix}_embeddings.npy"
            embeddings_blob = self.bucket.blob(embeddings_filename)
            
            with io.BytesIO() as f:
                np.save(f, self.embeddings)
                f.seek(0)
                # TIMEOUT AUMENTADO PARA 600 SEGUNDOS (10 MINUTOS)
                embeddings_blob.upload_from_file(
                    f, 
                    timeout=800,  # <-- AQUI!
                    retry=storage.retry.DEFAULT_RETRY  # Adiciona retry automático
                )
            
            logger.info(f"✅ Embeddings upload: {embeddings_filename}")
            
            # 2. Upload do índice FAISS
            index_filename = f"{prefix}_index.faiss"
            
            with tempfile.NamedTemporaryFile(suffix='.faiss', delete=True) as tmp_file:
                faiss.write_index(self.index, tmp_file.name)
                index_blob = self.bucket.blob(index_filename)
                # TIMEOUT AUMENTADO PARA 600 SEGUNDOS
                index_blob.upload_from_filename(
                    tmp_file.name,
                    timeout=800  # <-- AQUI!
                )
            
            logger.info(f"✅ Índice FAISS upload: {index_filename}")
            
            # 3. Upload dos metadados (.json) - 150MB
            metadata_filename = f"{prefix}_metadata.json"
            metadata_blob = self.bucket.blob(metadata_filename)
            
            logger.info("📦 Preparando metadata.json...")
            metadata_json = json.dumps(self.metadata, indent=2, ensure_ascii=False)
            tamanho_mb = len(metadata_json) / 1024 / 1024
            logger.info(f"   Tamanho do JSON: {tamanho_mb:.2f} MB")
            
            # TIMEOUT AUMENTADO PARA 900 SEGUNDOS (15 MINUTOS) - É O MAIOR ARQUIVO!
            metadata_blob._chunk_size = 20 * 1024 * 1024  # 20MB chunks
            metadata_blob.upload_from_string(
                metadata_json,
                content_type='application/json',
                timeout=900,  # <-- 15 MINUTOS!
                retry=storage.retry.DEFAULT_RETRY
            )
            
            logger.info(f"✅ Metadados upload: {metadata_filename}")
            
            # ... resto do código ...
            
        except Exception as e:
            logger.error(f"❌ Erro ao fazer upload: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_complete_pipeline(self, csv_path: str = None) -> bool:
        """
        Executa o pipeline completo:
        1. Baixa CSV
        2. Prepara textos
        3. Gera embeddings
        4. Cria índice FAISS
        5. Cria metadados
        6. Upload para GCS
        """
        logger.info("=" * 80)
        logger.info("🚀 INICIANDO PIPELINE COMPLETO DE EMBEDDINGS")
        logger.info("=" * 80)
        
        # 1. Inicializar modelo
        if not self.initialize_model():
            return False
        
        # 2. Baixar CSV
        if not self.download_csv_from_gcs(csv_path):
            return False
        
        # 3. Preparar textos
        if not self.prepare_texts():
            return False
        
        # 4. Gerar embeddings
        if not self.generate_embeddings():
            return False
        
        # 5. Criar índice FAISS
        if not self.create_faiss_index():
            return False
        
        # 6. Criar metadados
        self.create_metadata()
        
        # 7. Upload para GCS
        if not self.upload_to_gcs():
            return False
        
        logger.info("=" * 80)
        logger.info("🎉 PIPELINE COMPLETO FINALIZADO COM SUCESSO!")
        logger.info("=" * 80)
        
        return True