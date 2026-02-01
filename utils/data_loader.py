# utils/data_loader.py - Versão completa
import pandas as pd
import numpy as np
import ast
import logging
from google.cloud import storage
import io
import os
from datetime import datetime
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_path: str = None, gcs_bucket: str = None, gcs_prefix: str = "exports/"):
        self.data_path = data_path
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.data = None
        self.client = None
        self.stats = {}
        
        if gcs_bucket:
            try:
                self.client = storage.Client()
            except Exception as e:
                logger.warning(f"⚠️ Não foi possível inicializar cliente GCS: {e}")
                self.client = None
    
    def load_data(self) -> bool:
        """Carrega dados do GCS ou localmente e prepara o dataset"""
        try:
            # Prioridade 1: Carregar do GCS se configurado
            if self.gcs_bucket and self.client:
                logger.info(f"🔗 Tentando carregar dataset do GCS: {self.gcs_bucket}/{self.gcs_prefix}")
                if self._load_from_gcs():
                    return self._process_data()
            
            # Prioridade 2: Carregar localmente
            if self.data_path and os.path.exists(self.data_path):
                logger.info(f"📖 Carregando dataset local: {self.data_path}")
                if self._load_local():
                    return self._process_data()
            
            # Prioridade 3: Dataset vazio
            logger.warning("⚠️ Nenhum dataset configurado - usando vazio")
            self.data = pd.DataFrame()
            self._process_data()  # Processa mesmo vazio para consistência
            return True
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {e}")
            return False
    
    def _load_local(self) -> bool:
        """Carrega dataset localmente"""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"✅ Dataset local carregado: {len(self.data)} livros")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dataset local: {e}")
            return False
    
    def _load_from_gcs(self) -> bool:
        """Carrega o dataset mais recente do GCS"""
        try:
            # Listar arquivos CSV no bucket
            bucket = self.client.bucket(self.gcs_bucket)
            blobs = list(bucket.list_blobs(prefix=self.gcs_prefix))
            
            csv_files = []
            for blob in blobs:
                if blob.name.endswith('.csv') and 'EDU_books' in blob.name:
                    csv_files.append(blob)
            
            if not csv_files:
                logger.error(f"❌ Nenhum arquivo CSV encontrado em {self.gcs_bucket}/{self.gcs_prefix}")
                return False
            
            # Encontrar o arquivo mais recente pelo timestamp no nome
            latest_blob = self._get_latest_csv(csv_files)
            
            if not latest_blob:
                logger.error("❌ Não foi possível determinar o arquivo mais recente")
                return False
            
            logger.info(f"✅ Arquivo mais recente encontrado: {latest_blob.name}")
            logger.info(f"📏 Tamanho: {latest_blob.size / 1024 / 1024:.2f} MB")
            
            # Baixar e carregar o CSV
            content = latest_blob.download_as_bytes()
            
            # Converter para DataFrame
            self.data = pd.read_csv(io.BytesIO(content))

            # Adicione APÓS a linha: self.data = pd.read_csv(io.BytesIO(content))
            logger.info(f"=== DIAGNÓSTICO DO CSV CARREGADO ===")
            logger.info(f"📁 Arquivo CSV: {latest_blob.name}")
            logger.info(f"📊 Total de linhas: {len(self.data)}")
            logger.info(f"🔍 Verificando 'Six of Crows'...")
            # Buscar o livro específico
            mask = self.data['title'].str.contains('Six of Crows', case=False, na=False)
            if mask.any():
                row = self.data[mask].iloc[0]
                logger.info(f"✅ ENCONTRADO: ID {row.get('bookId')} - {row['title']}")
            else:
                logger.info(f"❌ NÃO ENCONTRADO: 'Six of Crows' não está neste CSV")
                
            # Mostrar primeiras linhas para debug
            logger.info(f"📋 Primeiros 5 títulos:")
            for idx, row in self.data.head().iterrows():
                logger.info(f"   {idx}: ID {row.get('bookId', 'N/A')} - {row.get('title', 'N/A')}")
            
            logger.info(f"🎉 Dataset carregado do GCS: {len(self.data)} livros")
            logger.info(f"📊 Colunas brutas: {list(self.data.columns)}")
            
            # Salvar localmente como cache (opcional)
            self._save_local_cache(latest_blob.name.split('/')[-1], content)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dataset do GCS: {e}")
            return False
    
    def _get_latest_csv(self, csv_files):
        """Encontra o arquivo CSV mais recente pelo timestamp no nome"""
        try:
            # Extrair timestamps dos nomes dos arquivos
            files_with_timestamps = []
            
            for blob in csv_files:
                filename = blob.name.split('/')[-1]
                
                # Padrão: YYYYMMDD_HHMMSS_EDU_books.csv
                pattern = r'(\d{8}_\d{6})_EDU_books\.csv'
                match = re.search(pattern, filename)
                
                if match:
                    timestamp_str = match.group(1)
                    try:
                        # Converter para datetime
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        files_with_timestamps.append((timestamp, blob))
                    except ValueError:
                        continue
            
            if not files_with_timestamps:
                # Se não conseguir extrair timestamp, usar data de modificação
                logger.warning("⚠️ Não foi possível extrair timestamp dos nomes, usando data de modificação")
                return max(csv_files, key=lambda x: x.updated)
            
            # Retornar o mais recente
            latest = max(files_with_timestamps, key=lambda x: x[0])
            return latest[1]
            
        except Exception as e:
            logger.error(f"❌ Erro ao determinar arquivo mais recente: {e}")
            return None
    
    def _save_local_cache(self, filename: str, content: bytes):
        """Salva localmente como cache"""
        try:
            cache_dir = "data/cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_path = os.path.join(cache_dir, filename)
            with open(cache_path, 'wb') as f:
                f.write(content)
            
            logger.info(f"💾 Cache salvo localmente: {cache_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Não foi possível salvar cache: {e}")
    
    def _process_data(self) -> bool:
        """Processa e prepara o dataset após carregamento - VERSÃO CORRIGIDA"""
        try:
            if self.data is None or self.data.empty:
                logger.warning("⚠️ Dataset vazio - sem processamento necessário")
                self.stats = {'total_books': 0, 'columns': []}
                return True
            
            logger.info("🔧 Processando dados...")
            
            # 1. Limpeza básica
            self.data = self.data.fillna('')
            logger.info("   ✅ Valores nulos preenchidos")
            
            # 2. Normalizar nomes de colunas (minúsculas, underscore)
            self.data.columns = [col.lower().replace(' ', '_') for col in self.data.columns]
            logger.info(f"   ✅ Colunas normalizadas: {list(self.data.columns)[:10]}...")
            
            # 3. Lista de colunas que devem ser convertidas de string para lista
            list_columns = ['all_genres', 'all_characters', 'all_setting', 'all_awards', 'author', 'authors', 'genres']
            
            for col in list_columns:
                if col in self.data.columns:
                    self.data[col] = self.data[col].apply(self._safe_convert_to_list)
                    logger.info(f"   ✅ Coluna '{col}' convertida para lista")
            
            # 4. GARANTIR QUE OS book_id NÃO SÃO RECRIADOS!
            # Primeiro, verificar se temos a coluna bookid no CSV
            if 'bookid' not in self.data.columns:
                # Se não tiver, criar mantendo o índice original + 1
                self.data['bookid'] = range(1, len(self.data) + 1)
                logger.info("   ✅ Coluna 'bookid' criada (1-based index)")
            else:
                # Se já existe, garantir que é numérico
                self.data['bookid'] = pd.to_numeric(self.data['bookid'], errors='coerce').fillna(0).astype(int)
                logger.info("   ✅ Coluna 'bookid' convertida para inteiro")
            
            # 5. Criar book_id a partir do bookid (para compatibilidade)
            if 'book_id' not in self.data.columns:
                self.data['book_id'] = self.data['bookid']
                logger.info("   ✅ Coluna 'book_id' criada a partir de bookid")
            
            # 6. Criar coluna combinada de autores se necessário
            if 'authors' not in self.data.columns and 'author' in self.data.columns:
                self.data['authors'] = self.data['author']
            
            # 7. Garantir tipos de dados corretos
            if 'rating' in self.data.columns:
                self.data['rating'] = pd.to_numeric(self.data['rating'], errors='coerce').fillna(0.0)
            
            # 8. VERIFICAR SE OS IDs BATEM
            logger.info(f"   📊 Primeiros 5 book_id: {list(self.data['book_id'].head())}")
            logger.info(f"   📊 Últimos 5 book_id: {list(self.data['book_id'].tail())}")
            
            # 9. Calcular estatísticas
            self._calculate_stats()
            
            logger.info(f"✅ Processamento concluído: {len(self.data)} livros")
            logger.info(f"📊 Estatísticas: {self.stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar dados: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _safe_convert_to_list(self, x):
        """Converte string para lista de forma segura"""
        if isinstance(x, list):
            return x
        elif isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            try:
                return ast.literal_eval(x)
            except:
                # Tentar limpar a string primeiro
                clean_str = x.strip('[]').replace("'", "").replace('"', '')
                return [item.strip() for item in clean_str.split(',') if item.strip()]
        elif pd.isna(x) or x == '':
            return []
        else:
            return [str(x)]
    
    def _calculate_stats(self):
        """Calcula estatísticas do dataset"""
        try:
            if self.data is None or self.data.empty:
                self.stats = {
                    'total_books': 0,
                    'columns': [],
                    'memory_usage_mb': 0
                }
                return
            
            # Calcular número de autores únicos
            unique_authors = 0
            if 'author' in self.data.columns:
                try:
                    # Para lista de autores
                    all_authors = []
                    for authors_list in self.data['author']:
                        if isinstance(authors_list, list):
                            all_authors.extend(authors_list)
                        elif isinstance(authors_list, str) and authors_list:
                            all_authors.append(authors_list)
                    unique_authors = len(set(all_authors))
                except:
                    unique_authors = self.data['author'].nunique()
            
            # Calcular estatísticas de rating
            avg_rating = max_rating = min_rating = 0.0
            if 'rating' in self.data.columns:
                avg_rating = float(self.data['rating'].mean())
                max_rating = float(self.data['rating'].max())
                min_rating = float(self.data['rating'].min())
            
            self.stats = {
                'total_books': len(self.data),
                'unique_authors': unique_authors,
                'avg_rating': avg_rating,
                'max_rating': max_rating,
                'min_rating': min_rating,
                'columns': list(self.data.columns),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
                'sample_titles': list(self.data['title'].head(3).values) if 'title' in self.data.columns else []
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao calcular estatísticas: {e}")
            self.stats = {
                'total_books': len(self.data) if self.data is not None else 0,
                'columns': list(self.data.columns) if self.data is not None else []
            }
    
    def get_data(self) -> pd.DataFrame:
        """Retorna os dados carregados"""
        return self.data
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do dataset"""
        return self.stats
    
    def get_sample(self, n: int = 5) -> List[Dict]:
        """Retorna uma amostra dos dados"""
        if self.data is None or self.data.empty:
            return []
        
        sample = self.data.head(n)
        return sample.to_dict('records')
    
    def get_book_by_id(self, book_id) -> Optional[Dict]:
        """Obtém livro por ID"""
        if self.data is None or self.data.empty:
            return None
        
        # Tentar diferentes nomes de coluna para ID
        id_columns = ['book_id', 'bookid', 'id', 'bookId', 'BookId']
        
        for col in id_columns:
            if col in self.data.columns:
                book = self.data[self.data[col] == book_id]
                if not book.empty:
                    return book.iloc[0].to_dict()
        
        return None
    
    def search_by_title(self, title_query: str, limit: int = 10) -> List[Dict]:
        """Busca livros por título"""
        if self.data is None or 'title' not in self.data.columns:
            return []
        
        mask = self.data['title'].str.contains(title_query, case=False, na=False)
        results = self.data[mask].head(limit)
        
        return results.to_dict('records')
    
    def get_books_by_genre(self, genre: str, limit: int = 10) -> List[Dict]:
        """Busca livros por gênero"""
        if self.data is None or ('genres' not in self.data.columns and 'all_genres' not in self.data.columns):
            return []
        
        results = []
        genre_lower = genre.lower()
        
        for _, row in self.data.iterrows():
            # Verificar em 'genres' ou 'all_genres'
            genres_list = []
            if 'genres' in row and isinstance(row['genres'], list):
                genres_list = [g.lower() for g in row['genres'] if isinstance(g, str)]
            elif 'all_genres' in row and isinstance(row['all_genres'], list):
                genres_list = [g.lower() for g in row['all_genres'] if isinstance(g, str)]
            
            if any(genre_lower in g for g in genres_list):
                results.append(row.to_dict())
            
            if len(results) >= limit:
                break
        
        return results