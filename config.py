import os
from dotenv import load_dotenv

# Determina qual arquivo .env carregar
env_file = '.env.production' if os.getenv('ENVIRONMENT') == 'production' else '.env'
load_dotenv(env_file)

class Config:
    # Ambiente
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    
    # Embeddings - LOCAL vs PRODUÇÃO
    EMBEDDINGS_SOURCE = os.getenv('EMBEDDINGS_SOURCE', 'local')  # 'local' ou 'gcs'
    
    # Caminhos LOCAIS
    LOCAL_EMBEDDINGS_PATH = os.getenv('LOCAL_EMBEDDINGS_PATH', 'embeddings/local/')
    
    # IMPORTANTE: Remover a barra extra na concatenação
    @property
    def LOCAL_INDEX_FILE(self):
        return os.path.join(self.LOCAL_EMBEDDINGS_PATH, 'book_index_gpu_index.faiss')
    
    @property
    def LOCAL_EMBEDDINGS_FILE(self):
        return os.path.join(self.LOCAL_EMBEDDINGS_PATH, 'book_index_gpu_embeddings.npy')
    
    # Configurações GCS (PRODUÇÃO)
    GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'book-agent-embeddings-bucket')
    GCS_INDEX_PATH = os.getenv('GCS_INDEX_PATH', 'book_index_gpu_index.faiss')
    GCS_EMBEDDINGS_PATH = os.getenv('GCS_EMBEDDINGS_PATH', 'book_index_gpu_embeddings.npy')
    GCS_EMBEDDINGS_PREFIX = os.getenv('GCS_EMBEDDINGS_PREFIX', 'embeddings/')
    
    # Ollama
    #OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'ollama-service.book-agent-ns.svc.cluster.local')
    
    # App
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Propriedade para compatibilidade
    @property
    def embeddings_base_path(self):
        """Caminho base para embeddings - para compatibilidade"""
        return self.LOCAL_EMBEDDINGS_PATH

class EmbeddingLoader:
    """Carrega embeddings do ambiente correto"""
    
    @staticmethod
    def load_index_files(config: Config):
        """Carrega arquivos de índice do local ou GCS"""
        
        if config.EMBEDDINGS_SOURCE == 'gcs':
            return EmbeddingLoader._load_from_gcs(config)
        else:
            return EmbeddingLoader._load_local(config)
    
    @staticmethod
    def _load_local(config: Config):
        """Carrega arquivos localmente"""
        import faiss
        import numpy as np
        
        print(f"📂 Carregando embeddings locais de: {config.LOCAL_EMBEDDINGS_PATH}")
        
        try:
            index = faiss.read_index(config.LOCAL_INDEX_FILE)
            embeddings = np.load(config.LOCAL_EMBEDDINGS_FILE)
            print(f"✅ Embeddings carregados: {embeddings.shape}")
            return index, embeddings
        except Exception as e:
            print(f"❌ Erro ao carregar arquivos locais: {e}")
            
            # Verificar se os arquivos existem
            print(f"🔍 Verificando existência dos arquivos...")
            print(f"   Índice: {config.LOCAL_INDEX_FILE} - {'EXISTE' if os.path.exists(config.LOCAL_INDEX_FILE) else 'NÃO EXISTE'}")
            print(f"   Embeddings: {config.LOCAL_EMBEDDINGS_FILE} - {'EXISTE' if os.path.exists(config.LOCAL_EMBEDDINGS_FILE) else 'NÃO EXISTE'}")
            
            # Tentar caminhos alternativos
            return EmbeddingLoader._try_alternative_paths()
    
    @staticmethod
    def _try_alternative_paths():
        """Tenta caminhos alternativos para os arquivos"""
        import faiss
        import numpy as np
        
        alternative_paths = [
            ('book_index_gpu_index.faiss', 'book_index_gpu_embeddings.npy'),
            ('embeddings/local/book_index_gpu_index.faiss', 'embeddings/local/book_index_gpu_embeddings.npy'),
            ('embeddings/book_index_gpu_index.faiss', 'embeddings/book_index_gpu_embeddings.npy'),
            ('embeddings/local/book_index.faiss', 'embeddings/local/book_embeddings.npy'),
        ]
        
        for index_path, embeddings_path in alternative_paths:
            try:
                if os.path.exists(index_path) and os.path.exists(embeddings_path):
                    print(f"📂 Encontrado em caminho alternativo: {index_path}")
                    index = faiss.read_index(index_path)
                    embeddings = np.load(embeddings_path)
                    print(f"✅ Embeddings carregados: {embeddings.shape}")
                    return index, embeddings
            except Exception as e:
                print(f"⚠️  Caminho alternativo falhou {index_path}: {e}")
                continue
        
        raise FileNotFoundError("Não foi possível encontrar arquivos de embeddings em nenhum caminho")
    
    @staticmethod
    def _load_from_gcs(config: Config):
        """Baixa e carrega arquivos do GCS"""
        import faiss
        import numpy as np
        from google.cloud import storage
        import tempfile
        
        print(f"☁️  Carregando embeddings do GCS: {config.GCS_BUCKET_NAME}")
        
        try:
            # Inicializa cliente GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(config.GCS_BUCKET_NAME)
            
            # Cria arquivos temporários
            with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp_index_file, \
                 tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_emb_file:
                
                tmp_index_path = tmp_index_file.name
                tmp_emb_path = tmp_emb_file.name
                
                try:
                    # Baixa do GCS para arquivos temporários
                    print(f"⬇️  Baixando {config.GCS_INDEX_PATH}...")
                    index_blob = bucket.blob(config.GCS_INDEX_PATH)
                    index_blob.download_to_filename(tmp_index_path)
                    
                    print(f"⬇️  Baixando {config.GCS_EMBEDDINGS_PATH}...")
                    emb_blob = bucket.blob(config.GCS_EMBEDDINGS_PATH)
                    emb_blob.download_to_filename(tmp_emb_path)
                    
                    # Carrega dos arquivos temporários
                    index = faiss.read_index(tmp_index_path)
                    embeddings = np.load(tmp_emb_path)
                    
                    print(f"✅ Embeddings carregados do GCS: {embeddings.shape}")
                    return index, embeddings
                    
                finally:
                    # Limpa arquivos temporários
                    if os.path.exists(tmp_index_path):
                        os.unlink(tmp_index_path)
                    if os.path.exists(tmp_emb_path):
                        os.unlink(tmp_emb_path)
                
        except Exception as e:
            print(f"❌ Erro ao carregar do GCS: {e}")
            # Fallback para local se GCS falhar
            print("🔄 Tentando fallback para arquivos locais...")
            return EmbeddingLoader._load_local(config)

config = Config()