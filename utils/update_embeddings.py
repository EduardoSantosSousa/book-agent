# update_embeddings.py
import subprocess
import google.cloud.storage

def update_embeddings(dataset_path="data/book_dataset_treated.csv"):
    """Atualiza embeddings e envia para GCS"""
    
    # 1. Reconstruir embeddings (usando seu código existente)
    from services.embedding_service import EmbeddingService
    
    embedding_service = EmbeddingService()
    embedding_service.initialize()
    
    # Carregar dados atualizados
    from utils.data_loader import DataLoader
    data_loader = DataLoader(dataset_path)
    data_loader.load_data()
    
    # Gerar novos embeddings
    texts = embedding_service.prepare_texts_batch(data_loader.data)
    embedding_service.build_index(texts, save_path="temp_index")
    
    # 2. Enviar para Cloud Storage
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.bucket("book-agent-embeddings-bucket")
    
    # Fazer upload dos novos arquivos
    bucket.blob("embeddings/latest_index.faiss").upload_from_filename("temp_index_index.faiss")
    bucket.blob("embeddings/latest_embeddings.npy").upload_from_filename("temp_index_embeddings.npy")
    
    # 3. (Opcional) Manter versões anteriores
    #import datetime
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #bucket.blob(f"embeddings/archive/index_{timestamp}.faiss").upload_from_filename("temp_index_index.faiss")
    
    print("Embeddings atualizados com sucesso!")