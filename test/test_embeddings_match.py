# test_embeddings_match.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services.embedding_service import EmbeddingService
from utils.data_loader import DataLoader
import logging

logging.basicConfig(level=logging.INFO)

def test_embeddings_match():
    print("🧪 TESTE DE CORRESPONDÊNCIA EMBEDDINGS vs CSV")
    print("="*50)
    
    # 1. Carregar CSV atual
    print("\n📖 1. CARREGANDO CSV...")
    loader = DataLoader(
        gcs_bucket="book-agent-embeddings-bucket",
        gcs_prefix="exports/"
    )
    
    if not loader.load_data():
        print("❌ Falha ao carregar CSV")
        return
    
    print(f"✅ CSV: {len(loader.data)} livros")
    
    # Buscar livros específicos no CSV
    six_row = loader.data[loader.data['bookid'] == 409]
    hp_row = loader.data[loader.data['bookid'] == 410]
    
    if not six_row.empty:
        print(f"📚 CSV - ID 409: '{six_row.iloc[0]['title']}'")
    if not hp_row.empty:
        print(f"📚 CSV - ID 410: '{hp_row.iloc[0]['title']}'")
    
    # 2. Carregar Embeddings
    print("\n🧠 2. CARREGANDO EMBEDDINGS...")
    embedding_service = EmbeddingService()
    
    if not embedding_service.initialize():
        print("❌ Falha ao inicializar embeddings")
        return
    
    stats = embedding_service.get_stats()
    print(f"✅ Embeddings carregados")
    print(f"   Índice: {stats.get('index', {}).get('size', 0)} vetores")
    
    # 3. TESTE CRÍTICO: Buscar livro por posição no índice
    print("\n🔍 3. TESTANDO CORRESPONDÊNCIA ÍNDICE-CSV...")
    
    # O índice FAISS usa posições (0, 1, 2...) que DEVEM corresponder às linhas do CSV
    # Posição 408 no índice (índice começa em 0) deve ser livro ID 409
    
    if embedding_service.gcs_consumer and embedding_service.gcs_consumer.current_embeddings is not None:
        index_size = embedding_service.gcs_consumer.current_index.ntotal
        csv_size = len(loader.data)
        
        print(f"📊 Tamanho CSV: {csv_size}")
        print(f"📊 Tamanho Índice: {index_size}")
        
        if index_size != csv_size:
            print(f"❌❌❌ DESCOMPASSO CRÍTICO!")
            print(f"   Índice tem {index_size} vetores")
            print(f"   CSV tem {csv_size} linhas")
            print(f"   Diferença: {abs(index_size - csv_size)} registros")
            print("\n🎯 ISSO EXPLICA O PROBLEMA!")
            print("Os embeddings foram gerados a partir de um CSV diferente.")
        else:
            print(f"✅ Tamanhos iguais: {index_size} = {csv_size}")
    
    # 4. Verificar versão dos embeddings
    print("\n📅 4. VERIFICANDO VERSÃO DOS EMBEDDINGS...")
    if embedding_service.gcs_consumer and embedding_service.gcs_consumer.version_info:
        version = embedding_service.gcs_consumer.version_info
        print(f"📁 Embeddings carregados:")
        print(f"   Timestamp: {version.get('timestamp', 'N/A')}")
        print(f"   Arquivo: {version.get('embeddings_filename', 'N/A')}")
        
        # Extrair data do nome do arquivo
        import re
        filename = version.get('embeddings_filename', '')
        match = re.search(r'(\d{8}_\d{6})', filename)
        if match:
            emb_timestamp = match.group(1)
            print(f"   Data embeddings: {emb_timestamp}")
            
            # Comparar com data do CSV
            csv_filename = "20260119_231738_EDU_books.csv"
            csv_match = re.search(r'(\d{8}_\d{6})', csv_filename)
            if csv_match:
                csv_timestamp = csv_match.group(1)
                print(f"   Data CSV: {csv_timestamp}")
                
                if emb_timestamp != csv_timestamp:
                    print(f"❌ DATAS DIFERENTES!")
                    print(f"   Embeddings: {emb_timestamp}")
                    print(f"   CSV: {csv_timestamp}")
                else:
                    print(f"✅ DATAS IGUAIS!")
    
    print("\n" + "="*50)
    print("🎯 CONCLUSÃO:")
    
    if 'DESCOMPASSO' in locals() and DESCOMPASSO:
        print("""
        ⚠️  PROBLEMA CONFIRMADO:
        Os embeddings no GCS são baseados em um CSV antigo.
        
        🛠️  SOLUÇÃO NECESSÁRIA:
        1. Regenerar embeddings a partir do CSV atual
        2. Fazer upload para o GCS
        3. Reiniciar o serviço
        """)
    else:
        print("✅ Sistema parece OK - problema deve estar em outro lugar")

if __name__ == "__main__":
    test_embeddings_match()