# diagnose_csv_fixed.py
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import DataLoader
import logging

logging.basicConfig(level=logging.INFO)

def diagnose():
    print("🔍 DIAGNÓSTICO COMPLETO DO SISTEMA")
    print("="*50)
    
    # 1. DataLoader
    print("\n📖 1. VERIFICANDO DATALOADER...")
    loader = DataLoader(
        gcs_bucket="book-agent-embeddings-bucket",
        gcs_prefix="exports/"
    )
    
    if loader.load_data():
        print(f"✅ CSV carregado: {len(loader.data)} livros")
        
        # Verificar Six of Crows
        six_mask = loader.data['title'].str.contains('Six of Crows', case=False, na=False)
        if six_mask.any():
            six_row = loader.data[six_mask].iloc[0]
            print(f"✅ 'Six of Crows': ID {six_row.get('bookid')} - '{six_row['title']}'")
            print(f"   Autores: {six_row.get('author')}")
            print(f"   Gêneros: {six_row.get('genres')}")
        else:
            print(f"❌ 'Six of Crows': NÃO ENCONTRADO")
        
        # Verificar Harry Potter Series Box Set
        hp_mask = loader.data['title'].str.contains('Harry Potter Series Box Set', case=False, na=False)
        if hp_mask.any():
            hp_row = loader.data[hp_mask].iloc[0]
            print(f"📚 'Harry Potter Series Box Set': ID {hp_row.get('bookid')}")
        else:
            print(f"ℹ️ 'Harry Potter Series Box Set': Não encontrado (ESPERADO!)")
        
        # Verificar Harry Potter e a Ordem da Fênix
        hp5_mask = loader.data['title'].str.contains('Order of the Phoenix', case=False, na=False)
        if hp5_mask.any():
            hp5_row = loader.data[hp5_mask].iloc[0]
            print(f"📚 'Harry Potter and the Order of the Phoenix': ID {hp5_row.get('bookid')}")
        
        print(f"\n📊 Primeiros 10 book_ids: {list(loader.data['bookid'].head(10).values)}")
        
    print("\n🔍 2. VERIFICANDO EMBEDDINGS...")
    
    # 2. Embedding Service
    try:
        from services.embedding_service import EmbeddingService
        
        embedding_service = EmbeddingService()
        if embedding_service.initialize():
            stats = embedding_service.get_stats()
            print(f"✅ Embeddings carregados")
            print(f"   Índice: {stats.get('index', {}).get('size', 0)} vetores")
            print(f"   Embeddings: {stats.get('embeddings', {}).get('shape', 'N/A')}")
            
            # TESTE CRÍTICO: Verificar se o índice tem 52478 vetores
            expected_size = len(loader.data)
            actual_size = stats.get('index', {}).get('size', 0)
            
            if actual_size == expected_size:
                print(f"✅ CORRESPONDÊNCIA PERFEITA: Índice ({actual_size}) = CSV ({expected_size})")
            else:
                print(f"❌ DESCOMPASSO: Índice ({actual_size}) ≠ CSV ({expected_size})")
                print(f"   Isso explica por que 'Six of Crows' não aparece nas buscas!")
        else:
            print("❌ Falha ao inicializar embeddings")
            
    except Exception as e:
        print(f"❌ Erro nos embeddings: {e}")
    
    print("\n🔍 3. BUSCA DE TESTE...")
    
    # 3. Buscar livro específico por ID
    print("Tentando buscar livro com ID 409...")
    try:
        book_409 = loader.get_book_by_id(409)
        if book_409:
            print(f"✅ Livro ID 409 encontrado via get_book_by_id:")
            print(f"   Título: {book_409.get('title')}")
            print(f"   Autores: {book_409.get('author')}")
        else:
            print(f"❌ Livro ID 409 NÃO encontrado via get_book_by_id")
    except Exception as e:
        print(f"❌ Erro na busca: {e}")

if __name__ == "__main__":
    diagnose()