# diagnose_csv.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import DataLoader
import logging

logging.basicConfig(level=logging.INFO)

def diagnose():
    print("🔍 Diagnosticando DataLoader...")
    
    # 1. Criar DataLoader igual ao agent_service
    loader = DataLoader(
        gcs_bucket="book-agent-embeddings-bucket",
        gcs_prefix="exports/"
    )
    
    # 2. Tentar carregar
    if loader.load_data():
        print(f"✅ CSV carregado: {len(loader.data)} livros")
        
        # 3. Verificar arquivos no bucket
        if hasattr(loader, 'client') and loader.client:
            bucket = loader.client.bucket("book-agent-embeddings-bucket")
            blobs = list(bucket.list_blobs(prefix="exports/"))
            print(f"📁 Arquivos no bucket 'exports/':")
            for blob in blobs:
                print(f"   - {blob.name} ({blob.size} bytes)")
        
        # 4. Buscar livros específicos
        print(f"\n🔎 Buscando livros no DataFrame carregado:")
        
        # Procurar Six of Crows
        six_mask = loader.data['title'].str.contains('Six of Crows', case=False, na=False)
        if six_mask.any():
            six_row = loader.data[six_mask].iloc[0]
            print(f"✅ 'Six of Crows': ID {six_row.get('bookid')}")
        else:
            print(f"❌ 'Six of Crows': NÃO ENCONTRADO")
        
        # Procurar Harry Potter Series Box Set
        hp_mask = loader.data['title'].str.contains('Harry Potter Series Box Set', case=False, na=False)
        if hp_mask.any():
            hp_row = loader.data[hp_mask].iloc[0]
            print(f"📚 'Harry Potter Series Box Set': ID {hp_row.get('bookid')}")
        
        # Mostrar primeiros IDs
        print(f"\n📋 Primeiros 10 bookIds: {list(loader.data['bookid'].head(10).values)}")
    else:
        print("❌ Falha ao carregar dados")

if __name__ == "__main__":
    diagnose()