#!/usr/bin/env python3
"""
Testa o consumidor GCS localmente.
"""
import os
import sys

# Adicionar diretório do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Agora importar diretamente sem passar por __init__.py
from services.gcs_consumer_service import GCSEmbeddingConsumer

def main():
    print("🧪 Testando Consumidor GCS")
    
    # Configurações diretas (evitar importar config)
    bucket_name = "book-agent-embeddings-bucket"
    embeddings_prefix = "embeddings/"
    
    print(f"   Bucket: {bucket_name}")
    print(f"   Prefixo: {embeddings_prefix}")
    print("=" * 50)
    
    try:
        # 1. Testar conexão
        print("\n1. 🔗 Testando conexão com GCS...")
        consumer = GCSEmbeddingConsumer(
            bucket_name=bucket_name,
            embeddings_prefix=embeddings_prefix
        )
        
        print("   ✅ Cliente GCS inicializado")
        
        # 2. Testar busca de versão mais recente
        print("\n2. 🔍 Buscando versão mais recente...")
        files = consumer.find_latest_embeddings_pair()
        
        print(f"   ✅ Versão encontrada: {files['timestamp']}")
        print(f"   📁 Embeddings: {files['embeddings_filename']}")
        print(f"   📁 Índice: {files['index_filename']}")
        
        # 3. Testar carregamento
        print("\n3. 📥 Carregando embeddings...")
        if consumer.load_latest_embeddings():
            stats = consumer.get_stats()
            print(f"   ✅ Embeddings carregados!")
            print(f"   Shape: {stats['embeddings']['shape']}")
            print(f"   Índice: {stats['index']['size']} vetores")
            print(f"   Memória: ~{stats['embeddings']['size_mb']:.1f}MB")
            
            # 4. Testar busca
            print("\n4. 🔍 Testando busca semântica...")
            import numpy as np
            dummy_embedding = np.random.randn(1, 384).astype('float32')
            
            indices, distances = consumer.semantic_search(dummy_embedding, k=3)
            print(f"   ✅ Busca funcionando: {len(indices)} resultados")
            
            # 5. Verificar atualização
            print("\n5. 📡 Verificando atualizações...")
            has_update = consumer.check_for_new_version()
            if has_update:
                print("   ⚠️  Há versão mais nova disponível")
            else:
                print("   ✅ Usando versão mais recente")
            
            print("\n🎉 Consumidor GCS testado com sucesso!")
            print("   Modo: Leitura pura do bucket")
            print("   Nada é salvo localmente")
            
            return True
        else:
            print("❌ Falha ao carregar embeddings")
            return False
        
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)