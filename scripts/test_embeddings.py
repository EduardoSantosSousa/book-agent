import os
import sys

# Adicionar diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config, EmbeddingLoader

print("=" * 60)
print("🔧 TESTE DE CARREGAMENTO DE EMBEDDINGS")
print("=" * 60)

print(f"Ambiente: {config.ENVIRONMENT}")
print(f"Fonte embeddings: {config.EMBEDDINGS_SOURCE}")
print(f"Caminho local: {config.LOCAL_EMBEDDINGS_PATH}")

# Verificar se os diretórios existem
print("\n📂 Verificando diretórios:")
print(f"  data/: {'✅ EXISTE' if os.path.exists('data') else '❌ NÃO EXISTE'}")
print(f"  embeddings/local/: {'✅ EXISTE' if os.path.exists('embeddings/local') else '❌ NÃO EXISTE'}")

# Verificar arquivos específicos
print("\n📄 Verificando arquivos específicos:")
print(f"  Dataset: {'✅ EXISTE' if os.path.exists('data/book_dataset_treated.csv') else '❌ NÃO EXISTE'}")
print(f"  Índice FAISS: {'✅ EXISTE' if os.path.exists(config.LOCAL_INDEX_FILE) else f'❌ NÃO EXISTE ({config.LOCAL_INDEX_FILE})'}")
print(f"  Embeddings numpy: {'✅ EXISTE' if os.path.exists(config.LOCAL_EMBEDDINGS_FILE) else f'❌ NÃO EXISTE ({config.LOCAL_EMBEDDINGS_FILE})'}")

# Tentar listar o que existe em embeddings/local
if os.path.exists('embeddings/local'):
    print(f"\n📁 Conteúdo de embeddings/local:")
    for file in os.listdir('embeddings/local'):
        size = os.path.getsize(f'embeddings/local/{file}') / (1024*1024)  # MB
        print(f"  - {file} ({size:.1f} MB)")

print("\n" + "=" * 60)
print("🚀 Tentando carregar embeddings...")
print("=" * 60)

try:
    index, embeddings = EmbeddingLoader.load_index_files(config)
    print(f"\n🎉 SUCESSO! Embeddings carregados:")
    print(f"   Forma: {embeddings.shape}")
    print(f"   Tipo: {embeddings.dtype}")
    print(f"   Tamanho do índice: {index.ntotal}")
    
except Exception as e:
    print(f"\n❌ FALHA: {e}")
    
    # Sugestões
    print("\n💡 SUGESTÕES:")
    print("1. Verifique se os arquivos existem:")
    print("   - embeddings/local/book_index_gpu_index.faiss")
    print("   - embeddings/local/book_index_gpu_embeddings.npy")
    print("\n2. Se não existirem, você pode:")
    print("   a) Copiá-los do notebook/ para embeddings/local/")
    print("   b) Criar embeddings mock com o script abaixo")

print("\n" + "=" * 60)