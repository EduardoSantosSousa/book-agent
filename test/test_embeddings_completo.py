# test_embeddings_completo.py
import logging
import sys
from services.embedding_service import EmbeddingService

# Configurar logging para ver TUDO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    print("=" * 80)
    print("🧪 TESTE COMPLETO DO SISTEMA DE EMBEDDINGS")
    print("=" * 80)
    
    # 1. Inicializar serviço
    print("\n1️⃣ Inicializando EmbeddingService...")
    embedding_service = EmbeddingService()
    
    if not embedding_service.initialize():
        print("❌ Falha na inicialização!")
        return
    
    print("✅ Serviço inicializado com sucesso!")
    
    # 2. Mostrar estatísticas
    print("\n2️⃣ Estatísticas do serviço:")
    stats = embedding_service.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 3. Tentar verificar cobertura
    print("\n3️⃣ Verificando cobertura de embeddings...")
    
    # Lista de possíveis caminhos para o CSV
    csv_paths = [
        "exports/20260119_231738_EDU_books.csv",
        "20260119_231738_EDU_books.csv",
        "exports/EDU_books.csv",
        "EDU_books.csv"
    ]
    
    resultado = None
    for csv_path in csv_paths:
        print(f"\n   Tentando CSV: {csv_path}")
        resultado = embedding_service.verificar_livros_sem_embedding(csv_path)
        if resultado:
            print(f"   ✅ CSV encontrado: {csv_path}")
            break
    
    if resultado:
        print("\n" + "=" * 80)
        print("📊 RESUMO DA VERIFICAÇÃO:")
        print("=" * 80)
        print(f"   Total de livros: {resultado['total_livros_csv']}")
        print(f"   Com embedding:   {resultado['total_com_embedding']}")
        print(f"   Sem embedding:   {resultado['total_sem_embedding']}")
        print(f"   Cobertura:       {resultado['cobertura_percentual']:.2f}%")
        
        if resultado['ids_sem_embedding']:
            print(f"\n   Primeiros IDs sem embedding:")
            for i, book_id in enumerate(resultado['ids_sem_embedding'][:10], 1):
                print(f"     {i}. {book_id}")
    else:
        print("\n❌ Não foi possível verificar cobertura - CSV não encontrado")
        print("\n   📋 Arquivos disponíveis no bucket:")
        if embedding_service.gcs_consumer:
            blobs = list(embedding_service.gcs_consumer.bucket.list_blobs())
            for blob in blobs[:20]:  # Mostrar primeiros 20
                print(f"     - {blob.name}")
    
    # 4. Testar busca semântica
    print("\n4️⃣ Testando busca semântica...")
    query = "Dragon ball Z"
    indices, distances = embedding_service.semantic_search(query, k=5)
    
    print(f"   Query: '{query}'")
    print(f"   Resultados encontrados: {len(indices)}")
    
    for i, (idx, dist) in enumerate(zip(indices[:5], distances[:5]), 1):
        book_id = embedding_service.get_book_id_by_index(idx)
        print(f"     {i}. Índice: {idx}, Book ID: {book_id}, Distância: {dist:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ TESTE CONCLUÍDO!")
    print("=" * 80)

if __name__ == "__main__":
    main()