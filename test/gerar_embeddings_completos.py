# gerar_embeddings_completos.py
import logging
import sys
from services.embedding_generator import EmbeddingGenerator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    print("=" * 80)
    print("🚀 GERADOR DE EMBEDDINGS - VERSÃO COMPLETA")
    print("=" * 80)
    
    # Inicializar gerador
    generator = EmbeddingGenerator(
        bucket_name="book-agent-embeddings-bucket",
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        use_gpu=True  # Mude para False se não tiver GPU
    )
    
    # Executar pipeline completo
    # Você pode especificar um CSV específico ou deixar None para pegar o mais recente
    sucesso = generator.run_complete_pipeline(
        csv_path="exports/20260119_231738_EDU_books.csv"  # ou None para automático
    )
    
    if sucesso:
        print("\n✅ PIPELINE EXECUTADO COM SUCESSO!")
        print(f"📚 Total de livros processados: {len(generator.df)}")
        print(f"📊 Shape dos embeddings: {generator.embeddings.shape}")
        print(f"📋 Metadados gerados: {len(generator.metadata)} registros")
        print("\n🎉 Agora você tem embeddings e metadados COMPLETOS no GCS!")
    else:
        print("\n❌ Falha na execução do pipeline")

if __name__ == "__main__":
    main()