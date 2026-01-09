#!/usr/bin/env python3
import os
import sys

# Adicionar apenas o diretório services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services'))

# Importar apenas o necessário
try:
    from services.gcs_consumer_service import GCSEmbeddingConsumer
    print("✅ Importação bem-sucedida!")
    
    # Testar criação
    consumer = GCSEmbeddingConsumer()
    print("✅ Consumidor criado com sucesso!")
    
except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()