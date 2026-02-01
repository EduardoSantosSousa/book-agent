# check_api_singleton.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Importar o MESMO módulo que a API usa
import api.routes

print("🔍 VERIFICANDO O SINGLETON DA API")
print("="*50)

# Verificar se há um singleton global
if hasattr(api.routes, '_agent_service'):
    agent = api.routes._agent_service
    print(f"✅ Singleton encontrado na API: {agent is not None}")
    
    if agent and hasattr(agent, 'data_loader') and agent.data_loader.data is not None:
        data = agent.data_loader.data
        print(f"📊 Dataset no singleton da API: {len(data)} livros")
        
        # Buscar os mesmos IDs
        print(f"\n🔍 BUSCANDO OS MESMOS IDs NO SINGLETON:")
        
        test_ids = [13817, 409, 1600, 11030]
        for test_id in test_ids:
            book_row = data[data['bookid'] == test_id]
            if not book_row.empty:
                book = book_row.iloc[0]
                print(f"   ✅ ID {test_id}: '{book['title']}'")
            else:
                print(f"   ❌ ID {test_id}: NÃO ENCONTRADO")
        
        # Mostrar origem
        print(f"\n📁 ORIGEM DO DATASET NO SINGLETON:")
        
        # Verificar se está usando arquivo local
        local_files = ['data/book_dataset_treated.csv', 'api/book_dataset_treated.csv']
        for file in local_files:
            if os.path.exists(file):
                print(f"   ⚠️  Arquivo local encontrado: {file}")
                print(f"      Tamanho: {os.path.getsize(file) / (1024*1024):.1f} MB")
                
                # Ler algumas linhas
                import pandas as pd
                try:
                    sample = pd.read_csv(file, nrows=3)
                    print(f"      Primeiros IDs: {list(sample.get('bookId', sample.get('bookid', 'N/A')))}")
                except:
                    pass
else:
    print("❌ Nenhum singleton encontrado em api.routes")

print("\n" + "="*50)
print("🎯 DIAGNÓSTICO:")