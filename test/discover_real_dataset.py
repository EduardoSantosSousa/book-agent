# discover_real_dataset.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flask import Flask
import json

def discover_real_dataset():
    print("🔍 DESCOBRINDO QUAL DATASET A API REALMENTE USA")
    print("="*60)
    
    try:
        # 1. Importar app para acessar o agente REAL
        from app import app
        
        with app.app_context():
            agent = app.config.get('agent_service')
            
            if not agent:
                print("❌ Agente não encontrado no app")
                return
            
            print(f"✅ Agente encontrado: {agent.initialized}")
            
            # 2. Verificar dados do agente REAL
            if agent.data_loader and agent.data_loader.data is not None:
                data = agent.data_loader.data
                print(f"\n📊 DADOS NO AGENTE DA API:")
                print(f"   Total livros: {len(data)}")
                print(f"   Colunas: {list(data.columns)[:10]}...")
                
                # 3. Buscar os livros que a API retornou
                print(f"\n🔍 BUSCANDO LIVROS QUE A API RETORNOU:")
                
                api_book_ids = [13817, 409, 1600, 11030, 13472, 3576, 14486, 42889]
                
                for api_id in api_book_ids:
                    # Buscar no dataset do agente
                    book_row = data[data['bookid'] == api_id]
                    
                    if not book_row.empty:
                        book = book_row.iloc[0]
                        print(f"   ✅ ID {api_id}: '{book['title']}'")
                    else:
                        print(f"   ❌ ID {api_id}: NÃO ENCONTRADO no dataset do agente!")
                
                # 4. Verificar origem dos dados
                print(f"\n📁 ORIGEM DOS DADOS:")
                
                # Verificar se há um caminho local
                if hasattr(agent.data_loader, 'data_path') and agent.data_loader.data_path:
                    print(f"   Caminho local: {agent.data_loader.data_path}")
                    
                    # Verificar se o arquivo existe
                    if os.path.exists(agent.data_loader.data_path):
                        print(f"   ✅ Arquivo existe localmente")
                        # Ler primeiras linhas
                        import pandas as pd
                        local_data = pd.read_csv(agent.data_loader.data_path, nrows=5)
                        print(f"   📋 Primeiros títulos locais:")
                        for idx, row in local_data.iterrows():
                            print(f"      ID {row.get('bookId', 'N/A')}: {row.get('title', 'N/A')}")
                
                # 5. Mostrar primeiros IDs do dataset REAL
                print(f"\n📋 PRIMEIROS 10 IDs DO DATASET REAL:")
                first_ids = list(data['bookid'].head(10).values)
                for i, book_id in enumerate(first_ids):
                    book = data[data['bookid'] == book_id].iloc[0]
                    print(f"   {i+1:2d}. ID {book_id}: '{book['title'][:40]}...'")
                
                # 6. Buscar Six of Crows no dataset REAL
                print(f"\n🔍 SIX OF CROWS NO DATASET REAL:")
                six_mask = data['title'].str.contains('Six of Crows', case=False, na=False)
                if six_mask.any():
                    six_row = data[six_mask].iloc[0]
                    print(f"   ✅ Encontrado: ID {six_row['bookid']} - '{six_row['title']}'")
                else:
                    print(f"   ❌ NÃO ENCONTRADO!")
                
                # 7. Salvar amostra do dataset REAL
                print(f"\n💾 SALVANDO AMOSTRA DO DATASET REAL...")
                sample = data.head(20)[['bookid', 'title', 'author']]
                sample_file = 'real_dataset_sample.json'
                sample.to_json(sample_file, orient='records', indent=2)
                print(f"   ✅ Amostra salva em: {sample_file}")
                
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("🎯 CONCLUSÃO:")

if __name__ == "__main__":
    discover_real_dataset()