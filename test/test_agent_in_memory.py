# test_agent_in_memory.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services.agent_service import BookAgentService
import logging

logging.basicConfig(level=logging.INFO)

def test_agent_memory():
    print("🧪 TESTANDO O AGENTE QUE A API REALMENTE USA")
    print("="*60)
    
    # 1. Criar agente IGUAL ao que a API usa
    print("\n🤖 1. INICIALIZANDO AGENTE...")
    agent = BookAgentService(config={})
    
    try:
        agent.initialize()
        print(f"✅ Agente inicializado: {agent.initialized}")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    # 2. Verificar dados carregados
    print("\n📊 2. DADOS CARREGADOS NO AGENTE:")
    
    if agent.data_loader and agent.data_loader.data is not None:
        data = agent.data_loader.data
        print(f"   Total livros: {len(data)}")
        
        # BUSCAR Six of Crows NO DATASET DO AGENTE
        six_mask = data['title'].str.contains('Six of Crows', case=False, na=False)
        if six_mask.any():
            six_row = data[six_mask].iloc[0]
            print(f"   ✅ 'Six of Crows' encontrado: ID {six_row.get('bookid')}")
        else:
            print(f"   ❌ 'Six of Crows' NÃO encontrado no dataset do agente!")
        
        # BUSCAR Harry Potter Series Box Set
        hp_mask = data['title'].str.contains('Harry Potter Series Box Set', case=False, na=False)
        if hp_mask.any():
            hp_row = data[hp_mask].iloc[0]
            print(f"   📚 'Harry Potter Series Box Set': ID {hp_row.get('bookid')}")
        
        # Mostrar primeiros IDs
        print(f"   📋 Primeiros 5 IDs: {list(data['bookid'].head().values)}")
        
        # TESTE CRÍTICO: Ver se há duplicatas no ID 409
        print(f"\n🔍 3. VERIFICANDO ID 409 ESPECÍFICO:")
        books_id_409 = data[data['bookid'] == 409]
        print(f"   Livros com ID 409: {len(books_id_409)}")
        
        if len(books_id_409) > 0:
            for idx, row in books_id_409.iterrows():
                print(f"      - Título: '{row['title']}'")
                print(f"        Autores: {row.get('author', 'N/A')}")
        
        # Verificar se há múltiplos Harry Potters
        print(f"\n🔍 4. TODOS OS HARRY POTTERS NO DATASET:")
        hp_all = data[data['title'].str.contains('Harry Potter', case=False, na=False)]
        print(f"   Total Harry Potters encontrados: {len(hp_all)}")
        
        for idx, row in hp_all.head(10).iterrows():
            print(f"      ID {row.get('bookid')}: '{row['title']}'")
    
    # 3. Verificar busca semântica DIRETAMENTE
    print("\n🔍 5. TESTE DE BUSCA SEMÂNTICA DIRETA:")
    
    if agent.search_engine:
        # Buscar "Six of Crows" semanticamente
        print("   Buscando 'Six of Crows'...")
        results = agent.search_engine.search_by_semantic("Six of Crows", k=5)
        
        print(f"   Resultados: {len(results)} livros")
        for i, book in enumerate(results[:3]):
            print(f"      {i+1}. ID {book.book_id}: '{book.title}'")
            print(f"         Similaridade: {book.similarity_score:.3f}")
    
    print("\n" + "="*60)
    print("🎯 DIAGNÓSTICO FINAL:")

if __name__ == "__main__":
    test_agent_memory()