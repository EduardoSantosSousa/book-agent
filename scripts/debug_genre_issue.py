# scripts/debug_genre_issue.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.agent_service import BookAgentService
import asyncio

def debug_genre_issue():
    print("🔍 Debug do problema de busca por gênero")
    
    # Inicializar agente
    agent = BookAgentService()
    agent.initialize()
    
    # Testar mensagem específica
    test_message = "Recomende livros de fantasia para iniciantes"
    
    print(f"\n📝 Mensagem de teste: '{test_message}'")
    
    # Testar extração de gênero
    print("\n1. 🔍 Testando extração de gênero...")
    genre = agent._extract_genre(test_message)
    print(f"   Gênero extraído: {genre}")
    
    # Testar extração de autor
    print("\n2. 🔍 Testando extração de autor...")
    author = agent._extract_author(test_message)
    print(f"   Autor extraído: {author}")
    
    # Testar intenção
    print("\n3. 🎯 Testando detecção de intenção...")
    intent = agent._analyze_intent(test_message)
    print(f"   Intenção detectada: {intent}")
    
    # Testar busca por gênero diretamente
    print("\n4. 📚 Testando busca direta por gênero...")
    if genre:
        print(f"   Buscando livros do gênero: '{genre}'")
        try:
            # Usar o search_engine diretamente
            books = agent.search_engine.search_by_genre(genre, limit=5)
            print(f"   Livros encontrados: {len(books)}")
            
            if books:
                print(f"\n   📖 Primeiros 3 livros:")
                for i, book in enumerate(books[:3], 1):
                    print(f"   {i}. {book.title}")
                    print(f"      Gêneros: {', '.join(book.genres)}")
                    print(f"      Autor(es): {', '.join(book.authors)}")
            else:
                print("   ❌ Nenhum livro encontrado!")
                
                # Verificar quais gêneros existem
                print(f"\n   🔍 Verificando gêneros disponíveis...")
                # Pegar alguns exemplos de gêneros do dataset
                sample_genres = set()
                for idx, row in agent.data_loader.data.head(20).iterrows():
                    main_genre = row.get('main_genre')
                    if pd.notnull(main_genre):
                        sample_genres.add(str(main_genre))
                print(f"   Exemplos de gêneros: {', '.join(list(sample_genres)[:10])}")
        
        except Exception as e:
            print(f"   ❌ Erro na busca: {e}")
    
    print("\n5. 🔧 Verificando estrutura do search_engine...")
    print(f"   Search engine disponível: {agent.search_engine is not None}")
    if agent.search_engine:
        print(f"   Data shape: {agent.data_loader.data.shape}")
        print(f"   Colunas disponíveis: {list(agent.data_loader.data.columns[:10])}")

if __name__ == "__main__":
    import pandas as pd
    debug_genre_issue()