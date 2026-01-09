# scripts/test_final_genre.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.agent_service import BookAgentService

def test_final():
    print("🔍 Teste final de busca por gênero")
    
    # Inicializar agente
    agent = BookAgentService()
    agent.initialize()
    
    # Testar
    test_message = "Recomende livros de fantasia para iniciantes"
    
    print(f"\n📝 Mensagem: '{test_message}'")
    
    # Extrair gênero
    genre = agent._extract_genre(test_message)
    print(f"🎯 Gênero extraído: {genre}")
    
    # Buscar
    if genre:
        print(f"🔍 Buscando por gênero: '{genre}'")
        books = agent.search_engine.search_by_genre(genre, limit=5)
        print(f"📚 Resultados: {len(books)} livros")
        
        if books:
            for i, book in enumerate(books, 1):
                print(f"\n   {i}. {book.title}")
                print(f"      Gêneros: {', '.join(book.genres)}")
                print(f"      Autor(es): {', '.join(book.authors)}")
                print(f"      Avaliação: {book.rating:.1f}")
        else:
            print("❌ Nenhum livro encontrado!")
    
    # Testar com o process_message completo
    print("\n" + "="*60)
    print("🧪 Testando process_message completo...")
    
    import asyncio
    async def test_process():
        result = await agent.process_message(test_message, "test-session", "pt")
        print(f"🎯 Intent: {result.get('intent')}")
        print(f"📚 Livros encontrados: {result.get('books_found', 0)}")
        if result.get('books'):
            for i, book in enumerate(result['books'][:3], 1):
                print(f"   {i}. {book.get('title')}")
    
    asyncio.run(test_process())

if __name__ == "__main__":
    test_final()