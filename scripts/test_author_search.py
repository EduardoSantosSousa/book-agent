# scripts/test_author_search.py
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.agent_service import BookAgentService

async def test_author_search():
    print("🔍 Testando busca por autor\n")
    
    # Inicializar agente
    agent = BookAgentService()
    agent.initialize()
    
    test_messages = [
        "Quais livros da autora 'J.K. Rowling' temos disponíveis?",
        "Livros do Stephen King",
        "Me recomende livros de George Orwell",
        "Autora Suzanne Collins",
        "Quem é o autor de The Hunger Games?",
        "Livros escritos por J.R.R. Tolkien",
    ]
    
    for message in test_messages:
        print(f"\n{'='*70}")
        print(f"📝 Mensagem: '{message}'")
        print('='*70)
        
        # Testar análise de intenção
        intent = agent._analyze_intent(message)
        print(f"🎯 Intenção detectada: {intent}")
        
        # Testar extração de autor
        author = agent._extract_author(message)
        print(f"✍️  Autor extraído: {author}")
        
        # Testar busca por autor no search_engine
        if author and intent == 'author':
            print(f"🔍 Buscando livros do autor: '{author}'")
            books = agent.search_engine.search_by_author(author, limit=5)
            print(f"📚 Resultados da busca direta: {len(books)} livros")
            
            if books:
                for i, book in enumerate(books[:3], 1):
                    print(f"   {i}. {book.title}")
                    print(f"      Autor(es): {', '.join(book.authors)}")
                    print(f"      Gêneros: {', '.join(book.genres)}")
        
        # Testar o process_message completo
        print("\n🧪 Testando process_message completo...")
        result = await agent.process_message(message, "author-test-session", "pt")
        print(f"   Intent final: {result.get('intent')}")
        print(f"   Livros encontrados: {result.get('books_found', 0)}")
        
        if result.get('books'):
            for i, book in enumerate(result['books'][:2], 1):
                print(f"   {i}. {book.get('title')}")
                print(f"      Autor(es): {', '.join(book.get('authors', []))}")

if __name__ == "__main__":
    asyncio.run(test_author_search())