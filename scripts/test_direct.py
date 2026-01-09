# scripts/test_direct.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.agent_service import BookAgentService
import asyncio

async def test_direct():
    print("🧪 Teste direto do agente...")
    
    agent = BookAgentService()
    agent.initialize()
    
    print(f"✅ Agente inicializado: {agent.initialized}")
    
    # Testar process_message diretamente
    try:
        print("\n💬 Testando process_message...")
        result = await agent.process_message(
            message="Livros de fantasia",
            session_id="test-direct",
            language="pt"
        )
        
        print(f"✅ Sucesso!")
        print(f"   Resposta: {result.get('response', '')[:100]}...")
        print(f"   Livros encontrados: {result.get('books_found', 0)}")
        
        if result.get('books'):
            print(f"\n📚 Primeiros livros:")
            for i, book in enumerate(result['books'][:3]):
                print(f"   {i+1}. {book.get('title')} - {book.get('rating')}/5")
                
    except Exception as e:
        print(f"❌ Erro no process_message: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct())