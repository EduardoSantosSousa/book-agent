# scripts/test_agent_directly.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.agent_service import BookAgentService
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agent_directly():
    print("🧪 Teste DIRETO do agente (mesmo processo que o Flask usaria)")
    print("=" * 60)
    
    try:
        # 1. Criar agente
        print("\n1. 🏗️  Criando BookAgentService...")
        agent = BookAgentService()
        
        # 2. Inicializar
        print("2. 🔄 Inicializando...")
        agent.initialize()
        agent.initialized = True
        
        print(f"   ✅ Initialized: {agent.initialized}")
        
        # 3. Verificar embedding_service
        print("\n3. 🧠 Verificando embedding_service...")
        if hasattr(agent, 'embedding_service'):
            emb_service = agent.embedding_service
            print(f"   ✅ embedding_service encontrado")
            
            # Verificar book_embeddings
            if hasattr(emb_service, 'book_embeddings'):
                if emb_service.book_embeddings is not None:
                    print(f"   ✅ book_embeddings carregado: shape={emb_service.book_embeddings.shape}")
                else:
                    print(f"   ❌ book_embeddings é None!")
            else:
                print(f"   ❌ Não tem atributo book_embeddings")
                
            # Verificar index
            if hasattr(emb_service, 'index'):
                if emb_service.index is not None:
                    print(f"   ✅ index carregado: ntotal={emb_service.index.ntotal}")
                else:
                    print(f"   ❌ index é None!")
            else:
                print(f"   ❌ Não tem atributo index")
        else:
            print(f"   ❌ Não tem embedding_service")
        
        # 4. Testar process_message
        print("\n4. 💬 Testando process_message...")
        try:
            result = await agent.process_message(
                message="Livros de programação Python",
                session_id="test-direct",
                language="pt"
            )
            
            print(f"   ✅ Sucesso!")
            print(f"   Response: {result.get('response', '')[:100]}...")
            print(f"   Books found: {result.get('books_found', 0)}")
            
            if result.get('books'):
                print(f"\n   📚 Primeiro livro:")
                book = result['books'][0]
                print(f"      Título: {book.get('title', 'N/A')}")
                print(f"      Autor(es): {', '.join(book.get('authors', []))}")
                print(f"      Gêneros: {', '.join(book.get('genres', []))}")
                print(f"      Rating: {book.get('rating', 'N/A')}")
                
        except Exception as e:
            print(f"   ❌ Erro no process_message: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("✅ Teste concluído!")
        
        # Manter referência para não ser garbage collected
        return agent
        
    except Exception as e:
        print(f"❌ Erro geral: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Manter o agente vivo
    agent = asyncio.run(test_agent_directly())
    
    if agent:
        print("\n💡 Se este teste funcionou mas o Flask não:")
        print("1. O problema é que o Flask está em processo diferente")
        print("2. Use a solução 'start_all_in_one.py'")
        
        # Manter processo vivo
        try:
            input("\nPressione Enter para finalizar...")
        except:
            pass