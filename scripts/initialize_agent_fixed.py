# scripts/initialize_agent_fixed.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from services.agent_service import BookAgentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_and_inspect():
    """Inicializa e inspeciona o agente completamente"""
    logger.info("🚀 Inicializando BookAgentService...")
    
    try:
        start_time = time.time()
        
        agent = BookAgentService()
        logger.info("✅ Instância criada")
        
        logger.info("🔄 Inicializando serviços...")
        agent.initialize()
        
        # Marcar como inicializado
        agent.initialized = True
        
        elapsed = time.time() - start_time
        logger.info(f"✅ BookAgentService inicializado em {elapsed:.2f} segundos")
        
        # INSPEÇÃO DETALHADA
        print("\n" + "="*60)
        print("🔍 INSPEÇÃO DETALHADA DO AGENTE")
        print("="*60)
        
        # 1. Verificar atributos principais
        print("\n1. 📋 ATRIBUTOS PRINCIPAIS:")
        print(f"   initialized: {getattr(agent, 'initialized', False)}")
        
        # 2. Verificar embedding_service
        print("\n2. 🧠 EMBEDDING SERVICE:")
        if hasattr(agent, 'embedding_service') and agent.embedding_service:
            emb_service = agent.embedding_service
            print(f"   ✅ Embedding Service disponível")
            
            # Verificar atributos do embedding_service
            emb_attrs = [attr for attr in dir(emb_service) if not attr.startswith('_')]
            print(f"   Atributos: {emb_attrs[:10]}...")  # Primeiros 10
            
            # Procurar embeddings
            for attr in emb_attrs:
                try:
                    attr_val = getattr(emb_service, attr)
                    if hasattr(attr_val, 'shape'):  # numpy array
                        print(f"   ✅ Embeddings encontrados em embedding_service.{attr}: shape={attr_val.shape}")
                    if hasattr(attr_val, 'ntotal'):  # faiss index
                        print(f"   ✅ Índice FAISS encontrado em embedding_service.{attr}: ntotal={attr_val.ntotal}")
                except:
                    pass
        else:
            print("   ❌ Embedding Service não encontrado")
        
        # 3. Verificar search_engine
        print("\n3. 🔍 SEARCH ENGINE:")
        if hasattr(agent, 'search_engine') and agent.search_engine:
            print(f"   ✅ Search Engine disponível")
            # Verificar se tem livros
            if hasattr(agent.search_engine, 'books_data'):
                print(f"   📚 Livros no search_engine: {len(agent.search_engine.books_data)}")
            else:
                print(f"   🔍 Procurando livros no search_engine...")
                se_attrs = [attr for attr in dir(agent.search_engine) if not attr.startswith('_')]
                for attr in se_attrs:
                    try:
                        attr_val = getattr(agent.search_engine, attr)
                        if isinstance(attr_val, list) and len(attr_val) > 0:
                            if isinstance(attr_val[0], dict) and 'title' in attr_val[0]:
                                print(f"   ✅ Livros encontrados em search_engine.{attr}: {len(attr_val)}")
                    except:
                        pass
        else:
            print("   ❌ Search Engine não encontrado")
        
        # 4. Verificar data_loader
        print("\n4. 📊 DATA LOADER:")
        if hasattr(agent, 'data_loader') and agent.data_loader:
            print(f"   ✅ Data Loader disponível")
            # Verificar dataset
            dl_attrs = [attr for attr in dir(agent.data_loader) if not attr.startswith('_')]
            for attr in dl_attrs:
                try:
                    attr_val = getattr(agent.data_loader, attr)
                    if isinstance(attr_val, list) and len(attr_val) > 1000:  # Provavelmente é o dataset
                        print(f"   ✅ Dataset encontrado em data_loader.{attr}: {len(attr_val)} registros")
                except:
                    pass
        else:
            print("   ❌ Data Loader não encontrado")
        
        # 5. Testar funcionalidades
        print("\n5. 🧪 TESTANDO FUNCIONALIDADES:")
        
        # Testar busca
        if hasattr(agent, 'search_books'):
            try:
                from models.schemas import SearchRequest
                print("   🔎 Testando busca...")
                search_req = SearchRequest(query="historia", limit=5)
                results = agent.search_books(search_req)
                print(f"   ✅ Busca funcionou: {len(results)} resultados")
                if results:
                    print(f"   📖 Primeiro livro: {results[0].get('title', 'N/A')}")
            except Exception as e:
                print(f"   ❌ Busca falhou: {str(e)[:100]}")
        
        # Testar get_book_by_id
        if hasattr(agent, 'get_book_by_id'):
            try:
                print("   🔢 Testando busca por ID...")
                book = agent.get_book_by_id(1)  # ID 1
                if book:
                    print(f"   ✅ Livro ID 1 encontrado: {book.get('title', 'N/A')}")
                else:
                    print("   ⚠️ Livro ID 1 não encontrado")
            except Exception as e:
                print(f"   ❌ Busca por ID falhou: {str(e)[:100]}")
        
        print("\n" + "="*60)
        print("🎉 INSPEÇÃO CONCLUÍDA")
        print("="*60)
        
        return agent
        
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    agent = initialize_and_inspect()
    if agent:
        print("\n✅ Agente está pronto para uso!")
        print("\n💡 Para usar:")
        print("1. Mantenha este processo rodando")
        print("2. Em OUTRO terminal, execute: python app.py")
        print("3. Teste com: python scripts/test_chat_immediate.py")
        
        # Manter o processo vivo
        try:
            input("\nPressione Enter para finalizar...")
        except:
            pass
    else:
        print("\n❌ Falha na inicialização do agente.")
        sys.exit(1)