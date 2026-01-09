# scripts/initialize_agent.py - Versão corrigida
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from services.agent_service import BookAgentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_agent_on_startup():
    """Inicializa o agente no startup da aplicação"""
    logger.info("🚀 Inicializando BookAgentService no startup...")
    
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
        
        # Verificar componentes - PROCURANDO EM TODOS OS LUGARES
        logger.info("🔍 Verificando componentes (busca profunda)...")
        
        # Lista todos os atributos do agente
        all_attrs = [attr for attr in dir(agent) if not attr.startswith('_')]
        logger.info(f"   Atributos do agente: {all_attrs}")
        
        # Verificar cada atributo importante
        checks = {}
        
        # Procurar por books_data
        if hasattr(agent, 'books_data'):
            checks['Dataset'] = len(agent.books_data) > 0
            logger.info(f"   books_data encontrado: {len(agent.books_data)} livros")
        else:
            # Procurar em subatributos
            for attr_name in all_attrs:
                attr = getattr(agent, attr_name)
                if hasattr(attr, 'books_data'):
                    checks['Dataset'] = len(attr.books_data) > 0
                    logger.info(f"   books_data em {attr_name}: {len(attr.books_data)} livros")
                    break
        
        # Procurar por embeddings
        if hasattr(agent, 'embeddings') and agent.embeddings is not None:
            checks['Embeddings'] = True
            logger.info(f"   embeddings direto: shape {agent.embeddings.shape}")
        elif hasattr(agent, 'embedding_service') and hasattr(agent.embedding_service, 'embeddings'):
            checks['Embeddings'] = agent.embedding_service.embeddings is not None
            logger.info(f"   embeddings em embedding_service: shape {agent.embedding_service.embeddings.shape}")
        elif hasattr(agent, 'gcs_consumer') and hasattr(agent.gcs_consumer, 'current_embeddings'):
            checks['Embeddings'] = agent.gcs_consumer.current_embeddings is not None
            logger.info(f"   embeddings em gcs_consumer: shape {agent.gcs_consumer.current_embeddings.shape}")
        
        # Procurar por faiss_index
        if hasattr(agent, 'faiss_index') and agent.faiss_index is not None:
            checks['FAISS Index'] = True
            logger.info(f"   faiss_index direto: {agent.faiss_index.ntotal} vetores")
        elif hasattr(agent, 'embedding_service') and hasattr(agent.embedding_service, 'faiss_index'):
            checks['FAISS Index'] = agent.embedding_service.faiss_index is not None
            logger.info(f"   faiss_index em embedding_service: {agent.embedding_service.faiss_index.ntotal} vetores")
        elif hasattr(agent, 'gcs_consumer') and hasattr(agent.gcs_consumer, 'current_index'):
            checks['FAISS Index'] = agent.gcs_consumer.current_index is not None
            logger.info(f"   faiss_index em gcs_consumer: {agent.gcs_consumer.current_index.ntotal} vetores")
        
        # Procurar por gcs_consumer
        checks['GCS Consumer'] = hasattr(agent, 'gcs_consumer') and agent.gcs_consumer is not None
        
        # Imprimir resumo
        logger.info("\n📋 Resumo:")
        for component, status in checks.items():
            logger.info(f"   {component}: {'✅' if status else '❌'}")
        
        # Se encontrou embeddings e índice, mostrar estatísticas
        if checks.get('Embeddings') and checks.get('FAISS Index'):
            logger.info("\n📊 Estatísticas detalhadas:")
            
            # Encontrar onde estão os embeddings
            if hasattr(agent, 'embeddings'):
                logger.info(f"   Embeddings shape: {agent.embeddings.shape}")
            elif hasattr(agent, 'embedding_service') and hasattr(agent.embedding_service, 'embeddings'):
                logger.info(f"   Embeddings shape: {agent.embedding_service.embeddings.shape}")
            elif hasattr(agent, 'gcs_consumer') and hasattr(agent.gcs_consumer, 'current_embeddings'):
                logger.info(f"   Embeddings shape: {agent.gcs_consumer.current_embeddings.shape}")
            
            # Encontrar onde está o índice
            if hasattr(agent, 'faiss_index'):
                logger.info(f"   Índice size: {agent.faiss_index.ntotal}")
            elif hasattr(agent, 'embedding_service') and hasattr(agent.embedding_service, 'faiss_index'):
                logger.info(f"   Índice size: {agent.embedding_service.faiss_index.ntotal}")
            elif hasattr(agent, 'gcs_consumer') and hasattr(agent.gcs_consumer, 'current_index'):
                logger.info(f"   Índice size: {agent.gcs_consumer.current_index.ntotal}")
        
        logger.info(f"\n📊 Total de livros: {checks.get('Dataset', False)}")
        
        return agent
        
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    agent = initialize_agent_on_startup()
    if agent:
        print("\n🎉 Agente inicializado com sucesso!")
        print("   O servidor pode ser iniciado agora.")
    else:
        print("\n❌ Falha na inicialização do agente.")
        sys.exit(1)