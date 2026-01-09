# scripts/start_server_with_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from services.agent_service import BookAgentService
from api.routes import get_agent_service
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pre_initialize_agent():
    """Pré-inicializa o agente antes do servidor Flask"""
    logger.info("🚀 Pré-inicializando BookAgentService...")
    
    try:
        # Cria e inicializa o agente
        agent = BookAgentService()
        agent.initialize()
        agent.initialized = True
        
        # Injeta no módulo routes
        import api.routes
        api.routes._agent_service = agent
        
        logger.info("✅ Agente pré-inicializado e injetado no módulo routes")
        
        # Verificar
        test_agent = get_agent_service()
        logger.info(f"✅ Teste get_agent_service(): {test_agent is not None}")
        
        return agent
        
    except Exception as e:
        logger.error(f"❌ Erro na pré-inicialização: {e}")
        return None

if __name__ == "__main__":
    # Pré-inicializar o agente
    agent = pre_initialize_agent()
    
    if agent:
        # Agora iniciar o servidor Flask
        logger.info("🌐 Iniciando servidor Flask...")
        
        from app import create_app
        app = create_app()
        
        # Iniciar servidor
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    else:
        logger.error("❌ Não foi possível iniciar o servidor devido a falha na inicialização do agente")
        sys.exit(1)