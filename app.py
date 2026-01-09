# app.py - VERSÃO SIMPLIFICADA E SEGURA
from flask import Flask, jsonify, g
from flask_cors import CORS
import logging
from api.routes import api_bp
from api.book_conversation_routes import book_conv_bp
from api.consumer_routes import consumer_bp
from api.middleware import setup_middleware
from api.routes import get_agent_service
import os
import time
import atexit
from dotenv import load_dotenv

load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app():
    """Factory function para criar a aplicação Flask - VERSÃO SIMPLIFICADA"""
    app = Flask(__name__)
    
    # Configurações
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        JSON_SORT_KEYS=False,
        THREADED=True
    )
    
    # Configurar CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Setup middleware
    setup_middleware(app)
    
   
    # ==============================================
    # INICIALIZAÇÃO SIMPLES DO AGENTE
    # ==============================================
    
    logger.info("🚀 Inicializando BookAgentService...")
    try:
        # Inicialização DIRETA sem loops complexos
        from services.agent_service import BookAgentService
        
        # Cria e inicializa o agente
        agent = BookAgentService(config={})
        success = agent.initialize()  # Método síncrono
        
        if success:
            # Armazena no app context
            app.config['agent_service'] = agent
            logger.info("✅ BookAgentService inicializado com sucesso")
            logger.info(f"   Total livros: {len(agent.data_loader.data)}")
        else:
            logger.error("❌ Falha na inicialização do agente")
            app.config['agent_service'] = None
            
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar agente: {e}", exc_info=True)
        app.config['agent_service'] = None
    
    # ==============================================
    # MODIFICAÇÃO DA FUNÇÃO get_agent_service PARA USAR app.config
    # ==============================================
    
    # Importante: Precisamos sobrescrever a função get_agent_service
    # para usar o app.config em vez de singleton global
    import api.routes
    
    def get_app_agent_service():
        """Obtém o agente do contexto do app"""
        from flask import current_app
        return current_app.config.get('agent_service')
    
    # Substitui a função original
    api.routes.get_agent_service = get_app_agent_service
    
    # ==============================================
    # REGISTRAR BLUEPRINTS
    # ==============================================
    
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    app.register_blueprint(book_conv_bp, url_prefix='/api/v1')
    app.register_blueprint(consumer_bp, url_prefix='/api/v1')
    
    # ==============================================
    # ROTAS BÁSICAS
    # ==============================================
    
    @app.route('/health', methods=['GET'])
    def health_check():
        agent = app.config.get('agent_service')
        return jsonify({
            'status': 'healthy' if agent and agent.initialized else 'degraded',
            'agent_initialized': bool(agent and agent.initialized)
        })
    
    @app.route('/', methods=['GET'])
    def index():
        return jsonify({
            'service': 'Book Agent API',
            'version': '1.0.0',
            'status': 'running'
        })
    
    # ==============================================
    # FUNÇÃO DE LIMPEZA
    # ==============================================
    
    def cleanup():
        logger.info("🧹 Limpando recursos...")
        agent = app.config.get('agent_service')
        if agent:
            # Fecha conexões se necessário
            pass
    
    atexit.register(cleanup)
    
    return app

# ==============================================
# EXECUÇÃO
# ==============================================

app = create_app()

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 8080))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Iniciando Book Agent API em {host}:{port}")
    
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        logger.info("👋 Aplicação encerrada")
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        raise