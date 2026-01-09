# D:\Django\book_agent\api\middleware.py
from flask import request, jsonify
import time
import logging

logger = logging.getLogger(__name__)

def setup_middleware(app):
    """Configura middleware para a aplicação Flask"""
    
    @app.before_request
    def before_request():
        """Middleware executado antes de cada requisição"""
        request.start_time = time.time()
        
        # Log da requisição
        if request.path not in ['/health', '/favicon.ico']:
            logger.info(f"Requisição: {request.method} {request.path}")
    
    @app.after_request
    def after_request(response):
        """Middleware executado após cada requisição"""
        # Adicionar headers de segurança
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Log do tempo de resposta
        if hasattr(request, 'start_time'):
            processing_time = time.time() - request.start_time
            response.headers['X-Processing-Time'] = str(round(processing_time, 3))
            
            if request.path not in ['/health', '/favicon.ico']:
                logger.info(f"Resposta: {request.method} {request.path} - {response.status_code} ({processing_time:.3f}s)")
        
        return response
    
    @app.errorhandler(429)
    def ratelimit_handler(e):
        """Handler para limite de requisições"""
        return jsonify({
            'success': False,
            'error': 'Limite de requisições excedido',
            'message': 'Por favor, aguarde antes de fazer novas requisições'
        }), 429
    
    @app.errorhandler(405)
    def method_not_allowed(e):
        """Handler para método não permitido"""
        return jsonify({
            'success': False,
            'error': 'Método não permitido',
            'message': f'O método {request.method} não é suportado para este endpoint'
        }), 405