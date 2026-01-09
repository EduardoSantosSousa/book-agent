# D:\Django\book_agent\api\routes.py

from flask import Blueprint, request, jsonify
import asyncio
import logging
from functools import wraps
from models.schemas import ChatRequest, SearchRequest
from services.agent_service import BookAgentService
from utils.validators import validate_request
import threading
import time
from services.translation_service import get_translation_service
from config import config
from services.gcs_consumer_service import GCSEmbeddingConsumer
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)

# Instância do serviço (singleton)
_agent_service = None
_agent_lock = threading.Lock()

def get_agent_service():
    """Obter instância singleton do serviço - AGORA SÍNCRONO"""
    global _agent_service
    
    if _agent_service is None:
        with _agent_lock:
            if _agent_service is None:
                logger.info("🔄 Inicializando BookAgentService...")
                _agent_service = BookAgentService()
                
                try:
                    # Inicialização SÍNCRONA (bloqueante)
                    logger.info("🔄 Inicializando serviços do agente...")
                    _agent_service.initialize()
                    logger.info("✅ BookAgentService inicializado com sucesso!")
                    
                    # Verificar se está realmente pronto
                    if not getattr(_agent_service, 'initialized', False):
                        logger.warning("⚠️ Agente criado mas não marcado como 'initialized'")
                        _agent_service.initialized = True  # Força marcação
                        
                except Exception as e:
                    logger.error(f"❌ Erro na inicialização do agente: {e}")
                    _agent_service = None  # Reseta para tentar novamente
                    raise
    
    return _agent_service

def async_handler(f):
    """Decorator para lidar com funções assíncronas"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

# ============================================================
# ROTAS PRINCIPAIS
# ============================================================

# routes.py - Atualize o endpoint /chat
# routes.py - Adicione logs de debug no início do endpoint /chat

@api_bp.route('/chat', methods=['POST'])
async def chat():
    """
    Processa uma mensagem do usuário
    """
    try:
        # Validar e parsear requisição
        data = validate_request(ChatRequest, request.json)
        
        logger.info(f"📨 Processando mensagem: '{data.message[:50]}...' - Sessão: {data.session_id}")
        
        # Processar mensagem
        agent_service = get_agent_service()
        
        # DEBUG DETALHADO
        logger.info(f"🔍 DEBUG - Estado do agente:")
        logger.info(f"   Agent service: {agent_service}")
        logger.info(f"   Initialized: {getattr(agent_service, 'initialized', 'ATTR_NOT_FOUND')}")
        logger.info(f"   Tem embedding_service: {hasattr(agent_service, 'embedding_service')}")
        
        if hasattr(agent_service, 'embedding_service'):
            emb_service = agent_service.embedding_service
            logger.info(f"   Embedding service: {emb_service}")
            logger.info(f"   Tem book_embeddings: {hasattr(emb_service, 'book_embeddings')}")
            if hasattr(emb_service, 'book_embeddings'):
                logger.info(f"   Book embeddings shape: {emb_service.book_embeddings.shape if emb_service.book_embeddings is not None else 'None'}")
            logger.info(f"   Tem index: {hasattr(emb_service, 'index')}")
            if hasattr(emb_service, 'index'):
                logger.info(f"   Index ntotal: {emb_service.index.ntotal if emb_service.index is not None else 'None'}")
        
        # Verificar se o serviço está inicializado
        if not agent_service or not getattr(agent_service, 'initialized', False):
            logger.warning("⚠️ Agente não inicializado, tentando inicializar...")
            
            # Tentar inicializar agora
            try:
                agent_service.initialize()
                agent_service.initialized = True
                logger.info("✅ Agente inicializado na requisição")
            except Exception as e:
                logger.error(f"❌ Falha na inicialização: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Agente ainda não está pronto. Por favor, aguarde.',
                    'retry_after': 30  # segundos
                }), 503
        
        # Verificar componentes críticos - CORRIGIDO
        if hasattr(agent_service, 'embedding_service'):
            emb_service = agent_service.embedding_service
            if not hasattr(emb_service, 'book_embeddings') or emb_service.book_embeddings is None:
                logger.error("❌ Embeddings não carregados no embedding_service!")
                return jsonify({
                    'success': False,
                    'error': 'Embeddings não carregados',
                    'details': 'book_embeddings está None no embedding_service'
                }), 500
            
            if not hasattr(emb_service, 'index') or emb_service.index is None:
                logger.error("❌ Índice FAISS não carregado no embedding_service!")
                return jsonify({
                    'success': False,
                    'error': 'Índice FAISS não carregado'
                }), 500
        else:
            logger.error("❌ embedding_service não encontrado no agente!")
            return jsonify({
                'success': False,
                'error': 'Serviço de embeddings não disponível'
            }), 500
        
        logger.info(f"🤖 Agente pronto - processando mensagem")
        
        # Processar mensagem
        result = await agent_service.process_message(
            message=data.message,
            session_id=data.session_id,
            language=data.language
        )
        
        return jsonify({
            'success': True,
            'data': result,
            'metadata': {
                'processing_time': result.get('processing_time_seconds'),
                'books_found': result.get('books_found', 0),
                'intent': result.get('intent', 'general')
            }
        }), 200
        
    except ValueError as e:
        logger.warning(f"Requisição inválida: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"❌ Erro no endpoint /chat: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Erro interno ao processar mensagem',
            'details': str(e)
        }), 500


# routes.py - Adicione esta rota

@api_bp.route('/initialize', methods=['POST', 'GET'])
def initialize_agent():
    """Inicializa o agente explicitamente"""
    try:
        logger.info("🚀 Inicialização manual do agente solicitada")
        
        # Força a inicialização
        agent_service = get_agent_service()
        
        # Verifica se está pronto
        if agent_service and getattr(agent_service, 'initialized', False):
            return jsonify({
                'success': True,
                'message': 'Agente já inicializado',
                'initialized': True
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Falha na inicialização do agente',
                'initialized': False
            }), 503
            
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/books/search', methods=['GET'])
def search_books():
    """
    Busca livros por diversos critérios
    """
    try:
        # Parsear parâmetros
        search_params = SearchRequest(**request.args.to_dict())
        
        agent_service = get_agent_service()
        books = agent_service.search_books(search_params)
        
        return jsonify({
            'success': True,
            'data': {
                'books': books,
                'count': len(books),
                'search_params': search_params.dict(exclude_none=True)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro no endpoint /books/search: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro na busca de livros'
        }), 500

@api_bp.route('/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    """
    Busca livro por ID
    """
    try:
        agent_service = get_agent_service()
        book = agent_service.get_book_by_id(book_id)
        
        if not book:
            return jsonify({
                'success': False,
                'error': f'Livro com ID {book_id} não encontrado'
            }), 404
        
        return jsonify({
            'success': True,
            'data': book
        }), 200
        
    except Exception as e:
        logger.error(f"Erro no endpoint /books/{book_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro ao buscar livro'
        }), 500

# ============================================================
# ROTAS PARA CONVERSAÇÃO SOBRE LIVROS
# ============================================================

@api_bp.route('/books/conversation', methods=['POST'])
@async_handler
async def book_conversation():
    """
    Conversa específica sobre livros já recomendados
    """
    try:
        # Validar e parsear requisição
        data = validate_request(ChatRequest, request.json)
        
        logger.info(f"📚 Conversa sobre livro - Sessão: {data.session_id}, Idioma: {data.language}")
        
        # Processar mensagem
        agent_service = get_agent_service()
        
        # Verificar se o serviço de conversação está disponível
        if not hasattr(agent_service, 'book_conversation_service') or not agent_service.book_conversation_service:
            # Fallback para o processamento normal
            result = await agent_service.process_message(
                message=data.message,
                session_id=data.session_id,
                language=data.language
            )
        else:
            # Registrar contexto de conversa
            agent_service.book_conversation_service.context_manager.add_message(
                data.session_id, 
                'user', 
                data.message
            )
            
            # Processar como conversa sobre livro
            conversation_result = await agent_service.book_conversation_service.chat_about_book(
                data.message, data.session_id, data.language
            )
            
            # Formatar resultado para compatibilidade
            result = {
                'response': conversation_result['response'],
                'intent': 'book_conversation',
                'books_found': 1 if conversation_result.get('book') else 0,
                'processing_time_seconds': 0.0,
                'session_id': data.session_id,
                'language': data.language,
                'books': [conversation_result['book']] if conversation_result.get('book') else []
            }
            
            # Atualizar contexto com a resposta
            agent_service.book_conversation_service.context_manager.add_message(
                data.session_id,
                'assistant',
                conversation_result['response'],
                books=[conversation_result['book']] if conversation_result.get('book') else []
            )
        
        return jsonify({
            'success': True,
            'data': result,
            'metadata': {
                'intent': result.get('intent', 'book_conversation'),
                'books_found': result.get('books_found', 0),
                'session_id': data.session_id,
                'is_book_conversation': result.get('intent') == 'book_conversation'
            }
        }), 200
        
    except ValueError as e:
        logger.warning(f"Requisição inválida em /books/conversation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Erro no endpoint /books/conversation: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro interno ao processar conversa sobre livro'
        }), 500

@api_bp.route('/books/conversation/history/<session_id>', methods=['GET'])
def get_conversation_history(session_id):
    """
    Obtém histórico de conversa específico
    """
    try:
        agent_service = get_agent_service()
        
        if not hasattr(agent_service, 'book_conversation_service') or not agent_service.book_conversation_service:
            return jsonify({
                'success': False,
                'error': 'Serviço de conversação não disponível'
            }), 404
        
        session = agent_service.book_conversation_service.context_manager.get_or_create_session(session_id)
        
        # Formatar timestamps para JSON
        def format_timestamp(ts):
            if hasattr(ts, 'isoformat'):
                return ts.isoformat()
            return str(ts)
        
        return jsonify({
            'success': True,
            'data': {
                'session_id': session_id,
                'created': format_timestamp(session['created']),
                'last_activity': format_timestamp(session['last_activity']),
                'message_count': len(session['conversation_history']),
                'discussed_books': list(session['discussed_books']),
                'has_last_recommendations': session.get('last_recommendations') is not None,
                'last_recommendations_count': len(session.get('last_recommendations', [])),
                'history': session['conversation_history'][-20:]  # Últimas 20 mensagens
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao obter histórico de conversa: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro ao obter histórico de conversa'
        }), 500

@api_bp.route('/books/conversation/analyze/<int:book_id>', methods=['POST'])
@async_handler
async def analyze_book_specific(book_id):
    """
    Análise específica de um livro por ID
    """
    try:
        # Validar e parsear requisição
        data = validate_request(ChatRequest, request.json)
        
        logger.info(f"🔍 Análise específica do livro ID {book_id} - Sessão: {data.session_id}")
        
        # Buscar livro por ID
        agent_service = get_agent_service()
        book = agent_service.get_book_by_id(book_id)
        
        if not book:
            return jsonify({
                'success': False,
                'error': f'Livro com ID {book_id} não encontrado'
            }), 404
        
        # Criar prompt de análise
        prompt = f"Analise este livro com base nos dados disponíveis:\n\n"
        prompt += f"Título: {book['title']}\n"
        prompt += f"Autores: {', '.join(book['authors'])}\n"
        prompt += f"Gêneros: {', '.join(book['genres'])}\n"
        prompt += f"Avaliação: {book['rating']}/5\n"
        if 'description' in book and book['description']:
            prompt += f"Descrição: {book['description']}\n\n"
        prompt += f"Pergunta do usuário: {data.message}\n\n"
        prompt += "Forneça uma análise detalhada:"
        
        # Gerar resposta com Ollama
        response_text = ""
        if hasattr(agent_service, 'ollama_service') and agent_service.ollama_service:
            try:
                response = await agent_service.ollama_service.chat([
                    {"role": "user", "content": prompt}
                ])
                response_text = response
            except Exception as ollama_error:
                logger.error(f"Erro ao consultar Ollama: {ollama_error}")
                response_text = f"Informações sobre '{book['title']}':\n"
                response_text += f"- Autor(es): {', '.join(book['authors'])}\n"
                response_text += f"- Gêneros: {', '.join(book['genres'])}\n"
                response_text += f"- Avaliação: ⭐ {book['rating']}/5\n"
                response_text += f"\nSua pergunta: {data.message}\n\n"
                response_text += "Para uma análise mais detalhada, recomendo buscar resenhas críticas deste livro."
        else:
            response_text = f"Informações sobre '{book['title']}':\n"
            response_text += f"- Autor(es): {', '.join(book['authors'])}\n"
            response_text += f"- Gêneros: {', '.join(book['genres'])}\n"
            response_text += f"- Avaliação: ⭐ {book['rating']}/5\n"
            response_text += f"\nSua pergunta: {data.message}\n\n"
            response_text += "Para uma análise mais detalhada, recomendo buscar resenhas críticas deste livro."
        
        # Atualizar contexto se o serviço de conversação estiver disponível
        if hasattr(agent_service, 'book_conversation_service') and agent_service.book_conversation_service:
            agent_service.book_conversation_service.context_manager.add_message(
                data.session_id,
                'user',
                f"Sobre o livro '{book['title']}' (ID: {book_id}): {data.message}"
            )
            agent_service.book_conversation_service.context_manager.add_message(
                data.session_id,
                'assistant',
                response_text,
                books=[book]
            )
        
        return jsonify({
            'success': True,
            'data': {
                'response': response_text,
                'book': book,
                'session_id': data.session_id,
                'language': data.language,
                'book_id': book_id
            },
            'metadata': {
                'book_title': book['title'],
                'analysis_type': 'specific_book',
                'book_id': book_id
            }
        }), 200
        
    except ValueError as e:
        logger.warning(f"Requisição inválida: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Erro na análise específica do livro: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro ao analisar livro específico'
        }), 500

@api_bp.route('/books/conversation/clear/<session_id>', methods=['POST'])
def clear_conversation_history(session_id):
    """
    Limpa o histórico de conversa de uma sessão específica
    """
    try:
        agent_service = get_agent_service()
        
        if not hasattr(agent_service, 'book_conversation_service') or not agent_service.book_conversation_service:
            return jsonify({
                'success': False,
                'error': 'Serviço de conversação não disponível'
            }), 404
        
        # Limpar sessão se existir
        if session_id in agent_service.book_conversation_service.context_manager.sessions:
            del agent_service.book_conversation_service.context_manager.sessions[session_id]
        
        return jsonify({
            'success': True,
            'message': f'Histórico da sessão {session_id} limpo com sucesso',
            'data': {
                'session_id': session_id,
                'cleared': True
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao limpar histórico de conversa: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro ao limpar histórico de conversa'
        }), 500

# ============================================================
# ROTAS DO CONSUMIDOR GCS
# ============================================================

@api_bp.route('/consumer/status', methods=['GET'])
def get_consumer_status():
    """Status do consumidor GCS"""
    try:
        consumer = GCSEmbeddingConsumer(
            bucket_name=config.GCS_BUCKET_NAME,
            embeddings_prefix=config.GCS_EMBEDDINGS_PREFIX
        )
        
        # Tentar carregar para obter status atual
        loaded = consumer.load_latest_embeddings()
        stats = consumer.get_stats()
        
        return jsonify({
            'success': True,
            'data': {
                'status': 'operational' if loaded else 'failed',
                'bucket': config.GCS_BUCKET_NAME,
                'prefix': config.GCS_EMBEDDINGS_PREFIX,
                'environment': config.ENVIRONMENT,
                'version': stats.get('version'),
                'embeddings_loaded': stats.get('status') == 'loaded',
                'embeddings_shape': stats.get('embeddings', {}).get('shape'),
                'index_size': stats.get('index', {}).get('size'),
                'loaded_at': stats.get('loaded_at'),
                'mode': 'gcs_consumer'
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao obter status: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'mode': 'gcs_consumer'
        }), 500

@api_bp.route('/consumer/reload', methods=['POST'])
def reload_embeddings():
    """Recarrega embeddings mais recentes"""
    try:
        consumer = GCSEmbeddingConsumer(
            bucket_name=config.GCS_BUCKET_NAME,
            embeddings_prefix=config.GCS_EMBEDDINGS_PREFIX
        )
        
        success = consumer.load_latest_embeddings()
        
        if success:
            stats = consumer.get_stats()
            return jsonify({
                'success': True,
                'message': f"Recarregado para versão {stats.get('version')}",
                'data': stats
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Falha ao recarregar embeddings'
            }), 500
            
    except Exception as e:
        logger.error(f"Erro ao recarregar: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/consumer/check-update', methods=['GET'])
def check_for_update():
    """Verifica se há versão mais nova"""
    try:
        consumer = GCSEmbeddingConsumer(
            bucket_name=config.GCS_BUCKET_NAME,
            embeddings_prefix=config.GCS_EMBEDDINGS_PREFIX
        )
        
        # Carrega atual primeiro
        consumer.load_latest_embeddings()
        
        # Verifica se há nova versão
        has_update = consumer.check_for_new_version()
        
        stats = consumer.get_stats()
        
        return jsonify({
            'success': True,
            'data': {
                'has_update': has_update,
                'current_version': stats.get('version'),
                'current_loaded_at': stats.get('loaded_at')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao verificar atualização: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================
# ROTAS DE SISTEMA E ESTATÍSTICAS
# ============================================================

@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """
    Obtém estatísticas do sistema
    """
    try:
        agent_service = get_agent_service()
        
        # Estatísticas do agente
        agent_stats = agent_service.get_agent_stats()
        
        # Estatísticas de busca
        search_stats = agent_service.get_search_stats()
        
        # Estatísticas do Ollama (se disponível)
        ollama_stats = agent_service.get_ollama_stats()
        
        # Estatísticas de embeddings
        embedding_stats = agent_service.get_embedding_stats()
        
        # Estatísticas de cache (se disponível)
        cache_stats = {}
        if hasattr(agent_service, 'get_cache_stats'):
            cache_stats = agent_service.get_cache_stats()
        
        # Estatísticas de conversação (se disponível)
        conversation_stats = {}
        if hasattr(agent_service, 'book_conversation_service') and agent_service.book_conversation_service:
            conversation_stats = {
                'active_sessions': len(agent_service.book_conversation_service.context_manager.sessions),
                'conversation_service_available': True
            }
        
        return jsonify({
            'success': True,
            'data': {
                'agent': agent_stats,
                'search': search_stats,
                'ollama': ollama_stats,
                'embeddings': embedding_stats,
                'cache': cache_stats,
                'conversation': conversation_stats,
                'system': {
                    'gpu_available': agent_service.is_gpu_available(),
                    'model_loaded': agent_service.is_model_loaded(),
                    'index_built': agent_service.is_index_built(),
                    'conversation_service_loaded': hasattr(agent_service, 'book_conversation_service') and agent_service.book_conversation_service is not None
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro no endpoint /stats: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro ao obter estatísticas'
        }), 500

@api_bp.route('/health/detailed', methods=['GET'])
def detailed_health():
    """
    Verificação de saúde detalhada
    """
    try:
        agent_service = get_agent_service()
        
        checks = {
            'api': True,
            'data_loaded': agent_service.is_data_loaded(),
            'embeddings_loaded': agent_service.is_model_loaded(),
            'index_built': agent_service.is_index_built(),
            'ollama_connected': agent_service.is_ollama_connected(),
            'gpu_available': agent_service.is_gpu_available(),
            'conversation_service': hasattr(agent_service, 'book_conversation_service') and agent_service.book_conversation_service is not None
        }
        
        all_healthy = all(checks.values())
        
        return jsonify({
            'status': 'healthy' if all_healthy else 'degraded',
            'checks': checks,
            'timestamp': asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else None,
            'service': 'Book Recommendation API v2.0',
            'features': {
                'recommendations': True,
                'semantic_search': True,
                'book_conversation': checks['conversation_service'],
                'ollama_integration': checks['ollama_connected']
            }
        }), 200 if all_healthy else 503
        
    except Exception as e:
        logger.error(f"Erro no health check detalhado: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

# ============================================================
# ROTAS DE TRADUÇÃO
# ============================================================

@api_bp.route('/translate', methods=['POST'])
@async_handler
async def translate_text():
    """
    Traduz texto entre idiomas
    """
    try:
        text = request.json.get('text', '')
        source_lang = request.json.get('source_lang', 'pt')
        target_lang = request.json.get('target_lang', 'en')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Texto é obrigatório'
            }), 400
        
        agent_service = get_agent_service()
        
        if source_lang == target_lang:
            translated = text
        elif target_lang == 'en':
            translated = await agent_service.translate_query(text, source_lang, target_lang)
        else:
            # Para outras direções de tradução
            translation_service = get_translation_service()
            translated = await translation_service.translate_from_english(
                await translation_service.translate_to_english(text, source_lang),
                target_lang
            )
        
        return jsonify({
            'success': True,
            'data': {
                'original': text,
                'translated': translated,
                'source_lang': source_lang,
                'target_lang': target_lang
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro na tradução: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro ao traduzir texto'
        }), 500

# ============================================================
# ROTA DE DOCUMENTAÇÃO
# ============================================================
@api_bp.route('/', methods=['GET'])
def api_documentation():
    """
    Documentação completa da API em formato estruturado
    """
    base_url = request.host_url.rstrip('/')
    
    documentation = {
        'service': 'Book Recommendation API',
        'version': '2.0.0',
        'description': 'API inteligente para recomendação e conversação sobre livros com memória por sessão no Redis',
        'repository': 'https://github.com/seu-usuario/book-agent',
        'author': 'Sua Equipe',
        'license': 'MIT',
        
        'features': [
            '💬 Chat inteligente com contexto por sessão',
            '🧠 Memória Redis para continuidade de conversas',
            '📚 Busca semântica usando embeddings',
            '🤖 Análise de livros com Ollama LLM',
            '🌐 Tradução automática entre idiomas',
            '☁️ Embeddings do Google Cloud Storage'
        ],
        
        'quick_start': {
            '1': {
                'description': 'Iniciar uma conversa',
                'method': 'POST',
                'endpoint': f'{base_url}/api/v1/chat',
                'example': {
                    'message': 'Livros de fantasia para iniciantes',
                    'session_id': 'meu-usuario-001',
                    'language': 'pt'
                }
            },
            '2': {
                'description': 'Falar sobre livro específico',
                'method': 'POST',
                'endpoint': f'{base_url}/api/v1/books/conversation',
                'example': {
                    'message': 'Sobre o primeiro livro que você recomendou, ele é complexo?',
                    'session_id': 'meu-usuario-001',
                    'language': 'pt'
                }
            },
            '3': {
                'description': 'Limpar memória da sessão',
                'method': 'POST',
                'endpoint': f'{base_url}/api/v1/memory/clear/meu-usuario-001',
                'note': 'Use para implementar botão "Nova Conversa"'
            }
        },
        
        'endpoints': {
            'chat': {
                'path': '/api/v1/chat',
                'method': 'POST',
                'description': 'Endpoint principal para chat e recomendações',
                'parameters': {
                    'body': {
                        'message': {'type': 'string', 'required': True, 'description': 'Mensagem do usuário'},
                        'session_id': {'type': 'string', 'required': False, 'default': 'default', 'description': 'ID único da sessão'},
                        'language': {'type': 'string', 'required': False, 'default': 'pt', 'enum': ['pt', 'en']}
                    }
                },
                'response_example': {
                    'success': True,
                    'data': {
                        'response': 'Aqui estão recomendações de livros...',
                        'intent': 'book_recommendation',
                        'books_found': 5,
                        'books': [
                            {
                                'book_id': 123,
                                'title': 'Nome do Livro',
                                'authors': ['Autor 1', 'Autor 2'],
                                'rating': 4.5,
                                'genres': ['Fantasia', 'Aventura']
                            }
                        ]
                    }
                }
            },
            
            'memory_management': {
                'clear_session': {
                    'path': '/api/v1/memory/clear/{session_id}',
                    'method': 'POST',
                    'description': 'Limpa toda a memória de uma sessão específica',
                    'note': 'Ideal para botão "Nova Conversa"'
                },
                'get_memory_info': {
                    'path': '/api/v1/memory/info/{session_id}',
                    'method': 'GET',
                    'description': 'Obtém informações sobre o estado da memória'
                }
            },
            
            'book_search': {
                'search': {
                    'path': '/api/v1/books/search',
                    'method': 'GET',
                    'description': 'Busca livros por diversos critérios',
                    'query_params': {
                        'query': 'Termo de busca semântica',
                        'author': 'Filtrar por autor',
                        'genre': 'Filtrar por gênero',
                        'min_rating': 'Avaliação mínima (1-5)',
                        'method': 'semantic|author|genre|popularity',
                        'limit': 'Número máximo de resultados (1-50)'
                    }
                },
                'get_by_id': {
                    'path': '/api/v1/books/{book_id}',
                    'method': 'GET',
                    'description': 'Obtém detalhes de um livro específico'
                }
            },
            
            'system': {
                'health': {
                    'path': '/api/v1/health',
                    'method': 'GET',
                    'description': 'Verificação simples de saúde'
                },
                'detailed_health': {
                    'path': '/api/v1/health/detailed',
                    'method': 'GET',
                    'description': 'Verificação detalhada de todos os componentes'
                },
                'stats': {
                    'path': '/api/v1/stats',
                    'method': 'GET',
                    'description': 'Estatísticas do sistema'
                }
            }
        },
        
        'examples': {
            'curl': {
                'basic_chat': f'curl -X POST {base_url}/api/v1/chat -H "Content-Type: application/json" -d \'{{"message": "Livros de programação", "session_id": "test-001", "language": "en"}}\'',
                'search_books': f'curl -X GET "{base_url}/api/v1/books/search?query=fantasia&limit=5"',
                'clear_memory': f'curl -X POST {base_url}/api/v1/memory/clear/test-001'
            },
            'python': '''import requests

# Chat com o agente
response = requests.post('http://localhost:8080/api/v1/chat', json={
    'message': 'Livros de ficção científica',
    'session_id': 'meu-usuario',
    'language': 'pt'
})

print(response.json())

# Limpar memória
requests.post('http://localhost:8080/api/v1/memory/clear/meu-usuario')'''
        },
        
        'testing': {
            'postman_collection': f'{base_url}/api/v1/postman.json',
            'openapi_spec': f'{base_url}/api/v1/openapi.json',
            'test_session_id': 'Test-001',
            'test_requests': [
                {'method': 'POST', 'endpoint': '/chat', 'body': {'message': 'Olá!', 'session_id': 'Test-001'}},
                {'method': 'GET', 'endpoint': '/memory/info/Test-001'},
                {'method': 'POST', 'endpoint': '/memory/clear/Test-001'}
            ]
        },
        
        'monitoring': {
            'health': f'{base_url}/api/v1/health',
            'detailed_health': f'{base_url}/api/v1/health/detailed',
            'stats': f'{base_url}/api/v1/stats',
            'consumer_status': f'{base_url}/api/v1/consumer/status'
        },
        
        'support': {
            'issues': 'https://github.com/seu-usuario/book-agent/issues',
            'documentation': f'{base_url}/docs',
            'api_reference': f'{base_url}/api/v1'
        }
    }
    
    return jsonify(documentation)

# ============================================================
# ROTA DE HEALTH SIMPLES
# ============================================================

@api_bp.route('/health', methods=['GET'])
def simple_health():
    """
    Verificação de saúde simples
    """
    return jsonify({
        'status': 'healthy',
        'service': 'Book Recommendation API',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat()
    }), 200


# routes.py - Adicione este endpoint

@api_bp.route('/reinitialize', methods=['POST'])
def reinitialize_agent():
    """Reinicializa o agente manualmente"""
    global _agent_service
    
    try:
        logger.info("🔄 Reinicializando agente manualmente...")
        
        with _agent_lock:
            # Limpa a instância existente
            _agent_service = None
            
            # Cria nova instância
            _agent_service = BookAgentService()
            
            # Inicializa
            _agent_service.initialize()
            _agent_service.initialized = True
            
            logger.info("✅ Agente reinicializado com sucesso")
            
            # Verificar
            emb_service = _agent_service.embedding_service
            logger.info(f"   Embeddings: {emb_service.book_embeddings.shape if hasattr(emb_service, 'book_embeddings') and emb_service.book_embeddings is not None else 'None'}")
            logger.info(f"   Índice: {emb_service.index.ntotal if hasattr(emb_service, 'index') and emb_service.index is not None else 'None'}")
        
        return jsonify({
            'success': True,
            'message': 'Agente reinicializado com sucesso',
            'embeddings_loaded': hasattr(_agent_service.embedding_service, 'book_embeddings') and _agent_service.embedding_service.book_embeddings is not None,
            'index_loaded': hasattr(_agent_service.embedding_service, 'index') and _agent_service.embedding_service.index is not None
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Erro na reinicialização: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Adicione estas rotas no final do arquivo, antes do final da classe

@api_bp.route('/memory/clear/<session_id>', methods=['POST'])
def clear_session_memory(session_id):
    """
    Limpa a memória de uma sessão específica
    ---
    tags:
      - Memory Management
    parameters:
      - name: session_id
        in: path
        required: true
        type: string
        description: ID da sessão a ser limpa
    responses:
      200:
        description: Sessão limpa com sucesso
      404:
        description: Sessão não encontrada ou serviço não disponível
    """
    try:
        logger.info(f"🧹 Rota chamada para limpar sessão: {session_id}")
        
        agent_service = get_agent_service()
        
        if not agent_service:
            return jsonify({
                'success': False,
                'error': 'Agente não inicializado'
            }), 503
        
        # Chamar método do agente
        result = agent_service.clear_session_data(session_id)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': f'Memória da sessão {session_id} limpa com sucesso',
                'data': result
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Erro desconhecido'),
                'message': result.get('message', '')
            }), 400
            
    except Exception as e:
        logger.error(f"❌ Erro ao limpar memória da sessão: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro interno ao limpar memória da sessão',
            'details': str(e)
        }), 500

@api_bp.route('/memory/clear-all', methods=['POST'])
def clear_all_memory():
    """
    Limpa TODAS as sessões (PERIGOSO - apenas para admin)
    ---
    tags:
      - Memory Management
    parameters:
      - name: admin_token
        in: query
        required: true
        type: string
        description: Token de administração para confirmar a operação perigosa
    responses:
      200:
        description: Todas as sessões limpas
      403:
        description: Token de administração inválido
    """
    try:
        # Verificar token de administração (simples)
        admin_token = request.args.get('admin_token')
        expected_token = os.getenv('ADMIN_TOKEN', 'admin123')
        
        if admin_token != expected_token:
            logger.warning("⚠️  Tentativa de limpeza total sem token válido")
            return jsonify({
                'success': False,
                'error': 'Token de administração inválido'
            }), 403
        
        logger.warning("⚠️  Iniciando limpeza total de todas as sessões...")
        
        agent_service = get_agent_service()
        
        if not agent_service:
            return jsonify({
                'success': False,
                'error': 'Agente não inicializado'
            }), 503
        
        # Chamar método perigoso
        result = agent_service.clear_all_sessions()
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': 'Todas as sessões foram limpas',
                'data': result,
                'warning': 'Operação perigosa concluída'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Erro desconhecido')
            }), 400
            
    except Exception as e:
        logger.error(f"❌ Erro na limpeza total: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro interno na limpeza total',
            'details': str(e)
        }), 500

@api_bp.route('/memory/info/<session_id>', methods=['GET'])
def get_memory_info(session_id):
    """
    Obtém informações sobre a memória de uma sessão
    ---
    tags:
      - Memory Management
    parameters:
      - name: session_id
        in: path
        required: true
        type: string
        description: ID da sessão
    responses:
      200:
        description: Informações da memória
      404:
        description: Sessão não encontrada
    """
    try:
        agent_service = get_agent_service()
        
        if not agent_service:
            return jsonify({
                'success': False,
                'error': 'Agente não inicializado'
            }), 503
        
        # Verificar se existe serviço de conversação
        if not hasattr(agent_service, 'book_conversation_service') or not agent_service.book_conversation_service:
            return jsonify({
                'success': False,
                'error': 'Serviço de conversação não disponível'
            }), 404
        
        # Obter sessão
        session = agent_service.book_conversation_service.context_manager.get_or_create_session(session_id)
        
        # Contar chaves no Redis para esta sessão
        try:
            redis_keys = agent_service.book_conversation_service.context_manager.redis.keys(f"*{session_id}*")
            redis_key_count = len(redis_keys)
        except:
            redis_key_count = 0
        
        return jsonify({
            'success': True,
            'data': {
                'session_id': session_id,
                'exists_in_redis': len(session.get('conversation_history', [])) > 0,
                'message_count': len(session.get('conversation_history', [])),
                'discussed_books_count': len(session.get('discussed_books', [])),
                'last_activity': session.get('last_activity'),
                'created': session.get('created'),
                'redis_keys_count': redis_key_count,
                'has_last_recommendations': session.get('last_recommendations') is not None,
                'last_recommendations_count': len(session.get('last_recommendations', []))
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao obter informações da memória: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro ao obter informações da memória'
        }), 500

@api_bp.route('/memory/list-sessions', methods=['GET'])
def list_all_sessions():
    """
    Lista todas as sessões ativas (apenas admin)
    ---
    tags:
      - Memory Management
    parameters:
      - name: admin_token
        in: query
        required: true
        type: string
        description: Token de administração
    responses:
      200:
        description: Lista de sessões
      403:
        description: Token de administração inválido
    """
    try:
        # Verificar token de administração
        admin_token = request.args.get('admin_token')
        expected_token = os.getenv('ADMIN_TOKEN', 'admin123')
        
        if admin_token != expected_token:
            return jsonify({
                'success': False,
                'error': 'Token de administração inválido'
            }), 403
        
        agent_service = get_agent_service()
        
        if not agent_service:
            return jsonify({
                'success': False,
                'error': 'Agente não inicializado'
            }), 503
        
        # Tentar listar sessões do Redis
        try:
            redis_client = agent_service.book_conversation_service.context_manager.redis
            # Buscar todas as chaves de conversação
            conversation_keys = redis_client.keys("conversation:*")
            chat_keys = redis_client.keys("chat:*")
            
            all_keys = list(set(conversation_keys + chat_keys))
            
            sessions = []
            for key in all_keys:
                try:
                    data = redis_client.get(key)
                    if data:
                        session_data = json.loads(data)
                        # Extrair session_id da chave
                        if key.startswith("conversation:"):
                            session_id = key.replace("conversation:", "")
                        elif ":" in key:
                            parts = key.split(":")
                            session_id = parts[1] if len(parts) > 1 else "unknown"
                        else:
                            session_id = "unknown"
                        
                        sessions.append({
                            'session_id': session_id,
                            'key': key,
                            'message_count': len(session_data.get('conversation_history', [])),
                            'last_activity': session_data.get('last_activity'),
                            'book_count': len(session_data.get('discussed_books', [])),
                            'has_recommendations': session_data.get('last_recommendations') is not None
                        })
                except:
                    continue
            
            return jsonify({
                'success': True,
                'data': {
                    'total_sessions': len(sessions),
                    'sessions': sessions[:100]  # Limitar a 100 para não sobrecarregar
                }
            }), 200
            
        except Exception as redis_error:
            logger.error(f"Erro ao listar sessões: {redis_error}")
            return jsonify({
                'success': False,
                'error': f'Erro ao listar sessões: {str(redis_error)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Erro geral ao listar sessões: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro ao listar sessões'
        }), 500