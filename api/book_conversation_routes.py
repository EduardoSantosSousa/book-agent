# D:\Django\book_agent\api\book_conversation_routes.py

from flask import Blueprint, request, jsonify
import asyncio
import logging
from functools import wraps
from models.schemas import ChatRequest
from services.agent_service import BookAgentService
from utils.validators import validate_request

logger = logging.getLogger(__name__)
book_conv_bp = Blueprint('book_conversation', __name__, url_prefix='/books')

def async_handler(f):
    """Decorator para lidar com funções assíncronas"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

def get_agent_service():
    """Importa a função de obtenção do serviço"""
    from api.routes import get_agent_service as get_service
    return get_service()

@book_conv_bp.route('/conversation', methods=['POST'])
@async_handler
async def book_conversation():
    """
    Endpoint para conversas específicas sobre livros
    ---
    tags:
      - Book Conversation
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - message
            properties:
              message:
                type: string
                description: Mensagem sobre livro específico
                example: "Sobre o livro 'The Time Machine', ele é muito complexo?"
              session_id:
                type: string
                description: ID da sessão do usuário
                example: "user-123"
              language:
                type: string
                description: Idioma preferido (pt/en)
                example: "pt"
                default: "pt"
    responses:
      200:
        description: Resposta sobre o livro específico
      400:
        description: Requisição inválida
    """
    try:
        # Validar e parsear requisição
        data = validate_request(ChatRequest, request.json)
        
        # Processar mensagem
        agent_service = get_agent_service()
        
        # Registrar contexto de conversa
        if hasattr(agent_service, 'book_conversation_service') and agent_service.book_conversation_service:
            agent_service.book_conversation_service.context_manager.add_message(
                data.session_id, 
                'user', 
                data.message
            )
        
        # Processar mensagem (agora já detecta automaticamente se é conversa sobre livro)
        result = await agent_service.process_message(
            message=data.message,
            session_id=data.session_id,
            language=data.language
        )
        
        return jsonify({
            'success': True,
            'data': {
                'response': result['response'],
                'intent': result.get('intent', 'conversation'),
                'books_found': result['books_found'],
                'processing_time_seconds': result['processing_time_seconds'],
                'session_id': result['session_id'],
                'language': result['language'],
                'books': result['books']
            },
            'metadata': {
                'processing_time': result.get('processing_time_seconds'),
                'books_found': result.get('books_found', 0),
                'is_book_conversation': result.get('intent') == 'book_conversation'
            }
        }), 200
        
    except ValueError as e:
        logger.warning(f"Requisição inválida: {e}")
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

@book_conv_bp.route('/conversation/history/<session_id>', methods=['GET'])
def get_conversation_history(session_id):
    """
    Obtém histórico de conversa específico
    ---
    tags:
      - Book Conversation
    parameters:
      - name: session_id
        in: path
        required: true
        type: string
        description: ID da sessão
    responses:
      200:
        description: Histórico de conversa
      404:
        description: Serviço de conversação não disponível
    """
    try:
        agent_service = get_agent_service()
        
        if not hasattr(agent_service, 'book_conversation_service') or not agent_service.book_conversation_service:
            return jsonify({
                'success': False,
                'error': 'Serviço de conversação não disponível'
            }), 404
        
        session = agent_service.book_conversation_service.context_manager.get_or_create_session(session_id)
        
        return jsonify({
            'success': True,
            'data': {
                'session_id': session_id,
                'created': session['created'].isoformat() if hasattr(session['created'], 'isoformat') else str(session['created']),
                'last_activity': session['last_activity'].isoformat() if hasattr(session['last_activity'], 'isoformat') else str(session['last_activity']),
                'message_count': len(session['conversation_history']),
                'discussed_books': list(session['discussed_books']),
                'history': session['conversation_history'][-20:]  # Últimas 20 mensagens
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao obter histórico de conversa: {e}")
        return jsonify({
            'success': False,
            'error': 'Erro ao obter histórico de conversa'
        }), 500

@book_conv_bp.route('/conversation/analyze/<int:book_id>', methods=['POST'])
@async_handler
async def analyze_book_specific(book_id):
    """
    Análise específica de um livro por ID
    ---
    tags:
      - Book Conversation
    parameters:
      - name: book_id
        in: path
        required: true
        type: integer
        description: ID do livro
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required:
              - message
            properties:
              message:
                type: string
                description: Pergunta específica sobre o livro
                example: "Qual a mensagem principal deste livro?"
              session_id:
                type: string
                description: ID da sessão do usuário
                example: "user-123"
              language:
                type: string
                description: Idioma preferido (pt/en)
                example: "pt"
                default: "pt"
    responses:
      200:
        description: Análise do livro
      404:
        description: Livro não encontrado
    """
    try:
        # Validar e parsear requisição
        data = validate_request(ChatRequest, request.json)
        
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
                f"Sobre o livro '{book['title']}': {data.message}"
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
                'language': data.language
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