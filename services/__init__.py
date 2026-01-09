# D:\Django\book_agent\services\__init__.py

from utils.data_loader import DataLoader
from services.embedding_service import EmbeddingService
from .ollama_service import OllamaService
from .search_engine import BookSearchEngine, BookResult
from .response_generator import ResponseGenerator
from .agent_service import BookAgentService
#from .conversation_context import ConversationContextManager
from .book_conversation_service import BookConversationService
from .translation_service import TranslationService, get_translation_service

from .conversation_memory import ConversationMemoryManager


ConversationContextManager = ConversationMemoryManager
__all__ = [
    'DataLoader',
    'Embedding_service',
    'OllamaService',
    'BookSearchEngine',
    'BookResult',
    'ResponseGenerator',
    'BookAgentService',
    # NOVAS EXPORTAÇÕES
    'ConversationContextManager',
    'BookConversationService',
    'TranslationService',
    'get_translation_service',
    'BookAgentService',
    'OllamaService',
    'BookSearchEngine',
    'ResponseGenerator',
    'BookConversationService',
    'TranslationService',
    'get_translation_service'
]