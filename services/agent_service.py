# D:\Django\book_agent\services\agent_service.py
import asyncio
import logging
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
from utils.data_loader import DataLoader
from services.embedding_service import EmbeddingService
from services.groq_service import GroqService
from models.schemas import SearchRequest
import json
from .book_conversation_service import BookConversationService
import re
from services.translation_service import get_translation_service
import os
from services.conversation_context import ConversationContextManager
from dotenv import load_dotenv
from services.query_refiner import QueryRefinerAgent

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    book_id: int
    title: str
    authors: List[str]
    genres: List[str]
    rating: float
    num_ratings: int
    description: str
    similarity_score: float
    search_method: str

class BookAgentService:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.data_loader = None
        self.embedding_service = None
        self.ollama_service = None
        self.search_engine = None
        self.response_generator = None
        self.initialized = False
        self.conversation_history = []
        self.book_conversation_service = None
        self.translation_service = None
        self.query_refiner = None  

        # Sistema de cache:
        self.search_cache = {}
        self.cache_ttl = timedelta(minutes=5)  # Cache por 5 minutos
        self.cache_hits = 0
        self.cache_misses = 0

        # 🔥 MEMÓRIA CENTRAL
        self.memory = ConversationContextManager(
            redis_url=os.getenv("REDIS_URL"),
            max_context_messages=50,
            ttl_hours=5,
        )
        
   
    def initialize(self):
        """Inicializa todos os componentes como consumidor puro"""
        if self.initialized:
            logger.info("Serviço já inicializado")
            return True
            
        try:
            # 1. Carregar dados do GCS (a versão mais recente)
            logger.info("📖 Carregando dataset do GCS...")
            
            # Configurar DataLoader para carregar do GCS
            self.data_loader = DataLoader(
                gcs_bucket="book-agent-embeddings-bucket",
                gcs_prefix="exports/"
            )
            
            if not self.data_loader.load_data():
                # Fallback: tentar carregar localmente
                logger.warning("⚠️ Falha ao carregar do GCS, tentando localmente...")
                self.data_loader = DataLoader(
                    data_path=self.config.get('data_path', 'data/book_dataset_treated.csv')
                )
                
                if not self.data_loader.load_data():
                    logger.error("❌ Falha ao carregar dataset local também")
                    # Criar dataset vazio para não quebrar o sistema
                    import pandas as pd
                    self.data_loader.data = pd.DataFrame()
                    logger.warning("⚠️ Usando dataset vazio - funcionalidade limitada")
            
            logger.info(f"✅ Dataset carregado: {len(self.data_loader.data)} livros")
            
            # 2. Inicializar sistema de embeddings (CONSUMIDOR PURO)
            logger.info("🔗 Conectando ao bucket GCS para embeddings...")
            
            self.embedding_service = EmbeddingService(
                model_name=self.config.get('embedding_model', 'paraphrase-multilingual-MiniLM-L12-v2'),
                use_gpu=self.config.get('use_gpu', True)
            )
            
            if not self.embedding_service.initialize():
                raise Exception("Falha ao conectar ao bucket GCS")
            
            # Log da versão carregada
            stats = self.embedding_service.get_stats()
            logger.info(f"✅ Embeddings carregados do bucket")
            logger.info(f"   Versão: {stats.get('version', 'N/A')}")
            logger.info(f"   Shape: {stats.get('embeddings', {}).get('shape', 'N/A')}")
            logger.info(f"   Índice: {stats.get('index', {}).get('size', 0)} vetores")
            
            # 3. Verificar correspondência entre embeddings e dataset
            if hasattr(self.embedding_service, 'book_embeddings'):
                num_embeddings = self.embedding_service.book_embeddings.shape[0]
                num_books = len(self.data_loader.data)
                
                logger.info(f"📊 Correspondência embeddings-dataset:")
                logger.info(f"   Embeddings: {num_embeddings}")
                logger.info(f"   Dataset: {num_books}")
                
                if num_embeddings != num_books:
                    logger.warning(f"⚠️ Diferença de {abs(num_embeddings - num_books)} registros")
            
            # 4. Inicializar Ollama ou Groq
            logger.info("🤖 Conectando ao Groq...")

            secret_path = os.getenv("GROQ_API_KEY_FILE")
            if secret_path:
                with open(secret_path, "r", encoding="utf-8") as f:
                    groq_api_key = f.read().strip()
            else:
                groq_api_key = os.getenv("GROQ_API_KEY")

            self.ollama_service = GroqService(
                model=self.config.get('groq_model', os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")),
                api_key=groq_api_key
            )

            logger.info("🤖 Groq configurado.")
            
            # 5. Criar motor de busca
            logger.info("🔍 Criando motor de busca...")
            from services.search_engine import BookSearchEngine
            self.search_engine = BookSearchEngine(
                data=self.data_loader.data,
                embedding_service=self.embedding_service
            )
            
            # 6. Criar gerador de respostas
            logger.info("💬 Criando gerador de respostas...")
            from services.response_generator import ResponseGenerator
            self.response_generator = ResponseGenerator(self.ollama_service)

            # 7. Criar serviço de conversação sobre livros
            logger.info("📚 Criando serviço de conversação...")
            self.book_conversation_service = BookConversationService(
                ollama_service=self.ollama_service,
                data_loader=self.data_loader,
                search_engine=self.search_engine
            )
            
            # 8. Inicializar serviço de tradução
            logger.info("🌐 Inicializando tradução...")
            self.translation_service = get_translation_service()
            
            # 9. Inicializar refinador de queries
            logger.info("🧠 Inicializando refinador de queries...")
            self.query_refiner = QueryRefinerAgent(self.ollama_service)
                
            self.initialized = True
            logger.info("🎉 Book Agent Service inicializado!")
            logger.info("   Fonte dados: GCS (versão mais recente)")
            logger.info(f"   Total livros: {len(self.data_loader.data)}")

            return True
            
        except Exception as e:
            logger.error(f"❌ Falha ao inicializar: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.initialized = False
            raise

    # ==============================================
    # MÉTODOS AUXILIARES - CONVERSÕES
    # ==============================================

    def _convert_book_results_to_dicts(self, books: List) -> List[Dict]:
        """Converte objetos BookResult para dicionários"""
        if not books:
            return []
        
        books_dicts = []
        for book in books:
            # Verificar se é BookResult (tem atributo book_id)
            if hasattr(book, 'book_id'):
                book_dict = {
                    "book_id": book.book_id,
                    "title": book.title,
                    "authors": book.authors,
                    "description": book.description,
                    "genres": book.genres,
                    "rating": book.rating,
                    "similarity_score": book.similarity_score,
                    "search_method": book.search_method
                }
                # Adicionar campos opcionais se existirem
                if hasattr(book, 'num_ratings'):
                    book_dict["num_ratings"] = book.num_ratings
                if hasattr(book, 'price'):
                    book_dict["price"] = book.price
                
                books_dicts.append(book_dict)
            # Se já for dicionário, usar como está
            elif isinstance(book, dict):
                books_dicts.append(book)
            else:
                logger.warning(f"Tipo de livro desconhecido: {type(book)}")
        
        return books_dicts

    def _format_books_for_response(self, books: List) -> List[Dict]:
        """Formata livros para resposta da API"""
        return self._convert_book_results_to_dicts(books)

    def _book_dict_to_result(self, book_dict: Dict):
        """Converte dicionário de livro para objeto BookResult"""
        from .search_engine import BookResult
        
        return BookResult(
            book_id=book_dict.get('book_id', 0),
            title=book_dict.get('title', ''),
            authors=book_dict.get('authors', []),
            description=book_dict.get('description', ''),
            genres=book_dict.get('genres', []),
            rating=book_dict.get('rating', 0),
            num_ratings=book_dict.get('num_ratings', 0),
            price=book_dict.get('price', 'N/A'),
            similarity_score=book_dict.get('similarity_score', 0.0),
            search_method=book_dict.get('search_method', 'redis_cache')
        )

    # ==============================================
    # MÉTODOS DE CONTEXTO E BUSCA
    # ==============================================

    def _analyze_context(self, message: str, conversation_history: List[Dict], 
                        last_recommendations: List[Dict], language: str) -> Dict:
        """Analisa o contexto da conversa de forma inteligente"""
        
        analysis = {
            'is_continuation': False,
            'topic_shift': False,
            'asking_about_previous': False,
            'similarity_score': 0.0,
            'previous_topic': None,
            'current_topic': None
        }
        
        if not conversation_history:
            return analysis
        
        # 1. Verificar se está perguntando sobre livros já recomendados
        message_lower = message.lower()
        asking_keywords = {
            'pt': ['algum dos', 'alguma das', 'os livros', 'as recomendações', 
                'já recomendados', 'que você recomendou', 'anteriores', 
                'desse', 'dessa', 'aquele', 'esse'],
            'en': ['any of the', 'the books', 'the recommendations', 
                'already recommended', 'that you recommended', 'previous',
                'that one', 'this one']
        }
        
        keywords = asking_keywords.get(language, asking_keywords['pt'])
        analysis['asking_about_previous'] = any(keyword in message_lower for keyword in keywords)
        
        # 2. Detectar tópicos
        analysis['current_topic'] = self._detect_topic(message, language)
        
        # Última mensagem do usuário
        last_user_message = None
        for msg in reversed(conversation_history):
            if msg.get('role') == 'user':
                last_user_message = msg.get('content', '')
                break
        
        if last_user_message:
            analysis['previous_topic'] = self._detect_topic(last_user_message, language)
            
            # Calcular similaridade entre tópicos
            if analysis['current_topic'] and analysis['previous_topic']:
                analysis['topic_shift'] = analysis['current_topic'] != analysis['previous_topic']
                
                # Similaridade textual
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(
                    None, 
                    message_lower, 
                    last_user_message.lower()
                ).ratio()
                analysis['similarity_score'] = similarity
        
        # 3. Determinar se é continuação
        analysis['is_continuation'] = (
            not analysis['topic_shift'] and 
            not analysis['asking_about_previous'] and
            analysis['similarity_score'] > 0.3
        )
        
        return analysis

    def _detect_topic(self, text: str, language: str) -> str:
        """Detecta o tópico principal do texto"""
        text_lower = text.lower()
        
        topic_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'programming', 'coding', 
                        'software', 'algorithm', 'data structure', 'web development',
                        'programação', 'programacao', 'código', 'codigo', 'algoritmo'],
            'data_science': ['data science', 'machine learning', 'artificial intelligence', 
                            'ai', 'data analysis', 'statistics', 'big data',
                            'ciência de dados', 'ciencia de dados', 'aprendizado de máquina',
                            'inteligência artificial', 'inteligencia artificial'],
            'physics': ['physics', 'física', 'fisica', 'mechanics', 'quantum', 'relativity',
                    'thermodynamics', 'optics', 'mecânica', 'mecanica', 'óptica', 'optica'],
            'mathematics': ['mathematics', 'math', 'calculus', 'algebra', 'geometry', 
                        'statistics', 'probability', 'matemática', 'matematica',
                        'cálculo', 'calculo', 'álgebra', 'algebra'],
            'leadership': ['leadership', 'management', 'team', 'lead', 'manager',
                        'liderança', 'lideranca', 'gestão', 'gestao', 'chefia', 'equipe'],
            'business': ['business', 'entrepreneurship', 'marketing', 'finance', 'economics',
                        'negócios', 'negocios', 'empreendedorismo', 'marketing', 'finanças'],
            'fiction': ['fiction', 'novel', 'story', 'fantasy', 'science fiction', 'romance',
                    'ficção', 'ficcao', 'romance', 'fantasia', 'ficção científica'],
            'self_help': ['self help', 'self-help', 'personal development', 'motivation',
                        'autoajuda', 'auto-ajuda', 'desenvolvimento pessoal', 'motivação']
        }
        
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_topics.append(topic)
                    break
        
        return detected_topics[0] if detected_topics else 'general'
    
    async def _intelligent_search(self, message: str, user_profile: Dict, 
                                conversation_history: List[Dict], language: str) -> List:
        """
        Busca inteligente com refinamento de query
        """
        # 1. Refinar a query
        refinement = await self.query_refiner.refine_search_query(message, language)
        
        normalized_query = refinement.get("normalized_query", message)
        synonyms = refinement.get("synonyms", [])
        keywords = refinement.get("keywords", [])
        search_intent = refinement.get("search_intent", "general")
        
        logger.info(f"🧠 Busca inteligente - Intenção: {search_intent}")
        logger.info(f"   Query normalizada: '{normalized_query}'")
        logger.info(f"   Sinônimos: {synonyms[:3]}...")
        
        # 2. Construir query expandida
        expanded_query = normalized_query
        
        # Adicionar sinônimos se for sobre quadrinhos
        if search_intent == "comics":
            expanded_query = f"{normalized_query} {' '.join(synonyms[:5])}"
            logger.info(f"   Query expandida (comics): {expanded_query}")
        
        # 3. Expandir com contexto se houver histórico
        if conversation_history:
            context_expansion = await self.query_refiner.expand_with_context(
                expanded_query, conversation_history, language
            )
            expanded_query = context_expansion.get("expanded_query", expanded_query)
            logger.info(f"   Query com contexto: {expanded_query}")
        
        # 4. Executar busca híbrida
        try:
            # Primeiro: busca semântica com query expandida
            semantic_results = self.search_engine.search_by_semantic(
                expanded_query, k=12
            )
            
            # Segundo: busca textual com termos-chave
            textual_results = []
            for keyword in keywords[:3]:
                textual = self.search_engine.search_by_textual(keyword, k=8)
                textual_results.extend(textual)
            
            # Combinar resultados
            all_results = semantic_results + textual_results
            
            # Remover duplicatas
            unique_results = self._remove_duplicate_books(all_results)
            
            # Ordenar por relevância
            if search_intent == "comics":
                unique_results.sort(
                    key=lambda x: (
                        1 if any(keyword.lower() in x.title.lower() 
                                for keyword in keywords) else 0,
                        x.similarity_score
                    ),
                    reverse=True
                )
            else:
                unique_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(f"📚 Resultados combinados: {len(unique_results)} livros")
            return unique_results[:10]
            
        except Exception as e:
            logger.error(f"❌ Erro na busca inteligente: {e}")
            # Fallback para busca semântica simples
            return self.search_engine.search_by_semantic(normalized_query, k=8)

    def _determine_search_strategy(self, message: str, intent: str, 
                                context_analysis: Dict, last_recommendations: List[Dict]) -> str:
        """Determina a estratégia de busca ideal"""
        
        message_lower = message.lower()
        
        # 1. Se está explicitamente perguntando sobre livros anteriores
        if context_analysis['asking_about_previous']:
            return "use_previous_only"
        
        # 2. Se é mudança clara de tópico
        if context_analysis['topic_shift']:
            return "new_search"
        
        # 3. Se é uma intenção específica que requer busca nova
        if intent in ['author', 'genre', 'popular']:
            return "new_search"
        
        # 4. Se há recomendações anteriores E é continuação do mesmo tópico
        if last_recommendations and context_analysis['is_continuation']:
            # Verificar se a pergunta é sobre aspectos específicos dos livros anteriores
            specific_aspects = ['melhor', 'mais', 'recomenda', 'indica', 'sugere']
            if any(aspect in message_lower for aspect in specific_aspects):
                return "context_boosted"
            else:
                return "similar_to_previous"
        
        # 5. Caso padrão: busca nova
        return "new_search"

    def _extract_keywords_from_books(self, books: List[Dict]) -> str:
        """Extrai palavras-chave dos livros para usar como contexto"""
        keywords = []
        
        for book in books[:2]:
            # Título
            title = book.get('title', '')
            keywords.extend(title.split()[:3])
            
            # Autores
            authors = book.get('authors', [])
            if authors:
                keywords.extend(authors[0].split()[:2])
            
            # Gêneros
            genres = book.get('genres', [])
            if genres:
                keywords.extend(genres[:2])
        
        # Remover duplicados e limitar
        unique_keywords = list(set(keywords))[:5]
        return ' '.join(unique_keywords)

    def _remove_duplicate_books(self, books: List) -> List:
        """Remove livros duplicados da lista"""
        unique_books = []
        seen_ids = set()
        
        for book in books:
            if hasattr(book, 'book_id'):
                book_id = book.book_id
            else:
                book_id = book.get('book_id')
                
            if book_id and book_id not in seen_ids:
                seen_ids.add(book_id)
                unique_books.append(book)
            elif not book_id:  # Se não tem ID, usa título como identificador
                if hasattr(book, 'title'):
                    title = book.title
                else:
                    title = book.get('title', '')
                    
                if title not in seen_ids:
                    seen_ids.add(title)
                    unique_books.append(book)
        
        return unique_books

    # ==============================================
    # MÉTODO PRINCIPAL - process_message
    # ==============================================

    async def process_message(self, message: str, session_id: str = "default", language: str = "pt") -> Dict:
        """Processa uma mensagem do usuário COM HISTÓRICO DO REDIS"""
        if not self.initialized:
            raise Exception("Serviço não inicializado")
        
        start_time = datetime.now()
        
        try:
            # ==============================================
            # 1. OBTER HISTÓRICO DA CONVERSA DO REDIS
            # ==============================================
            conversation_history = []
            last_recommendations = []
            
            if self.book_conversation_service:
                session = self.book_conversation_service.context_manager.get_or_create_session(session_id)
                conversation_history = session.get("conversation_history", [])
                last_recommendations = session.get("last_recommendations", []) or []
                
                logger.info(f"📖 Histórico Redis - Sessão '{session_id}':")
                logger.info(f"   📝 Mensagens: {len(conversation_history)}")
                logger.info(f"   📚 Últimas recomendações: {len(last_recommendations)} livros")

            # ==============================================
            # 2. ANALISAR INTENÇÃO
            # ==============================================
            intent = self._analyze_intent(message)
            
            # Verificar se é referência a livro anterior
            is_reference_to_previous = self._is_reference_to_previous_books(
                message, last_recommendations, conversation_history, language
            )
            
            if is_reference_to_previous:
                logger.info(f"🔍 Referência a livros anteriores detectada")
                intent = "book_conversation"
            
            # ==============================================
            # 3. CASOS ESPECIAIS: CLOSING E AUTHOR
            # ==============================================
            
            # Closing - responder imediatamente
            if intent == 'closing':
                logger.info(f"🎯 Intenção 'closing' detectada")
                
                response = await self.response_generator.generate_personalized_recommendation(
                    user_message=message,
                    books=[],
                    intent=intent,
                    language=language,
                    conversation_history=conversation_history
                )
                
                # Salvar no Redis
                if self.book_conversation_service:
                    books_dicts = self._convert_book_results_to_dicts([])
                    self.book_conversation_service.context_manager.add_message(
                        session_id, 'user', message, intent=intent
                    )
                    self.book_conversation_service.context_manager.add_message(
                        session_id, 
                        "assistant", 
                        response, 
                        books=books_dicts,
                        intent=intent
                    )
                
                result = {
                    'response': response,
                    'intent': intent,
                    'books_found': 0,
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                    'session_id': session_id,
                    'language': language,
                    'books': []
                }
                
                return result
            
            # Author - busca por autor
            if intent == 'author':
                logger.info(f"🎯 Intenção 'author' detectada")
                
                author = self._extract_author(message)
                logger.info(f"✍️  Autor extraído: {author}")
                
                if author:
                    books = self.search_engine.search_by_author(author, limit=10)
                    logger.info(f"📚 Livros encontrados para autor '{author}': {len(books)}")
                else:
                    user_profile = self._extract_user_profile(message, language)
                    search_query = self._build_search_query(message, user_profile)
                    books = self.search_engine.search_by_semantic(search_query, k=10)
                
                # Converter BookResult para dicionários antes de passar para response_generator
                books_dicts = self._convert_book_results_to_dicts(books)
                
                # Gerar resposta
                response_text = await self.response_generator.generate_personalized_recommendation(
                    user_message=message,
                    books=books_dicts,
                    intent=intent,
                    language=language,
                    conversation_history=conversation_history
                )
                
                # Salvar no Redis - IMPORTANTE: passar dicionários, não objetos BookResult
                if self.book_conversation_service:
                    self.book_conversation_service.context_manager.add_message(
                        session_id, 'user', message, intent=intent
                    )
                    self.book_conversation_service.context_manager.add_message(
                        session_id, 'assistant', response_text, 
                        books=books_dicts[:3],  # Passar dicionários convertidos
                        intent=intent
                    )
                
                result = {
                    'response': response_text,
                    'intent': intent,
                    'user_profile': self._extract_user_profile(message, language),
                    'books_found': len(books),
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                    'session_id': session_id,
                    'language': language,
                    'books': self._format_books_for_response(books[:8])
                }
                
                return result
            
            # ==============================================
            # 4. SISTEMA HÍBRIDO INTELIGENTE DE BUSCA
            # ==============================================
            
            user_profile = self._extract_user_profile(message, language)
            books = []
            
            logger.info(f"🎯 SISTEMA HÍBRIDO - Intenção: {intent}")
            logger.info(f"📊 Perfil extraído: {user_profile}")
            
            try:
                # Análise de contexto inteligente
                context_analysis = self._analyze_context(
                    message, 
                    conversation_history, 
                    last_recommendations, 
                    language
                )
                logger.info(f"🧠 Análise de contexto: {context_analysis}")
                
                # DECISÃO INTELIGENTE: Como buscar livros?
                search_strategy = self._determine_search_strategy(
                    message, 
                    intent, 
                    context_analysis, 
                    last_recommendations
                )
                logger.info(f"🎯 Estratégia de busca: {search_strategy}")
                
                # Construir query baseada na mensagem atual
                search_query = self._build_search_query(message, user_profile)
                logger.info(f"🔍 Query base: {search_query}")
                
                # Executar estratégia de busca
                if search_strategy == "new_search":
                    logger.info("🔄 Busca completamente nova")
                    books = await self._intelligent_search(
                        search_query, user_profile, conversation_history, language
                    )
                    
                elif search_strategy == "context_boosted":
                    logger.info("🚀 Busca com boost de contexto")
                    
                    if last_recommendations:
                        context_keywords = self._extract_keywords_from_books(last_recommendations[:2])
                        boosted_query = f"{search_query} {context_keywords}"
                        logger.info(f"🔍 Query com boost: {boosted_query}")
                        books = await self._intelligent_search(
                            boosted_query, user_profile, conversation_history, language
                        )
                    else:
                        books = await self._intelligent_search(
                            search_query, user_profile, conversation_history, language
                        )
                        
                elif search_strategy == "similar_to_previous":
                    logger.info("📚 Buscando livros similares aos anteriores")
                    
                    similar_books = []
                    for book_dict in last_recommendations[:2]:
                        query = f"{book_dict.get('title', '')}"
                        if book_dict.get('authors'):
                            query += f" {book_dict.get('authors')[0]}"
                        
                        similar = self.search_engine.search_by_semantic(query, k=4)
                        similar_books.extend(similar)
                    
                    books = self._remove_duplicate_books(similar_books)[:8]
                    
                elif search_strategy == "use_previous_only":
                    logger.info("💾 Usando apenas livros já recomendados")
                    
                    previous_books = []
                    for book_dict in last_recommendations:
                        book_result = self._book_dict_to_result(book_dict)
                        previous_books.append(book_result)
                    
                    books = previous_books[:8]
                
                else:
                    logger.info("⚡ Fallback: busca normal")
                    books = self.search_engine.search_by_semantic(search_query, k=8)
                    
            except Exception as e:
                logger.error(f"❌ Erro no sistema híbrido de busca: {e}")
                # Fallback para busca simples
                search_query = self._build_search_query(message, user_profile)
                logger.info(f"🔄 Fallback: busca simples com query: {search_query}")
                books = self.search_engine.search_by_semantic(search_query, k=5)
            
            # Garantir que temos livros para recomendar
            if not books and last_recommendations:
                logger.info(f"⚠️ Nenhum livro novo encontrado, usando recomendações anteriores")
                books = [self._book_dict_to_result(book) for book in last_recommendations[:3]]
            
            # Converter BookResult para dicionários antes de passar para response_generator
            books_dicts = self._convert_book_results_to_dicts(books[:5])
            
            # Gerar resposta PERSONALIZADA COM HISTÓRICO
            response_text = await self.response_generator.generate_personalized_recommendation(
                user_message=message,
                books=books_dicts,
                intent=intent,
                language=language,
                conversation_history=conversation_history
            )
            
            # ==============================================
            # 5. SALVAR NO REDIS PARA PRÓXIMAS INTERAÇÕES
            # ==============================================
            
            if self.book_conversation_service:
                # Salvar mensagem do usuário
                self.book_conversation_service.context_manager.add_message(
                    session_id, 'user', message, intent=intent
                )
                
                # Salvar resposta do assistente COM LIVROS (dicionários)
                response_books = self._convert_book_results_to_dicts(books[:3])
                self.book_conversation_service.context_manager.add_message(
                    session_id, 'assistant', response_text, 
                    books=response_books, 
                    intent=intent
                )
                
                logger.info(f"💾 Salvo no Redis - Total mensagens: {len(conversation_history) + 2}")
            
            # ==============================================
            # 6. PREPARAR RESPOSTA FINAL
            # ==============================================
            
            result = {
                'response': response_text,
                'intent': intent,
                'user_profile': user_profile,
                'books_found': len(books),
                'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                'session_id': session_id,
                'language': language,
                'books': self._format_books_for_response(books[:8]),
                'metadata': {
                    'has_conversation_history': len(conversation_history) > 0,
                    'previous_books_count': len(last_recommendations),
                    'is_continuation': bool(last_recommendations and context_analysis.get('is_continuation', False))
                }
            }
            
            logger.info(f"✅ Processamento concluído - Livros: {len(books)}, Tempo: {result['processing_time_seconds']:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar mensagem: {e}", exc_info=True)
            raise

    # ==============================================
    # FUNÇÕES AUXILIARES
    # ==============================================

    def _is_reference_to_previous_books(self, message: str, last_recommendations: List[Dict], 
                                    conversation_history: List[Dict], language: str) -> bool:
        """Verifica se a mensagem faz referência a livros anteriores"""
        if not last_recommendations:
            return False
        
        message_lower = message.lower()
        
        # Palavras-chave que indicam referência a anterior
        reference_keywords = {
            'pt': ['aquele', 'esse', 'desse', 'desses', 'que você', 'você me', 'mencionou', 'falou', 'citou', 
                'anterior', 'antes', 'primeiro', 'segundo', 'terceiro', 'último', 'recomendou'],
            'en': ['that', 'this', 'the one', 'you said', 'mentioned', 'talked', 'cited', 
                'previous', 'before', 'first', 'second', 'third', 'last', 'recommended']
        }
        
        keywords = reference_keywords.get(language, reference_keywords['pt'])
        
        # Verificar keywords
        if any(keyword in message_lower for keyword in keywords):
            return True
        
        # Verificar se menciona títulos específicos
        for book in last_recommendations:
            title = book.get('title', '').lower()
            if title and title in message_lower:
                return True
        
        return False

    def _build_search_query(self, message: str, user_profile: Dict) -> str:
        """Constrói query de busca considerando perfil do usuário"""
        query_parts = []
        
        # Adicionar termos da mensagem
        query_parts.append(message)
        
        # Adicionar termos baseados no perfil
        if user_profile.get('study_area'):
            query_parts.append(user_profile['study_area'])
        
        if user_profile.get('level') == 'beginner':
            query_parts.append("beginner introduction fundamentals")
        elif user_profile.get('level') == 'advanced':
            query_parts.append("advanced expert professional")
        
        if 'learning' in user_profile.get('goals', []):
            query_parts.append("learning education tutorial guide")
        
        return ' '.join(query_parts)

    def _analyze_intent(self, message: str) -> str:
        """Analisa a intenção da mensagem"""
        message_lower = message.lower().strip()
        
        logger.info(f"📝 Analisando intenção da mensagem: '{message_lower}'")
        
        # Verificar se é sobre CARREIRA/LIDERANÇA
        career_keywords = [
            'promovido', 'promoção', 'carreira', 'liderança', 'líder', 'gestor', 'gerente',
            'promoted', 'promotion', 'career', 'leadership', 'leader', 'manager'
        ]
        
        if any(keyword in message_lower for keyword in career_keywords):
            logger.info("🎯 Intenção: career_growth (palavras de carreira detectadas)")
            return 'career_growth'
        
        # Verificar se menciona GÊNERO específico
        genre_keywords = [
            'fantasia', 'fantasy', 'ficção científica', 'sci-fi', 'science fiction',
            'romance', 'terror', 'horror', 'mistério', 'mystery', 'suspense',
            'história', 'history', 'biografia', 'biography', 'autoajuda', 'self-help',
            'negócios', 'business', 'ciência', 'science', 'tecnologia', 'technology'
        ]
        
        # Se menciona gênero E palavras de recomendação, é "general"
        book_request_words = ['recomende', 'recomendação', 'sugestão', 'livro', 'livros',
                            'recommend', 'recommendation', 'suggestion', 'book', 'books']
        
        has_genre = any(genre in message_lower for genre in genre_keywords)
        has_book_request = any(word in message_lower for word in book_request_words)
        
        if has_genre and has_book_request:
            logger.info("🎯 Intenção: general (gênero + pedido de livro detectado)")
            return 'general'
        
        # Verificar se menciona AUTOR específico
        author_keywords = [
            'autor', 'autora', 'writer', 'author', 'escritor', 'escritora',
            'livros de', 'obras de', 'books by'
        ]
        
        # Verificar autores conhecidos
        known_authors = [
            'j.k. rowling', 'jk rowling', 'stephen king', 'george orwell',
            'agatha christie', 'j.r.r. tolkien', 'dan brown', 'paulo coelho',
            'suzanne collins', 'veronica roth', 'rick riordan'
        ]
        
        has_author_keyword = any(keyword in message_lower for keyword in author_keywords)
        has_known_author = any(author in message_lower for author in known_authors)
        
        if has_author_keyword or has_known_author:
            logger.info("🎯 Intenção: author (autor detectado)")
            return 'author'
        
        # Se tem palavras de pedido de livros, é general
        if has_book_request:
            logger.info("🎯 Intenção: general (solicitação de livros detectada)")
            return 'general'
        
        # Verificar closing (agradecimento/despedida)
        closing_keywords = [
            'obrigado', 'obrigada', 'valeu', 'thank you', 'thanks', 'bye', 'tchau',
            'até logo', 'goodbye', 'adeus'
        ]
        
        # Só é closing se NÃO tem palavras de pedido
        is_closing = any(keyword in message_lower for keyword in closing_keywords)
        if is_closing and not has_book_request:
            logger.info("🎯 Intenção: closing (agradecimento/despedida)")
            return 'closing'
        
        # Padrão
        logger.info("🎯 Intenção: social (padrão)")
        return 'social'

    def _extract_author(self, message: str) -> Optional[str]:
        """Extrai autor da mensagem do usuário"""
        message_lower = message.lower()
        
        # Autores conhecidos (com variações)
        known_authors = {
            'j.k. rowling': ['j.k. rowling', 'jk rowling', 'joanne rowling', 'rowling'],
            'stephen king': ['stephen king', 'king'],
            'george orwell': ['george orwell', 'orwell'],
            'agatha christie': ['agatha christie', 'christie'],
            'j.r.r. tolkien': ['j.r.r. tolkien', 'tolkien'],
            'suzanne collins': ['suzanne collins', 'collins'],
            'paulo coelho': ['paulo coelho', 'coelho'],
            'dan brown': ['dan brown', 'brown'],
            'rick riordan': ['rick riordan', 'riordan'],
            'veronica roth': ['veronica roth', 'roth'],
        }
        
        # Procurar autores conhecidos
        for author, variations in known_authors.items():
            for variation in variations:
                if variation in message_lower:
                    return author
        
        # Padrões para extrair nomes de autores
        import re
        author_patterns = [
            r'(?:do|da|de)\s+(?:autor|autora|escritor|escritora|writer|author)\s+["\']?(.+?)["\']?(?:\s|$|\.|,)',
            r'(?:livros?|obras?)\s+(?:do|da|de)\s+["\']?(.+?)["\']?(?:\s|$|\.|,)',
            r'["\'](.+?)["\']\s+(?:é|são)\s+(?:o\s+)?(?:autor|autora|escritor|escritora)',
        ]
        
        for pattern in author_patterns:
            matches = re.findall(pattern, message_lower)
            if matches:
                author_name = matches[0].strip()
                # Limpar e capitalizar
                author_name = ' '.join([word.capitalize() for word in author_name.split()])
                return author_name
        
        return None

    def _extract_user_profile(self, message: str, language: str) -> Dict:
        """Extrai perfil do usuário da mensagem"""
        profile = {
            'interests': [],
            'study_area': None,
            'level': None,
            'goals': [],
            'preferences': []
        }
        
        message_lower = message.lower()
        
        # Detectar se é sobre quadrinhos/personagens
        comic_keywords = [
            'homem-aranha', 'spider-man', 'marvel', 'dc comics', 'dc',
            'super-herói', 'super hero', 'superhero', 'quadrinhos', 'comics', 'hq',
            'batman', 'superman', 'x-men', 'avengers', 'thor', 'iron man', 'hulk'
        ]
        
        if any(keyword in message_lower for keyword in comic_keywords):
            profile['interests'].append('comics')
            profile['interests'].append('superheroes')
            profile['preferences'].append('action')
            profile['preferences'].append('adventure')
            return profile
        
        # Detectar áreas de estudo
        study_areas = {
            'computer science': [
                'computer science', 'ciência da computação', 'ciencia da computacao',
                'engenharia de software', 'software engineering'
            ],
            'data science': [
                'data science', 'ciência de dados', 'ciencia de dados',
                'machine learning', 'aprendizado de máquina'
            ],
            'artificial intelligence': [
                'artificial intelligence', 'inteligência artificial', 'inteligencia artificial',
                'neural network', 'rede neural'
            ],
            'engineering': [
                'engineering', 'engenharia', 
                'engenharia civil', 'civil engineering',
                'engenharia mecânica', 'mechanical engineering'
            ],
            'business': [
                'business', 'negócios', 'negocios',
                'administração', 'administracao', 'administration'
            ],
            'design': [
                'design', 'ux design', 'ui design',
                'user experience', 'user interface'
            ],
            'medicine': [
                'medicina', 'medicine', 'médico', 'medico',
                'saúde', 'saude', 'health'
            ]
        }
        
        for area, keywords in study_areas.items():
            keyword_matches = [keyword for keyword in keywords if keyword in message_lower]
            
            if keyword_matches:
                if area == 'data science':
                    exact_matches = ['data science', 'ciência de dados', 'ciencia de dados']
                    if any(exact in message_lower for exact in exact_matches) or len(keyword_matches) >= 2:
                        profile['study_area'] = area
                        profile['interests'].append(area)
                        break
                else:
                    profile['study_area'] = area
                    profile['interests'].append(area)
                    break
        
        # Detectar objetivos
        if any(word in message_lower for word in ['learn', 'aprender', 'study', 'estudar', 'curso', 'course']):
            profile['goals'].append('learning')
        if any(word in message_lower for word in ['project', 'projeto', 'work', 'trabalho', 'aplicação', 'application']):
            profile['goals'].append('project')
        if any(word in message_lower for word in ['career', 'carreira', 'job', 'emprego', 'profissional', 'professional']):
            profile['goals'].append('career')
        
        # Detectar nível
        if any(word in message_lower for word in ['beginner', 'iniciante', 'starting', 'básico', 'basic']):
            profile['level'] = 'beginner'
        elif any(word in message_lower for word in ['intermediate', 'intermediário', 'intermediario', 'experienced']):
            profile['level'] = 'intermediate'
        elif any(word in message_lower for word in ['advanced', 'avançado', 'avancado', 'expert', 'especialista']):
            profile['level'] = 'advanced'
        
        return profile

    # ==============================================
    # MÉTODOS RESTANTES (mantidos do código original)
    # ==============================================

    def search_books(self, search_params: SearchRequest) -> List[Dict]:
        """Busca livros com cache"""
        if not self.initialized:
            raise Exception("Serviço não inicializado")
        
        # Preparar filtros
        filters = {}
        if search_params.genre:
            filters['genre'] = search_params.genre
        if search_params.author:
            filters['author'] = search_params.author
        if search_params.min_rating:
            filters['min_rating'] = search_params.min_rating
        
        # Usar cache para diferentes métodos de busca
        if search_params.method == 'semantic' and search_params.query:
            books = self._cached_search(
                method='semantic',
                query=search_params.query,
                filters=filters if filters else None,
                limit=search_params.limit
            )
        
        elif search_params.method == 'genre' and search_params.genre:
            books = self._cached_search(
                method='genre',
                query=search_params.genre,
                filters=None,
                limit=search_params.limit
            )
        
        elif search_params.method == 'author' and search_params.author:
            books = self._cached_search(
                method='author',
                query=search_params.author,
                filters=None,
                limit=search_params.limit
            )
        
        elif search_params.method == 'popularity':
            books = self._cached_search(
                method='popularity',
                query='',
                filters=filters if filters else None,
                limit=search_params.limit
            )
        else:
            books = []
        
        # Log de cache
        total_searches = self.cache_hits + self.cache_misses
        if total_searches > 0:
            hit_rate = (self.cache_hits / total_searches) * 100
            logger.debug(f"Cache stats: Hits={self.cache_hits}, Misses={self.cache_misses}, Rate={hit_rate:.1f}%")
        
        return self._format_books_for_response(books)
    
    def _cached_search(self, method: str, query: str, filters: Dict, limit: int):
        """Busca com cache"""
        cache_key = f"{method}:{query}:{str(filters)}:{limit}"
        
        # Verificar cache
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if datetime.now() - cache_entry['timestamp'] < self.cache_ttl:
                self.cache_hits += 1
                return cache_entry['results']
        
        self.cache_misses += 1
        
        # Executar busca
        if method == 'semantic':
            results = self.search_engine.search_by_semantic(query, filters, limit)
        elif method == 'genre':
            results = self.search_engine.search_by_genre(query, limit)
        elif method == 'author':
            results = self.search_engine.search_by_author(query, limit)
        elif method == 'popularity':
            results = self.search_engine.search_by_popularity(filters, limit)
        else:
            results = []
        
        # Armazenar no cache
        self.search_cache[cache_key] = {
            'timestamp': datetime.now(),
            'results': results
        }
        
        # Limitar tamanho do cache
        if len(self.search_cache) > 100:
            oldest_key = min(self.search_cache.keys(), key=lambda k: self.search_cache[k]['timestamp'])
            del self.search_cache[oldest_key]
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Retorna estatísticas do cache"""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self.search_cache),
            'cache_hit_rate': (self.cache_hits / (self.cache_hits + self.cache_misses) * 100 
                             if (self.cache_hits + self.cache_misses) > 0 else 0),
            'cache_entries': list(self.search_cache.keys())[:10]
        }
    
    def clear_cache(self):
        """Limpa o cache"""
        self.search_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Cache limpo")
    
    def get_book_by_id(self, book_id: int) -> Optional[Dict]:
        """Busca livro por ID"""
        if not self.initialized:
            raise Exception("Serviço não inicializado")
        
        book_result = self.search_engine.get_book_by_id(book_id)
        
        if not book_result:
            return None
        
        return {
            'book_id': book_result.book_id,
            'title': book_result.title,
            'authors': book_result.authors,
            'genres': book_result.genres,
            'rating': book_result.rating,
            'num_ratings': book_result.num_ratings,
            'description': book_result.description,
            'price': book_result.price
        }
    
    def get_agent_stats(self) -> Dict:
        """Obtém estatísticas do agente"""
        return {
            'initialized': self.initialized,
            'conversations_count': len(self.conversation_history),
            'data_size': len(self.data_loader.data) if self.data_loader else 0,
            'last_initialization': getattr(self, '_last_init_time', None)
        }
    
    def get_search_stats(self) -> Dict:
        """Obtém estatísticas de busca"""
        if not self.search_engine:
            return {}
        
        return self.search_engine.get_search_stats()
    
    def get_ollama_stats(self) -> Dict:
        """Obtém estatísticas do Ollama"""
        if not self.ollama_service:
            return {'connected': False}
        
        return {
            'connected': True,
            'model': self.ollama_service.model,
            'performance': self.ollama_service.get_performance_stats()
        }
    
    def get_embedding_stats(self) -> Dict:
        """Obtém estatísticas do sistema de embeddings"""
        if not self.embedding_service:
            return {}
        
        return {
            'model_name': self.embedding_service.model_name,
            'gpu_enabled': self.embedding_service.use_gpu,
            'index_built': self.embedding_service.index_built,
            'index_size': self.embedding_service.index.ntotal if self.embedding_service.index else 0
        }
    
    def is_gpu_available(self) -> bool:
        """Verifica se GPU está disponível"""
        return torch.cuda.is_available()
    
    def is_data_loaded(self) -> bool:
        """Verifica se dados estão carregados"""
        return self.data_loader is not None and self.data_loader.data is not None
    
    def is_model_loaded(self) -> bool:
        """Verifica se modelo de embeddings está carregado"""
        return self.embedding_service is not None and self.embedding_service.embedding_model is not None
    
    def is_index_built(self) -> bool:
        """Verifica se índice está construído"""
        return self.embedding_service is not None and self.embedding_service.index_built
    
    def is_ollama_connected(self) -> bool:
        """Verifica se Ollama está conectado"""
        if not self.ollama_service:
            return False
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            connected = loop.run_until_complete(self.ollama_service.health_check())
            loop.close()
            return connected
        except:
            return False
    
    def clear_session_data(self, session_id: str) -> Dict[str, any]:
        """Limpa os dados de uma sessão específica"""
        logger.info(f"🧹 Solicitada limpeza da sessão: {session_id}")
        
        try:
            if not hasattr(self, 'book_conversation_service') or not self.book_conversation_service:
                logger.warning("❌ Serviço de conversação não disponível")
                return {
                    "success": False,
                    "error": "Serviço de conversação não disponível",
                    "session_id": session_id
                }
            
            result = self.book_conversation_service.context_manager.clear_session_data(session_id)
            
            logger.info(f"✅ Resultado da limpeza: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao limpar sessão {session_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }

    def clear_all_sessions(self) -> Dict[str, any]:
        """Limpa todas as sessões"""
        logger.warning("⚠️  Solicitada limpeza de TODAS as sessões")
        
        try:
            if not hasattr(self, 'book_conversation_service') or not self.book_conversation_service:
                logger.warning("❌ Serviço de conversação não disponível")
                return {
                    "success": False,
                    "error": "Serviço de conversação não disponível"
                }
            
            result = self.book_conversation_service.context_manager.clear_all_sessions()
            
            if hasattr(self, 'search_cache'):
                self.search_cache.clear()
                logger.info("🧹 Cache local limpo")
            
            logger.info(f"✅ Resultado da limpeza total: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao limpar todas as sessões: {e}")
            return {
                "success": False,
                "error": str(e)
            }   
        
    async def translate_query(self, query: str, source_lang: str = 'pt', target_lang: str = 'en') -> str:
        """Traduz uma query para o idioma de destino"""
        if not self.translation_service:
            self.translation_service = get_translation_service()
        
        if source_lang == target_lang:
            return query
        
        try:
            return await self.translation_service.translate_to_english(query, source_lang)
        except Exception as e:
            logger.error(f"Erro ao traduzir query: {e}")
            return query