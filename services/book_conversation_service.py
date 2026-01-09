# D:\Django\book_agent\services\book_conversation_service.py

import re
import logging
from typing import Dict, List, Optional, Tuple
from .conversation_context import ConversationContextManager

logger = logging.getLogger(__name__)

class BookConversationService:
    """Serviço para conversas específicas sobre livros"""
    
    def __init__(self, ollama_service, data_loader, search_engine):
        self.ollama_service = ollama_service
        self.data_loader = data_loader
        self.search_engine = search_engine
        self.context_manager = ConversationContextManager()
        
        # Frases que NÃO são livros (stop phrases)
        self.stop_phrases = {
            'pt': [
                'me recomenda', 'recomenda', 'quais', 'livros',
                'tenho interesse', 'gostaria', 'para estudar',
                'sobre', 'de', 'em', 'algum', 'alguma',
                'qual o contexto', 'em relação aos',
                'para iniciantes', 'tem livros',
                'mudando de assunto', 'gostaria de estudar',
                'qual o', 'contexto', 'relação', 'outros',
                'tem disponível', 'tem disponivel', 'disponível',
                'da saga', 'saga', 'harry potter',
                'esse livro', 'este livro', 'aquele livro',
                'o livro', 'a livro', 'os livros', 'as livros',
                'queria saber', 'poderia me dizer',
                'você tem', 'você conhece',
                'qual seria', 'quais são', 'quais seriam'
            ],
            'en': [
                'recommend', 'recommends', 'which', 'books',
                'i have interest', 'i would like', 'to study',
                'about', 'of', 'in', 'some', 'any',
                'what is the context', 'in relation to',
                'for beginners', 'are there books',
                'changing subject', 'i would like to study',
                'what is the', 'context', 'relation', 'others',
                'available', 'from the saga', 'saga',
                'this book', 'that book', 'the book',
                'a book', 'books', 'the books',
                'i wanted to know', 'could you tell me',
                'do you have', 'do you know',
                'what would be', 'what are', 'which would be'
            ]
        }
        
        # Padrões para detectar livros (somente casos claros)
        self.book_reference_patterns = {
            'pt': [
                # Títulos entre aspas (o mais seguro)
                r'["\'](.+?)["\']',
                # "O livro X" (só se X tiver mais de 4 caracteres)
                r'\b(?:o|a)\s+livro\s+["\']?([^"\'\n]{5,50})["\']?(?:\s|$|\.|\?|,)',
                # "Livro chamado X"
                r'\blivro\s+(?:chamado|intitulado)\s+["\']?([^"\'\n]{5,50})["\']?(?:\s|$|\.)',
                # "X é um livro"
                r'["\']?([^"\'\n]{5,50})["\']?\s+(?:é|foi)\s+(?:um\s+)?livro',
            ],
            'en': [
                # Titles in quotes (safest)
                r'["\'](.+?)["\']',
                # "The book X" (only if X has more than 4 chars)
                r'\b(?:the|a)\s+book\s+["\']?([^"\'\n]{5,50})["\']?(?:\s|$|\.|\?|,)',
                # "Book called X"
                r'\bbook\s+(?:called|titled)\s+["\']?([^"\'\n]{5,50})["\']?(?:\s|$|\.)',
                # "X is a book"
                r'["\']?([^"\'\n]{5,50})["\']?\s+(?:is|was)\s+(?:a\s+)?book',
            ]
        }
        
        # Padrões que são SEMPRE solicitações gerais (não conversa sobre livro)
        self.always_general_patterns = {
            'pt': [
                r'me\s+recomenda[dr]?\s+livros?',
                r'quais\s+livros',
                r'tenho\s+interesse\s+em',
                r'gostaria\s+de\s+(?:estudar|aprender|saber)',
                r'para\s+estudar',
                r'livros\s+(?:para|sobre|de)',
                r'tem\s+livros',
                r'buscar\s+livros',
                r'procurar\s+livros',
                r'queria\s+livros',
                r'quer[ia]?\s+livros',
                r'preciso\s+de\s+livros',
                r'mudando\s+de\s+assunto',
                r'gostaria\s+de\s+estudar',
                r'qual\s+o\s+contexto',
                r'em\s+relação\s+aos',
                r'da\s+saga',
                r'para\s+iniciantes'
            ],
            'en': [
                r'recommend\s+(?:me\s+)?books?',
                r'which\s+books',
                r'i\s+have\s+interest\s+in',
                r'i\s+would\s+like\s+to\s+(?:study|learn|know)',
                r'to\s+study',
                r'books\s+(?:for|about|on)',
                r'are\s+there\s+books',
                r'search\s+for\s+books',
                r'look\s+for\s+books',
                r'i\s+wanted\s+books',
                r'i\s+want\s+books',
                r'i\s+need\s+books',
                r'changing\s+subject',
                r'i\s+would\s+like\s+to\s+study',
                r'what\s+is\s+the\s+context',
                r'in\s+relation\s+to',
                r'from\s+the\s+saga',
                r'for\s+beginners'
            ]
        }
    
    def _is_stop_phrase(self, text: str, language: str = 'pt') -> bool:
        """Verifica se o texto é uma frase comum, não um título"""
        text_lower = text.lower()
        stop_phrases = self.stop_phrases.get(language, self.stop_phrases['pt'])
        
        # Verificar se contém alguma stop phrase
        for phrase in stop_phrases:
            if phrase in text_lower:
                return True
        
        # Verificar se é muito curto
        if len(text.strip()) < 4:
            return True
        
        # Verificar se são só palavras comuns
        common_words = {
            'pt': ['qual', 'que', 'como', 'onde', 'quando', 'porque', 'para', 'com', 'sem', 'em', 'de', 'e', 'ou', 'a', 'o', 'os', 'as', 'um', 'uma', 'uns', 'umas'],
            'en': ['what', 'which', 'how', 'where', 'when', 'why', 'for', 'with', 'without', 'in', 'of', 'and', 'or', 'a', 'an', 'the']
        }
        
        words = text_lower.split()
        if len(words) <= 3:
            # Se tem 3 palavras ou menos, verificar se são comuns
            common_words_list = common_words.get(language, common_words['pt'])
            all_common = all(word in common_words_list for word in words if len(word) > 2)
            if all_common:
                return True
        
        return False
    
    def _is_always_general_request(self, message: str, language: str = 'pt') -> bool:
        """Verifica se a mensagem é sempre uma solicitação geral"""
        message_lower = message.lower()
        patterns = self.always_general_patterns.get(language, self.always_general_patterns['pt'])
        
        for pattern in patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return True
        
        return False
    
    def detect_book_reference(self, message: str, language: str = 'pt') -> Tuple[Optional[str], Optional[int]]:
        """Detecta referência a livro específico na mensagem - VERSÃO CORRIGIDA"""
        logger.info(f"🔍 Analisando mensagem para referência a livro: '{message}'")
        
        # PRIMEIRO: Verificar se é uma solicitação geral (NÃO conversa sobre livro)
        if self._is_always_general_request(message, language):
            logger.info("🎯 É solicitação geral, não referência a livro")
            return None, None
        
        # SEGUNDO: Procurar IDs
        id_patterns = [
            r'\bID\s*[:#]?\s*(\d{4,})\b',
            r'\bid\s*[:#]?\s*(\d{4,})\b',
            r'\blivro\s+(?:número|num|nº|n°|#)?\s*(\d{4,})\b',
            r'\b(\d{4,})\s+(?:º|°)?\s+(?:livro|book)\b',
        ]
        
        for pattern in id_patterns:
            matches = re.findall(pattern, message)
            for match in matches:
                if match and match.isdigit():
                    try:
                        book_id = int(match)
                        logger.info(f"📘 ID detectado: {book_id}")
                        return None, book_id
                    except:
                        pass
        
        # TERCEIRO: Procurar títulos entre aspas (o método mais seguro)
        # Aspas duplas
        double_quote_matches = re.findall(r'"([^"]+)"', message)
        if double_quote_matches:
            title = double_quote_matches[0].strip()
            if len(title) > 3 and not self._is_stop_phrase(title, language):
                logger.info(f"📘 Título entre aspas detectado: '{title}'")
                return title, None
        
        # Aspas simples
        single_quote_matches = re.findall(r"'([^']+)'", message)
        if single_quote_matches:
            title = single_quote_matches[0].strip()
            if len(title) > 3 and not self._is_stop_phrase(title, language):
                logger.info(f"📘 Título entre aspas simples detectado: '{title}'")
                return title, None
        
        # QUARTO: Títulos mencionados com "livro" (com validação)
        patterns = self.book_reference_patterns.get(language, self.book_reference_patterns['pt'])
        
        for pattern in patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Se o padrão tem grupos, pegar o grupo que contém o título
                    for item in match:
                        if item and len(item.strip()) > 3:
                            title = item.strip()
                            if not self._is_stop_phrase(title, language):
                                logger.info(f"📘 Título detectado no padrão: '{title}'")
                                return title, None
                elif match and len(match.strip()) > 3:
                    title = match.strip()
                    if not self._is_stop_phrase(title, language):
                        logger.info(f"📘 Título detectado: '{title}'")
                        return title, None
        
        logger.info(f"🔍 Nenhuma referência a livro detectada")
        return None, None
    
    def detect_multiple_books(self, message: str, language: str = 'pt') -> List[Tuple[Optional[str], Optional[int]]]:
        """Detecta múltiplos livros na mesma mensagem - VERSÃO CORRIGIDA"""
        logger.info(f"🔍 Detectando múltiplos livros em: '{message}'")
        
        # VERIFICAÇÃO CRÍTICA: Se for uma solicitação geral, NÃO detectar livros
        if self._is_always_general_request(message, language):
            logger.info(f"🎯 É uma solicitação geral, NÃO conversa sobre livro específico")
            return []  # Retorna lista vazia - NÃO detectar livros
        
        detected_books = []
        
        # 1. Detectar IDs (se houver)
        id_patterns = [
            r'\b(ID|id|Id)\s*[:#]?\s*(\d{3,})\b',
            r'\blivro\s+(?:número|num|nº|n°|#)?\s*(\d{3,})\b',
            r'\b(\d{4,})\s+(?:º|°)?\s+(?:livro|book)\b',
        ]
        
        for pattern in id_patterns:
            matches = re.findall(pattern, message)
            for match in matches:
                if isinstance(match, tuple):
                    # Extrair o número do match
                    for item in match:
                        if item and item.isdigit():
                            book_id = int(item)
                            if (None, book_id) not in detected_books:
                                detected_books.append((None, book_id))
                                logger.info(f"📘 ID detectado: {book_id}")
                            break
                elif match and match.isdigit():
                    book_id = int(match)
                    if (None, book_id) not in detected_books:
                        detected_books.append((None, book_id))
                        logger.info(f"📘 ID detectado: {book_id}")
        
        # 2. Detectar títulos entre aspas (MAIS SEGURO)
        # Primeiro: títulos entre aspas duplas
        double_quote_titles = re.findall(r'"([^"]+)"', message)
        for title in double_quote_titles:
            if len(title.strip()) > 3 and not self._is_stop_phrase(title, language):
                clean_title = title.strip()
                if (clean_title, None) not in detected_books:
                    detected_books.append((clean_title, None))
                    logger.info(f"📘 Título entre aspas detectado: '{clean_title}'")
        
        # Segundo: títulos entre aspas simples
        single_quote_titles = re.findall(r"'([^']+)'", message)
        for title in single_quote_titles:
            if len(title.strip()) > 3 and not self._is_stop_phrase(title, language):
                clean_title = title.strip()
                if (clean_title, None) not in detected_books:
                    detected_books.append((clean_title, None))
                    logger.info(f"📘 Título entre aspas simples detectado: '{clean_title}'")
        
        # 3. Detectar títulos mencionados explicitamente (com validação)
        patterns = self.book_reference_patterns.get(language, self.book_reference_patterns['pt'])
        
        for pattern in patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Se o padrão tem grupos, pegar o título
                    for item in match:
                        if item and len(item.strip()) > 3:
                            title = item.strip()
                            if not self._is_stop_phrase(title, language):
                                if (title, None) not in detected_books:
                                    detected_books.append((title, None))
                                    logger.info(f"📘 Título mencionado detectado: '{title}'")
                elif match and len(match.strip()) > 3:
                    title = match.strip()
                    if not self._is_stop_phrase(title, language):
                        if (title, None) not in detected_books:
                            detected_books.append((title, None))
                            logger.info(f"📘 Título detectado: '{title}'")
        
        logger.info(f"📚 Total de livros detectados: {len(detected_books)}")
        for i, (title, book_id) in enumerate(detected_books, 1):
            logger.info(f"  {i}. Título: '{title}', ID: {book_id}")
        
        return detected_books
    
    def detect_single_book_reference(self, message: str, language: str = 'pt') -> Tuple[Optional[str], Optional[int]]:
        """Detecta um único livro (versão anterior mantida para compatibilidade)"""
        return self.detect_book_reference(message, language)
    
    def _looks_like_book_title(self, text: str, original_message: str) -> bool:
        """Verifica se um texto parece ser um título de livro"""
        # Se estiver entre aspas na mensagem original, provavelmente é título
        if f'"{text}"' in original_message or f"'{text}'" in original_message:
            return True
        
        # Verificar se é stop phrase
        if self._is_stop_phrase(text, 'pt'):
            return False
        
        # Títulos geralmente têm palavras significativas
        words = text.split()
        if len(words) < 1 or len(words) > 7:  # Títulos muito curtos ou muito longos
            return False
        
        # Verificar padrões comuns em títulos
        title_patterns = [
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Palavras capitalizadas: "The Hunger Games"
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # Pelo menos 3 palavras capitalizadas
            r'\b(?:and|the|of|in|for|with|without|e|o|a|os|as|de|da|do|das|dos)\b',  # Contém palavras comuns em títulos
        ]
        
        for pattern in title_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _extract_title_heuristic(self, message: str, language: str) -> Optional[str]:
        """Extrai título por heurística quando padrões não funcionam"""
        
        # Palavras que indicam início de referência a título
        start_indicators = {
            'pt': ['livro', 'obra', 'romance', 'conto', 'novela', 'título', 'chamado', 'intitulado'],
            'en': ['book', 'novel', 'story', 'tale', 'title', 'called', 'named']
        }
        
        # Palavras que indicam fim de referência
        end_indicators = {
            'pt': ['é', 'foi', 'tem', 'possui', 'é um', 'do autor', 'escrito', 'publicado'],
            'en': ['is', 'was', 'has', 'by', 'written', 'published', 'author']
        }
        
        indicators = start_indicators.get(language, start_indicators['pt'])
        end_words = end_indicators.get(language, end_indicators['pt'])
        
        message_lower = message.lower()
        
        for indicator in indicators:
            if indicator in message_lower:
                # Encontrar posição do indicador
                idx = message_lower.find(indicator)
                if idx != -1:
                    # Extrair texto após o indicador
                    start_pos = idx + len(indicator)
                    remaining = message[start_pos:].strip()
                    
                    # Encontrar onde termina o título
                    title_end = len(remaining)
                    for end_word in end_words:
                        if end_word in remaining.lower():
                            end_pos = remaining.lower().find(end_word)
                            if end_pos > 0 and end_pos < title_end:
                                title_end = end_pos
                    
                    # Extrair possível título
                    possible_title = remaining[:title_end].strip()
                    
                    # Limpar pontuação no início/fim
                    possible_title = possible_title.strip(' "\':;,.-')
                    
                    # Verificar se tem tamanho razoável e não é stop phrase
                    if 3 <= len(possible_title) <= 100 and not self._is_stop_phrase(possible_title, language):
                        # Capitalizar primeira letra de cada palavra (como títulos)
                        words = possible_title.split()
                        capitalized_words = []
                        for word in words:
                            if word and word[0].isalpha():
                                capitalized_words.append(word[0].upper() + word[1:].lower())
                            else:
                                capitalized_words.append(word)
                        
                        return ' '.join(capitalized_words)
        
        return None
    
    def get_book_from_context(self, session_id: str, book_title: str = None, 
                            book_id: int = None) -> Optional[Dict]:
        """Tenta encontrar livro no contexto ou no dataset completo - VERSÃO COM BUSCA DIFUSA"""
        
        logger.info(f"🔍 Buscando livro - Título: '{book_title}', ID: {book_id}")
        
        # 1. Se tiver ID, buscar por ID primeiro
        if book_id:
            # Primeiro nas recomendações
            recommendations = self.context_manager.get_last_recommendations(session_id)
            if recommendations:
                for book in recommendations:
                    if book.get('book_id') == book_id:
                        logger.info(f"✅ Livro encontrado no contexto por ID: {book_id}")
                        return book
            
            # Depois no dataset completo
            book_result = self.search_engine.get_book_by_id(book_id)
            if book_result:
                logger.info(f"✅ Livro encontrado no dataset por ID: {book_id}")
                return self._book_result_to_dict(book_result)
        
        # 2. Se tiver título, usar busca difusa
        if book_title and book_title.strip():
            # Primeiro, tentar busca exata nas recomendações
            recommendations = self.context_manager.get_last_recommendations(session_id)
            if recommendations:
                query_lower = book_title.lower()
                for book in recommendations:
                    book_title_lower = book.get('title', '').lower()
                    
                    # Verificar várias formas de match
                    if (query_lower == book_title_lower or
                        query_lower in book_title_lower or
                        book_title_lower in query_lower):
                        logger.info(f"✅ Livro encontrado no contexto (match exato/parcial): {book['title']}")
                        return book
            
            # Se não encontrou, usar busca difusa
            fuzzy_match = self.find_book_by_title_fuzzy(book_title, session_id)
            if fuzzy_match:
                logger.info(f"✅ Livro encontrado por busca difusa: {fuzzy_match['title']}")
                return fuzzy_match
        
        logger.warning(f"❌ Livro não encontrado: Título='{book_title}', ID={book_id}")
        return None
    
    def _book_result_to_dict(self, book_result) -> Dict:
        """Converte BookResult para dicionário"""
        return {
            'book_id': book_result.book_id,
            'title': book_result.title,
            'authors': book_result.authors,
            'description': book_result.description,
            'genres': book_result.genres,
            'rating': book_result.rating,
            'num_ratings': book_result.num_ratings
        }
    
    def _extract_authors(self, book) -> List[str]:
        """Extrai autores"""
        authors = book.get('author', [])
        if isinstance(authors, list):
            return [str(a) for a in authors[:2]]
        elif authors and str(authors).strip():
            return [str(authors)]
        return []
    
    def _extract_genres(self, book) -> List[str]:
        """Extrai gêneros"""
        genres = []
        if book.get('main_genre'):
            genres.append(str(book['main_genre']))
        if book.get('all_genres'):
            if isinstance(book['all_genres'], list):
                genres.extend([str(g) for g in book['all_genres'][:2]])
            else:
                genres.append(str(book['all_genres']))
        return list(set(genres))[:3]
    
    async def chat_about_book(self, user_message: str, session_id: str, 
                            language: str = 'pt') -> Dict:
        """Conversa sobre um livro específico"""
        logger.info(f"Conversando sobre livro: {user_message}")
        
        # Detectar referência ao livro
        book_title, book_id = self.detect_book_reference(user_message, language)
        
        if not book_title and not book_id:
            return {
                'response': self._get_no_book_reference_response(language),
                'book': None,
                'context': 'no_book_reference'
            }
        
        # Buscar livro
        book = self.get_book_from_context(session_id, book_title, book_id)
        
        if not book:
            return {
                'response': self._get_book_not_found_response(book_title or f"ID {book_id}", language),
                'book': None,
                'context': 'book_not_found'
            }
        
        # Obter contexto da conversa
        conversation_context = self.context_manager.get_conversation_context(session_id)
        
        # Gerar resposta detalhada com Ollama
        response = await self._generate_book_conversation_response(
            user_message, book, conversation_context, language
        )
        
        # Atualizar contexto
        self.context_manager.add_message(session_id, 'user', user_message)
        self.context_manager.add_message(session_id, 'assistant', response)
        
        return {
            'response': response,
            'book': book,
            'context': 'book_conversation',
            'book_id': book['book_id']
        }
    
    async def _generate_book_conversation_response(self, user_message: str, book: Dict, 
                                                 conversation_context: str, language: str) -> str:
        """Gera resposta de conversa sobre o livro"""
        
        book_context = self._create_detailed_book_context(book)
        
        if language == 'pt':
            prompt = f"""
            VOCÊ É: Um especialista literário e crítico de livros com profundo conhecimento.
            
            CONTEXTO DA CONVERSA:
            {conversation_context}
            
            LIVRO EM DISCUSSÃO:
            {book_context}
            
            PERGUNTA DO USUÁRIO:
            "{user_message}"
            
            SUA TAREFA:
            1. RESPONDER especificamente à pergunta sobre ESTE livro
            2. Fornecer informações precisas baseadas nos dados do livro
            3. Oferecer análise crítica e insights
            4. Se relevante, comparar com outros livros similares
            5. Dar recomendações relacionadas à pergunta
            
            DIRETRIZES:
            - Seja específico e detalhado
            - Use informações reais do livro (gênero, autor, avaliação)
            - Se não souber algo, seja honesto mas sugere onde encontrar a informação
            - Mantenha a conversa envolvente
            - Convide para mais perguntas
            
            RESPOSTA (em português, 3-5 parágrafos):
            """
        else:
            prompt = f"""
            YOU ARE: A literary expert and book critic with deep knowledge.
            
            CONVERSATION CONTEXT:
            {conversation_context}
            
            BOOK BEING DISCUSSED:
            {book_context}
            
            USER QUESTION:
            "{user_message}"
            
            YOUR TASK:
            1. ANSWER specifically about THIS book
            2. Provide accurate information based on book data
            3. Offer critical analysis and insights
            4. If relevant, compare with similar books
            5. Give related recommendations
            
            GUIDELINES:
            - Be specific and detailed
            - Use real book information (genre, author, rating)
            - If you don't know something, be honest but suggest where to find info
            - Keep the conversation engaging
            - Invite further questions
            
            RESPONSE (in English, 3-5 paragraphs):
            """
        
        try:
            response = await self.ollama_service.chat([
                {"role": "user", "content": prompt}
            ])
            return response.strip()
        except Exception as e:
            logger.error(f"Erro gerando resposta sobre livro: {e}")
            return self._get_fallback_response(book, user_message, language)
    
    def _create_detailed_book_context(self, book: Dict) -> str:
        """Cria contexto detalhado sobre o livro"""
        context = f"""
        TÍTULO: {book.get('title', 'Desconhecido')}
        ID: {book.get('book_id', 'N/A')}
        
        AUTOR(ES): {', '.join(book.get('authors', ['Desconhecido']))}
        
        GÊNEROS: {', '.join(book.get('genres', ['Desconhecido']))}
        
        AVALIAÇÃO: ⭐ {book.get('rating', 0):.1f}/5
        NÚMERO DE AVALIAÇÕES: {book.get('num_ratings', 0):,}
        
        DESCRIÇÃO:
        {book.get('description', 'Descrição não disponível')}
        """
        return context
    
    def _get_no_book_reference_response(self, language: str) -> str:
        """Resposta quando não há referência a livro"""
        if language == 'pt':
            return "Não consegui identificar sobre qual livro você está perguntando. " \
                   "Pode especificar o título ou ID do livro? Por exemplo: " \
                   "\"Sobre o livro '1984'\" ou \"Fale mais sobre o livro com ID 123\"."
        else:
            return "I couldn't identify which book you're asking about. " \
                   "Can you specify the book title or ID? For example: " \
                   "\"About the book '1984'\" or \"Tell me more about book ID 123\"."
    
    def _get_book_not_found_response(self, book_reference: str, language: str) -> str:
        """Resposta quando livro não é encontrado"""
        if language == 'pt':
            return f"Desculpe, não encontrei informações sobre '{book_reference}' em minhas recomendações recentes. " \
                   "Pode especificar melhor o título ou tentar pedir novas recomendações primeiro?"
        else:
            return f"Sorry, I couldn't find information about '{book_reference}' in my recent recommendations. " \
                   "Can you specify the title better or try asking for new recommendations first?"
    
    def _get_fallback_response(self, book: Dict, user_message: str, language: str) -> str:
        """Resposta de fallback quando Ollama falha"""
        title = book.get('title', 'Este livro')
        authors = ', '.join(book.get('authors', ['Autor desconhecido']))
        rating = book.get('rating', 0)
        
        if language == 'pt':
            return f"Com base nas informações que tenho sobre **{title}**:\n\n" \
                  f"📖 **Autor:** {authors}\n" \
                  f"⭐ **Avaliação:** {rating:.1f}/5\n\n" \
                  f"Sua pergunta foi: \"{user_message}\"\n\n" \
                  f"Infelizmente não consigo analisar profundamente sua pergunta no momento. " \
                  f"Recomendo buscar análises críticas deste livro ou ler resenhas detalhadas. " \
                  f"Posso ajudá-lo com outras informações sobre este livro ou recomendar livros similares."
        else:
            return f"Based on the information I have about **{title}**:\n\n" \
                  f"📖 **Author:** {authors}\n" \
                  f"⭐ **Rating:** {rating:.1f}/5\n\n" \
                  f"Your question was: \"{user_message}\"\n\n" \
                  f"Unfortunately, I can't deeply analyze your question at the moment. " \
                  f"I recommend looking for critical analyses of this book or reading detailed reviews. " \
                  f"I can help you with other information about this book or recommend similar books."

    def find_book_by_title_fuzzy(self, title_query: str, session_id: str = None) -> Optional[Dict]:
        """Busca livro por título com correspondência difusa - VERSÃO CORRIGIDA"""
        logger.info(f"🔍 Buscando livro com título difuso: '{title_query}'")
        
        # 1. Primeiro, verificar nas recomendações recentes
        if session_id:
            recommendations = self.context_manager.get_last_recommendations(session_id)
            if recommendations:
                # Busca exata nas recomendações
                for book in recommendations:
                    book_title = book.get('title', '').lower()
                    query_lower = title_query.lower()
                    
                    # Verificar correspondências
                    if (query_lower == book_title or 
                        query_lower in book_title or
                        book_title in query_lower or
                        self._calculate_similarity(query_lower, book_title) > 0.7):
                        logger.info(f"✅ Livro encontrado nas recomendações: {book['title']}")
                        return book
        
        # 2. Buscar no dataset completo
        book_data = self.data_loader.data
        
        # Preparar o query
        query_lower = title_query.lower()
        query_words = query_lower.split()
        
        best_match = None
        best_score = 0
        
        for idx, book in book_data.iterrows():
            book_title = str(book.get('title', '')).lower()
            
            if not book_title:
                continue
            
            # Calcular score de similaridade
            score = self._calculate_title_similarity(query_lower, book_title, query_words)
            
            if score > best_score:
                best_score = score
                best_match = book
            
            # Se score muito alto, parar
            if score > 0.9:
                break
        
        if best_match is not None and best_score > 0.3:  # Threshold mínimo
            logger.info(f"✅ Melhor correspondência encontrada: {best_match['title']} (score: {best_score:.2f})")
            return {
                'book_id': int(best_match.get('bookId', 0)),
                'title': str(best_match['title']),
                'authors': self._extract_authors(best_match),
                'description': str(best_match.get('description', ''))[:300],
                'genres': self._extract_genres(best_match),
                'rating': float(best_match.get('rating', 0)),
                'num_ratings': int(best_match.get('numRatings', 0)) if 'numRatings' in best_match else 0
            }
        
        return None
    
    def _calculate_title_similarity(self, query: str, title: str, query_words: List[str]) -> float:
        """Calcula similaridade entre query e título"""
        # Se for match exato (ignorando case)
        if query == title.lower():
            return 1.0
        
        # Se o título contém a query ou vice-versa
        if query in title or title in query:
            return 0.9
        
        # Verificar palavras-chave em comum
        title_words_list = title.split()
        title_words_set = set(title_words_list)
        query_words_set = set(query_words)
        common_words = title_words_set.intersection(query_words_set)
        
        if common_words:
            # Quanto mais palavras em comum, maior o score
            word_score = len(common_words) / max(len(query_words), len(title_words_list))
            
            # Bonus se as palavras estão na mesma ordem
            order_bonus = 0
            for i in range(min(len(query_words), len(title_words_list))):
                if query_words[i] == title_words_list[i]:
                    order_bonus += 0.1
            
            return min(0.8, word_score + order_bonus)
        
        # Similaridade de Levenshtein para títulos curtos
        if len(query) < 20 and len(title) < 20:
            try:
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, query, title).ratio()
                return similarity * 0.7
            except:
                return 0.0
        
        return 0.0
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calcula similaridade entre duas strings"""
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
        except:
            return 0.0

    async def compare_multiple_books(self, user_message: str, session_id: str, 
                                   language: str = 'pt') -> Dict:
        """Compara múltiplos livros na mesma mensagem"""
        logger.info(f"🔍 Comparando múltiplos livros: '{user_message}'")
        
        # Detectar todos os livros na mensagem
        detected_books = self.detect_multiple_books(user_message, language)
        
        if len(detected_books) < 2:
            # Se não encontrou múltiplos, tratar como conversa normal
            return await self.chat_about_book(user_message, session_id, language)
        
        # Buscar informações de cada livro
        books_info = []
        books_found = []
        
        for title, book_id in detected_books:
            book = self.get_book_from_context(session_id, title, book_id)
            if book:
                books_info.append(book)
                books_found.append(book)
            else:
                # Se não encontrou, adicionar placeholder
                placeholder = {
                    'book_id': book_id or 0,
                    'title': title or f"Livro ID {book_id}",
                    'authors': ['Desconhecido'],
                    'description': 'Livro não encontrado no banco de dados',
                    'genres': ['Desconhecido'],
                    'rating': 0,
                    'num_ratings': 0
                }
                books_info.append(placeholder)
        
        if len(books_found) == 0:
            return {
                'response': self._get_no_books_found_response(detected_books, language),
                'books': [],
                'context': 'no_books_found'
            }
        
        # Gerar análise comparativa
        response = await self._generate_comparison_response(
            user_message, books_info, language
        )
        
        # Atualizar contexto
        self.context_manager.add_message(session_id, 'user', user_message)
        self.context_manager.add_message(session_id, 'assistant', response, books=books_found)
        
        return {
            'response': response,
            'books': books_found,
            'context': 'book_comparison',
            'books_count': len(books_found),
            'total_detected': len(detected_books)
        }
    
    async def _generate_comparison_response(self, user_message: str, books: List[Dict], 
                                          language: str) -> str:
        """Gera resposta de comparação entre múltiplos livros"""
        
        # Criar contexto detalhado dos livros
        books_context = "📚 LIVROS PARA COMPARAÇÃO:\n\n"
        
        for i, book in enumerate(books, 1):
            books_context += f"LIVRO {i}: '{book['title']}'\n"
            
            if book.get('authors'):
                authors = ', '.join(book['authors'])
                books_context += f"  Autor(es): {authors}\n"
            
            if book.get('genres'):
                genres = ', '.join(book['genres'])
                books_context += f"  Gêneros: {genres}\n"
            
            if book.get('rating', 0) > 0:
                books_context += f"  Avaliação: ⭐ {book['rating']:.1f}/5"
                if book.get('num_ratings', 0) > 0:
                    books_context += f" ({book['num_ratings']} avaliações)\n"
                else:
                    books_context += "\n"
            
            if book.get('description'):
                desc = book['description']
                if len(desc) > 150:
                    desc = desc[:150] + "..."
                books_context += f"  Descrição: {desc}\n"
            
            books_context += "\n" + "-"*50 + "\n"
        
        if language == 'pt':
            prompt = f"""
            VOCÊ É: Um crítico literário especialista em análise comparativa de livros.
            
            PERGUNTA DO USUÁRIO:
            "{user_message}"
            
            {books_context}
            
            SUA TAREFA:
            1. ANALISAR cada livro individualmente
            2. COMPARAR os livros entre si baseado na pergunta do usuário
            3. IDENTIFICAR qual livro melhor atende ao critério solicitado
            4. EXPLICAR suas conclusões com exemplos específicos
            5. DAR uma recomendação final
            
            DIRETRIZES:
            - Seja imparcial e baseie-se nos dados fornecidos
            - Use exemplos concretos das descrições/gêneros
            - Compare aspectos relevantes para a pergunta
            - Se um livro não foi encontrado, mencione isso
            - Formato: Análise individual + comparação + conclusão
            - Seja detalhado mas objetivo
            
            RESPOSTA (em português, 6-10 parágrafos):
            """
        else:
            prompt = f"""
            YOU ARE: A literary critic expert in comparative book analysis.
            
            USER QUESTION:
            "{user_message}"
            
            {books_context}
            
            YOUR TASK:
            1. ANALYZE each book individually
            2. COMPARE the books based on the user's question
            3. IDENTIFY which book best meets the requested criteria
            4. EXPLAIN your conclusions with specific examples
            5. PROVIDE a final recommendation
            
            GUIDELINES:
            - Be impartial and base on provided data
            - Use concrete examples from descriptions/genres
            - Compare aspects relevant to the question
            - If a book wasn't found, mention that
            - Format: Individual analysis + comparison + conclusion
            - Be detailed but objective
            
            RESPONSE (in English, 6-10 paragraphs):
            """
        
        try:
            response = await self.ollama_service.chat([
                {"role": "user", "content": prompt}
            ])
            return response.strip()
        except Exception as e:
            logger.error(f"Erro gerando comparação: {e}")
            return self._generate_fallback_comparison(books, user_message, language)
    
    def _generate_fallback_comparison(self, books: List[Dict], user_message: str, 
                                    language: str) -> str:
        """Fallback para comparação quando Ollama falha"""
        if language == 'pt':
            response = f"📊 **Análise Comparativa**\n\n"
            response += f"**Pergunta:** {user_message}\n\n"
            
            for i, book in enumerate(books, 1):
                response += f"**📖 Livro {i}: {book['title']}**\n"
                
                if book.get('authors'):
                    response += f"- Autor(es): {', '.join(book['authors'])}\n"
                
                if book.get('genres'):
                    response += f"- Gêneros: {', '.join(book['genres'])}\n"
                
                if book.get('rating', 0) > 0:
                    response += f"- Avaliação: ⭐ {book['rating']:.1f}/5\n"
                
                response += "\n"
            
            response += "🔍 **Observação:** Para uma análise comparativa detalhada, "
            response += "recomendo consultar resenhas críticas de cada livro e comparar os temas abordados."
            
        else:
            response = f"📊 **Comparative Analysis**\n\n"
            response += f"**Question:** {user_message}\n\n"
            
            for i, book in enumerate(books, 1):
                response += f"**📖 Book {i}: {book['title']}**\n"
                
                if book.get('authors'):
                    response += f"- Author(s): {', '.join(book['authors'])}\n"
                
                if book.get('genres'):
                    response += f"- Genres: {', '.join(book['genres'])}\n"
                
                if book.get('rating', 0) > 0:
                    response += f"- Rating: ⭐ {book['rating']:.1f}/5\n"
                
                response += "\n"
            
            response += "🔍 **Note:** For a detailed comparative analysis, "
            response += "I recommend consulting critical reviews of each book and comparing the themes covered."
        
        return response
    
    def _get_no_books_found_response(self, detected_books: List[Tuple], language: str) -> str:
        """Resposta quando nenhum livro é encontrado"""
        book_names = []
        for title, book_id in detected_books:
            if title:
                book_names.append(f"'{title}'")
            elif book_id:
                book_names.append(f"ID {book_id}")
        
        books_str = ", ".join(book_names)
        
        if language == 'pt':
            return f"Não consegui encontrar informações sobre {books_str} em meu banco de dados. "
        else:
            return f"I couldn't find information about {books_str} in my database. "   