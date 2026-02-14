# D:\Django\book_agent\services\response_generator.py

import logging
import random
from typing import List, Dict
from .ollama_service import OllamaService
from .search_engine import BookResult

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, ollama_service: OllamaService):
        self.ollama_service = ollama_service
        self.response_templates = {
            'pt': {
                'emotional_support': [
                    "Sinto muito que você esteja passando por um momento difícil. 💛 Livros podem ser grandes aliados em momentos como este. Posso recomendar alguns que podem trazer conforto, inspiração ou uma nova perspectiva para você.",
                    "Lamento saber que você não está se sentindo bem. 📚 A leitura pode ser uma ótima companhia nos dias difíceis. Posso sugerir alguns livros que podem ajudar a clarear seus pensamentos ou trazer um pouco de leveza?",
                    "É compreensível se sentir assim às vezes. ❤️‍🩹 Muitas pessoas encontram nos livros um refúgio e uma forma de processar emoções. Posso recomendar algumas leituras que podem ser úteis para você neste momento?",
                    "Obrigado por compartilhar isso comigo. 🤗 Livros têm o poder de nos acolher nos momentos mais difíceis. Posso sugerir algumas obras que oferecem conforto, sabedoria ou simplesmente uma boa distração quando precisamos?"
                ],
                'greeting': [
                    "Olá! 😊 Sou seu assistente especializado em recomendações de livros. Como você está se sentindo hoje? Conte-me sobre seus interesses ou o que está passando, e vou sugerir livros que realmente combinam com você!",
                    "Oi! Sou seu consultor literário pessoal. 🤗 Me diga como está seu dia ou o que você gosta de ler, e encontrarei os livros perfeitos para você! 📚",
                    "Olá, leitor! Estou aqui para ajudar você a encontrar livros que vão além do óbvio. 😊 Como você está? Conte-me sobre seus objetivos de leitura ou como está se sentindo!"
                ],
                'no_results': [
                    "Não encontrei livros específicos para essa busca, mas posso recomendar outros títulos relacionados. Que tal me contar mais sobre o que você precisa ou como está se sentindo?",
                    "Vamos ajustar a busca! Me fale mais sobre o que você precisa ou está passando. Às vezes, um bom livro aparece quando menos esperamos!",
                    "Hmm, preciso de mais detalhes para encontrar o livro perfeito para você. 🧐 Você está procurando algo para se inspirar, se distrair, aprender algo novo ou apenas uma boa companhia?"
                ],
                'closing': [
                    "Foi um prazer ajudar você em sua jornada literária! 📚 Volte sempre que quiser novas recomendações da Smart Library!",
                    "Espero que encontre nos livros o que precisa neste momento. Até a próxima! Lembre-se: estou sempre aqui para ajudar.",
                    "Lembre-se: cada livro é uma nova aventura. Boas leituras! 😊 E não se esqueça, sou seu assistente da Smart Library!",
                    "Até logo! Se precisar de mais recomendações, é só chamar. Fui criado para ajudar leitores como você a encontrar histórias incríveis!"
                ],
            },
            'en': {
                'emotional_support': [
                    "I'm sorry to hear you're going through a difficult time. 💛 Books can be great allies in moments like these. I can recommend some that might bring comfort, inspiration, or a new perspective to you.",
                    "I'm sorry you're not feeling well. 📚 Reading can be great company on difficult days. Can I suggest some books that might help clear your thoughts or bring a little lightness?",
                    "It's understandable to feel this way sometimes. ❤️‍🩹 Many people find in books a refuge and a way to process emotions. Can I recommend some readings that might be helpful for you right now?",
                    "Thank you for sharing this with me. 🤗 Books have the power to embrace us in the most difficult moments. Can I suggest some works that offer comfort, wisdom, or simply a good distraction when we need it?"
                ],
                'greeting': [
                    "Hello! 😊 I'm your book recommendation assistant. How are you feeling today? Tell me about your interests or what you're going through, and I'll suggest books that truly match your needs!",
                    "Hi! I'm your personal literary consultant. 🤗 Tell me how your day is going or what you like to read, and I'll find the perfect books for you! 📚",
                    "Hello, reader! I'm here to help you find books that go beyond the obvious. 😊 How are you? Tell me about your reading goals or how you're feeling!"
                ],
                'no_results': [
                    "I didn't find specific books for that search, but I can recommend other related titles. How about telling me more about what you need or how you're feeling?",
                    "Let's adjust the search! Tell me more about what you need or what you're going through. Sometimes a good book appears when we least expect it!",
                    "Hmm, I need more details to find the perfect book for you. 🧐 Are you looking for something to inspire you, distract you, learn something new, or just good company?"
                ],
                'closing': [
                    "It was a pleasure helping you on your literary journey! 📚 Come back anytime you want new recommendations from Smart Library!",
                    "I hope you find in books what you need at this moment. See you next time! Remember: I'm always here to help.",
                    "Remember: every book is a new adventure. Happy reading! 😊 And don't forget, I'm your Smart Library assistant!",
                    "Goodbye! If you need more recommendations, just call. I was created to help readers like you find amazing stories!"
                ]
            }
        }

    async def generate(self, user_message: str, books: List, conversation_context: List = None, language: str = "pt"):
        """
        Método simplificado para gerar resposta
        """
        # Aqui você pode usar seus métodos existentes
        return await self.generate_personalized_recommendation(
            user_message=user_message,
            books=books,
            language=language
        )    

    async def generate_personalized_recommendation(self, user_message: str,
                                                   books: List[BookResult],
                                                   intent: str = 'book_recommendation',
                                                   language: str = 'pt',
                                                   conversation_history: List = None) -> str:
        """Gera recomendações personalizadas COM HISTÓRICO"""
        
        # Inicializar histórico se None
        if conversation_history is None:
            conversation_history = []
        
        # Respostas para intents específicas
        if intent == 'emotional_support':
            return await self._generate_emotional_support_response(user_message, books, language)
        
        if intent == 'social':
            return random.choice(self.response_templates[language]['greeting'])
        
        if intent == 'closing':
            return random.choice(self.response_templates[language]['closing'])
        
        # Se não há livros, resposta padrão
        if not books:
            return random.choice(self.response_templates[language]['no_results'])
        
        # Limitar livros para evitar contexto muito longo
        if len(books) > 4:
            books = books[:4]
        
        # Extrair contexto do usuário
        user_context = self._extract_user_context(user_message, language)
        
        # Criar contexto detalhado dos livros
        books_context = self._create_detailed_book_context(books, "", language)
        
        # CONSTRUIR HISTÓRICO DE CONVERSA para o Ollama
        messages = []
        
        # 1. Adicionar instruções do sistema com contexto atual
        system_message = self._create_system_message(
            user_message, books_context, user_context, language
        )
        messages.append({"role": "system", "content": system_message})
        
        # 2. Adicionar histórico se disponível (máximo 6 mensagens)
        if conversation_history:
            conversation_history = conversation_history[-2:]  # APENAS últimas 2 mensagens
            logger.info(f"📚 Usando {len(conversation_history)} mensagens de histórico")
            
            # Filtrar apenas as últimas N mensagens para caber no contexto
            for msg in conversation_history[-2:]:  # Últimas 6 mensagens
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")[:400]  # Limitar tamanho
                })
        
        # 3. Adicionar mensagem atual do usuário
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = await self.ollama_service.chat(messages)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Erro gerando recomendação personalizada: {e}")
            return self._generate_fallback_recommendation(user_message, books, language)
    
    def _create_system_message(self, user_message: str, books_context: str, 
                              user_context: str, language: str) -> str:
        """Cria mensagem do sistema com contexto"""
        if language == 'pt':
            return f"""
            VOCÊ É: Um assistente de recomendações de livros empático e compreensivo. Você se importa genuinamente com o bem-estar das pessoas.
            
            CONTEXTO DO USUÁRIO:
            {user_context}
            
            LIVROS DISPONÍVEIS (APENAS ESTES PODEM SER RECOMENDADOS):
            {books_context}
            
            REGRAS IMPORTANTES:
            1. Recomende APENAS livros da lista acima
            2. Use o histórico da conversa para manter continuidade
            3. Seja específico sobre POR QUE cada livro é relevante
            4. Relacione com a conversa anterior quando aplicável
            5. Se o usuário perguntar sobre livros já mencionados, foque neles
            6. Não invente livros que não estão na lista
            7. Não sugira livros que não foram fornecidos
            
            SUA RESPOSTA DEVE:
            - Ser natural e conversacional
            - Manter o contexto da conversa
            - Referenciar livros anteriores se relevante
            - Explicar por que cada recomendação é boa para o usuário
            - Mostrar empatia pela situação do usuário
            - Ser acolhedor e compreensivo
            
            EXEMPLOS DO QUE NÃO FAZER:
            - "Recomendo o livro X" (se X não está na lista) ❌
            - "Existe um livro chamado Y" (se Y não está na lista) ❌
            
            EXEMPLOS DO QUE FAZER:
            - "Baseado nos livros disponíveis, recomendo: [título da lista]..." ✅
            - "Dos livros que temos, o mais relevante é: [título da lista]..." ✅
            
            REGRA IMPORTANTE:
            Não assuma interesses técnicos (como programação, ciência de dados,
            machine learning ou IA) a menos que o usuário mencione explicitamente
            esses temas na mensagem.
            """
        else:
            return f"""
            YOU ARE: An empathetic and understanding book recommendation assistant. You genuinely care about people's well-being.
            
            USER CONTEXT:
            {user_context}
            
            AVAILABLE BOOKS (ONLY THESE CAN BE RECOMMENDED):
            {books_context}
            
            IMPORTANT RULES:
            1. Recommend ONLY books from the list above
            2. Use conversation history to maintain continuity
            3. Be specific about WHY each book is relevant
            4. Relate to previous conversation when applicable
            5. If user asks about previously mentioned books, focus on them
            6. Do not invent books that are not in the list
            7. Do not suggest books that were not provided
            
            YOUR RESPONSE SHOULD:
            - Be natural and conversational
            - Maintain conversation context
            - Reference previous books if relevant
            - Explain why each recommendation is good for the user
            - Show empathy for the user's situation
            - Be welcoming and understanding
            
            EXAMPLES OF WHAT NOT TO DO:
            - "I recommend book X" (if X is not in the list) ❌
            - "There's a book called Y" (if Y is not in the list) ❌
            
            EXAMPLES OF WHAT TO DO:
            - "Based on the available books, I recommend: [list title]..." ✅
            - "From the books we have, the most relevant is: [list title]..." ✅
            
            IMPORTANT RULE:
            Do not assume technical interests (such as programming, data science,
            machine learning, or AI) unless the user explicitly mentions
            these topics in their message.
            """
    
    # No response_generator.py, linha 235:

    def _create_detailed_book_context(self, books: List, user_query: str, language: str) -> str:
        """Cria contexto detalhado dos livros - VERSÃO COMPATÍVEL COM DICT E BOOKRESULT"""
        if not books:
            return ""
        
        context_lines = []
        
        for i, book in enumerate(books[:10], 1):
            try:
                # Verificar se é dict ou BookResult
                if isinstance(book, dict):
                    title = book.get('title', '')
                    authors = book.get('authors', [])
                    description = book.get('description', '')
                    genres = book.get('genres', [])
                    rating = book.get('rating', 0)
                    similarity_score = book.get('similarity_score', 0)
                    search_method = book.get('search_method', '')
                else:
                    # É BookResult
                    title = book.title
                    authors = book.authors
                    description = book.description
                    genres = book.genres
                    rating = getattr(book, 'rating', 0)
                    similarity_score = getattr(book, 'similarity_score', 0)
                    search_method = getattr(book, 'search_method', '')
                
                # Formatar autores
                if isinstance(authors, list):
                    authors_str = ', '.join(authors[:2])
                else:
                    authors_str = str(authors)
                
                # Formatar gêneros
                if isinstance(genres, list):
                    genres_str = ', '.join(genres[:3])
                else:
                    genres_str = str(genres)
                
                # Criar linha do livro
                book_info = f"LIVRO {i}: '{title}'\n"
                book_info += f"  Autores: {authors_str}\n"
                book_info += f"  Gêneros: {genres_str}\n"
                
                if description:
                    # Limitar descrição
                    desc_limit = 150 if language == 'pt' else 120
                    short_desc = description[:desc_limit] + '...' if len(description) > desc_limit else description
                    book_info += f"  Descrição: {short_desc}\n"
                
                if rating > 0:
                    book_info += f"  Avaliação: ⭐ {rating:.1f}/5.0\n"
                
                if similarity_score > 0:
                    book_info += f"  Relevância: {similarity_score:.2f}\n"
                
                if search_method:
                    book_info += f"  Método de busca: {search_method}\n"
                
                book_info += "-" * 40
                context_lines.append(book_info)
                
            except Exception as e:
                logger.error(f"Erro ao processar livro {i}: {e}")
                continue
        
        return "\n\n".join(context_lines)
    
    def _extract_user_context(self, message: str, language: str) -> str:
        """Extrai contexto do usuário APENAS da mensagem atual"""
        message_lower = message.lower()
        
        context_parts = []
        
        # Detectar estado emocional
        emotional_keywords = {
            'pt': {
                'triste': ['triste', 'tristeza', 'deprimido', 'chorando', 'chateado'],
                'ansioso': ['ansioso', 'ansiedade', 'nervoso', 'preocupado'],
                'estressado': ['estressado', 'estresse', 'cansado', 'exausto'],
                'sozinho': ['sozinho', 'solidão', 'isolado'],
                'feliz': ['feliz', 'alegre', 'contente', 'animado']
            },
            'en': {
                'sad': ['sad', 'depressed', 'crying', 'upset'],
                'anxious': ['anxious', 'anxiety', 'nervous', 'worried'],
                'stressed': ['stressed', 'stress', 'tired', 'exhausted'],
                'lonely': ['lonely', 'loneliness', 'isolated'],
                'happy': ['happy', 'joyful', 'content', 'excited']
            }
        }
        
        lang_dict = emotional_keywords.get(language, emotional_keywords['en'])
        for emotion, keywords in lang_dict.items():
            if any(keyword in message_lower for keyword in keywords):
                context_parts.append(f"Estado emocional: {emotion}")
                break
        
        # Detectar área de interesse APENAS na mensagem atual
        study_areas = {
            'pt': {
                'comics': ['quadrinhos', 'hq', 'homem-aranha', 'spider-man', 'marvel', 'dc', 'super-herói', 'superhero', 'comics'],
                'computer science': ['ciência da computação', 'computação', 'programação', 'desenvolvimento', 'software', 'engenharia de software', 'algoritmo'],
                'data science': ['ciência de dados', 'data science', 'machine learning', 'aprendizado de máquina', 'inteligência artificial', 'ia'],
                'business': ['administração', 'negócios', 'empreendedorismo', 'marketing', 'gestão', 'management'],
                'design': ['design', 'ux', 'interface', 'user experience', 'ui'],
                'engineering': ['engenharia', 'civil', 'elétrica', 'mecânica', 'produção'],
                'culinary': ['culinária', 'culinaria', 'gastronomia', 'cozinha', 'receitas', 'comida', 'alimentação']
            },
            'en': {
                'comics': ['comics', 'graphic novel', 'spider-man', 'marvel', 'dc', 'superhero', 'super hero', 'hq'],
                'computer science': ['computer science', 'programming', 'coding', 'software development', 'software engineering', 'algorithm'],
                'data science': ['data science', 'machine learning', 'artificial intelligence', 'ai', 'data analysis'],
                'business': ['business', 'entrepreneurship', 'marketing', 'management', 'administration'],
                'design': ['design', 'ux', 'user experience', 'ui', 'interface'],
                'engineering': ['engineering', 'civil', 'electrical', 'mechanical', 'industrial'],
                'culinary': ['culinary', 'cuisine', 'cooking', 'gastronomy', 'recipes', 'food', 'cookbook']
            }
        }
        
        lang_dict = study_areas.get(language, study_areas['en'])
        
        for area, keywords in lang_dict.items():
            if any(keyword in message_lower for keyword in keywords):
                context_parts.append(f"Área de interesse: {area}")
                break
        
        # Se não encontrou nenhuma área específica
        if not context_parts:
            context_parts.append("Área de interesse: ")
        
        # Detectar nível (iniciante, intermediário, avançado) APENAS na mensagem atual
        levels = {
            'pt': {
                'beginner': ['iniciante', 'começando', 'básico', 'primeiro', 'novato'],
                'intermediate': ['intermediário', 'intermedia', 'já sei', 'experiente'],
                'advanced': ['avançado', 'expert', 'especialista', 'profissional']
            },
            'en': {
                'beginner': ['beginner', 'starting', 'basic', 'first', 'newbie'],
                'intermediate': ['intermediate', 'already know', 'experienced'],
                'advanced': ['advanced', 'expert', 'professional', 'specialist']
            }
        }
        
        level_dict = levels.get(language, levels['en'])
        for level, keywords in level_dict.items():
            if any(keyword in message_lower for keyword in keywords):
                context_parts.append(f"Nível: {level}")
                break
        
        # Detectar objetivos APENAS na mensagem atual
        objectives_keywords = {
            'pt': ['aprender', 'estudar', 'melhorar', 'desenvolver', 'crescer', 'entender', 'conhecer'],
            'en': ['learn', 'study', 'improve', 'develop', 'grow', 'understand', 'know']
        }
        
        if any(keyword in message_lower for keyword in objectives_keywords.get(language, objectives_keywords['en'])):
            context_parts.append("Objetivo: Aprendizado/Desenvolvimento")
        
        # Juntar contexto
        if context_parts:
            return " | ".join(context_parts)
        else:
            return "Perfil: Interesses gerais de leitura"
    
    async def _generate_emotional_support_response(self, user_message: str, books: List[BookResult], language: str) -> str:
        """Gera resposta para mensagens emocionais/negativas"""
        
        # Começar com mensagem empática
        empathic_opening = random.choice(self.response_templates[language]['emotional_support'])
        
        # Se não há livros, oferecer suporte conversacional
        if not books:
            if language == 'pt':
                return f"{empathic_opening}\n\nÀs vezes, só de conversar já ajuda. Eu estou aqui para ouvir você. Quer me contar mais sobre o que está passando? 😊\n\nSe preferir, posso tentar buscar livros sobre bem-estar emocional ou autoajuda para você."
            else:
                return f"{empathic_opening}\n\nSometimes just talking helps. I'm here to listen to you. Would you like to tell me more about what you're going through? 😊\n\nIf you prefer, I can try to find books about emotional well-being or self-help for you."
        
        # Buscar livros específicos para apoio emocional
        emotional_support_books = self._filter_emotional_support_books(books)
        
        if emotional_support_books:
            books_context = self._create_detailed_book_context(emotional_support_books[:4], "", language)
        else:
            books_context = self._create_detailed_book_context(books[:4], "", language)
        
        if language == 'pt':
            prompt = f"""
            VOCÊ É: Um assistente empático que usa livros como ferramenta de apoio emocional.

            MENSAGEM DO USUÁRIO (mostrando sofrimento/necessidade):
            "{user_message}"

            SUA ABERTURA EMPÁTICA (já usada):
            "{empathic_opening}"

            LIVROS DISPONÍVEIS (alguns podem ser terapêuticos):
            {books_context}

            SUA TAREFA:
            1. Manter o tom EMPÁTICO e ACONCHEGANTE
            2. Recomendar 2-3 livros que possam ajudar no momento
            3. Explicar GENTILMENTE como cada livro pode ser útil
            4. Oferecer espaço para o usuário falar mais se quiser
            5. Não ser invasivo ou dar conselhos médicos
            6. Usar linguagem calorosa e humanizada

            EXEMPLOS DO QUE DIZER:
            - "Em momentos difíceis, ler sobre [tema] pode trazer algum conforto..."
            - "Este livro me fez pensar que poderia ajudar você porque..."
            - "Não sei exatamente o que você está passando, mas talvez esta leitura..."

            EVITAR:
            - "Você deveria..."
            - "O que você precisa fazer é..."
            - Soluções simplistas
            - Julgamentos

            REGRA IMPORTANTE:
            Não assuma interesses técnicos (como programação, ciência de dados,
            machine learning ou IA) a menos que o usuário mencione explicitamente
            esses temas na mensagem.

            RESPOSTA (em português, 4-6 parágrafos, muito acolhedor):
            """
        else:
            prompt = f"""
            YOU ARE: An empathetic assistant who uses books as emotional support tools.

            USER MESSAGE (showing distress/need):
            "{user_message}"

            YOUR EMPATHIC OPENING (already used):
            "{empathic_opening}"

            AVAILABLE BOOKS (some may be therapeutic):
            {books_context}

            YOUR TASK:
            1. Maintain an EMPATHETIC and WELCOMING tone
            2. Recommend 2-3 books that might help in the moment
            3. Explain GENTLY how each book could be useful
            4. Offer space for the user to talk more if they want
            5. Don't be invasive or give medical advice
            6. Use warm, humanized language

            EXAMPLES OF WHAT TO SAY:
            - "In difficult times, reading about [topic] can bring some comfort..."
            - "This book made me think it could help you because..."
            - "I don't know exactly what you're going through, but perhaps this reading..."

            AVOID:
            - "You should..."
            - "What you need to do is..."
            - Simplistic solutions
            - Judgments

            IMPORTANT RULE:
            Do not assume technical interests (such as programming, data science,
            machine learning, or AI) unless the user explicitly mentions
            these topics in their message.

            RESPONSE (in English, 4-6 paragraphs, very welcoming):
            """
        
        try:
            response = await self.ollama_service.chat([
                {"role": "user", "content": prompt}
            ])
            return f"{empathic_opening}\n\n{response.strip()}"
            
        except Exception as e:
            logger.error(f"Erro gerando resposta emocional: {e}")
            return f"{empathic_opening}\n\n" + self._generate_fallback_emotional_response(books, language)
    
    def _filter_emotional_support_books(self, books: List[BookResult]) -> List[BookResult]:
        """Filtra livros que podem ser úteis para apoio emocional"""
        emotional_keywords = [
            # Gêneros terapêuticos
            'self-help', 'self help', 'autoajuda', 'auto-ajuda',
            'psychology', 'psicologia', 'therapy', 'terapia',
            'mindfulness', 'meditation', 'meditação',
            'happiness', 'felicidade', 'well-being', 'bem-estar',
            'inspiration', 'inspiração', 'motivation', 'motivação',
            'philosophy', 'filosofia', 'spiritual', 'espiritual',
            'poetry', 'poesia', 'memoir', 'autobiografia',
            'comfort', 'conforto', 'healing', 'cura',
            
            # Títulos/keywords positivos
            'joy', 'alegria', 'peace', 'paz', 'hope', 'esperança',
            'light', 'luz', 'calm', 'calma', 'serenity', 'serenidade',
            'gratitude', 'gratidão', 'kindness', 'bondade', 'compassion', 'compaixão'
        ]
        
        filtered_books = []
        for book in books:
            # Verificar no título
            title_lower = book.title.lower()
            # Verificar em gêneros
            genres_lower = ' '.join([g.lower() for g in book.genres]) if book.genres else ''
            # Verificar na descrição
            desc_lower = book.description.lower() if book.description else ''
            
            full_text = f"{title_lower} {genres_lower} {desc_lower}"
            
            if any(keyword in full_text for keyword in emotional_keywords):
                filtered_books.append(book)
        
        return filtered_books if filtered_books else books[:3]  # Retorna os primeiros se não encontrar específicos
    
    def _generate_fallback_emotional_response(self, books: List[BookResult], language: str) -> str:
        """Fallback para respostas emocionais"""
        if language == 'pt':
            if books:
                response = "Encontrei alguns livros que podem trazer algum conforto ou distração:\n\n"
                for i, book in enumerate(books[:3], 1):
                    response += f"{i}. **{book.title}**"
                    if book.authors:
                        response += f" por {', '.join(book.authors[:2])}"
                    response += "\n"
                
                response += "\nÀs vezes, mergulhar em uma boa história pode ajudar a ver as coisas de outra perspectiva. 😊\n\n"
                response += "Quer que eu busque livros sobre algum tema específico que possa ajudar?"
            else:
                response = "Às vezes, apenas ter alguém para conversar já faz diferença. Estou aqui se quiser desabafar ou se precisar de alguma recomendação específica. 🤗"
        else:
            if books:
                response = "I found some books that might bring some comfort or distraction:\n\n"
                for i, book in enumerate(books[:3], 1):
                    response += f"{i}. **{book.title}**"
                    if book.authors:
                        response += f" by {', '.join(book.authors[:2])}"
                    response += "\n"
                
                response += "\nSometimes diving into a good story can help see things from another perspective. 😊\n\n"
                response += "Would you like me to look for books on any specific topic that might help?"
            else:
                response = "Sometimes just having someone to talk to makes a difference. I'm here if you want to vent or if you need any specific recommendations. 🤗"
        
        return response
    
    def _generate_fallback_recommendation(self, user_message: str, books: List[BookResult], language: str) -> str:
        """Fallback quando o Ollama não está disponível"""
        if not books:
            if language == 'pt':
                return "Não encontrei livros específicos para sua busca. Pode me contar mais sobre o que você precisa?"
            else:
                return "I didn't find specific books for your search. Can you tell me more about what you need?"
        
        # Selecionar os livros mais relevantes
        top_books = books[:4]
        
        if language == 'pt':
            response = f"Baseado na sua mensagem '{user_message[:50]}...', encontrei {len(books)} livros relevantes. Aqui estão minhas recomendações:\n\n"
            
            for i, book in enumerate(top_books, 1):
                response += f"📚 **{book.title}** (ID: {book.book_id})\n"
                
                if book.authors:
                    response += f"   👤 **Autores:** {', '.join(book.authors)}\n"
                
                if book.genres:
                    response += f"   🎭 **Gêneros:** {', '.join(book.genres[:3])}\n"
                
                if book.rating > 0:
                    response += f"   ⭐ **Avaliação:** {book.rating:.1f}/5"
                    if book.num_ratings > 0:
                        response += f" ({book.num_ratings} avaliações)\n"
                    else:
                        response += "\n"
                
                # Recomendação genérica
                response += f"   💡 **Por que recomendo:** Oferece uma leitura envolvente com conteúdo relevante e bem avaliado."
                response += "\n\n"
            
            response += "🔍 **Dica:** Para aprofundar em algum tema específico, me pergunte sobre 'livros de [assunto]'.\n"
            response += "📖 **Ordem sugerida:** Comece pelo livro que mais chamou sua atenção.\n\n"
            response += "Gostaria de saber mais detalhes sobre algum desses livros?"
            
        else:
            response = f"Based on your message '{user_message[:50]}...', I found {len(books)} relevant books. Here are my recommendations:\n\n"
            
            for i, book in enumerate(top_books, 1):
                response += f"📚 **{book.title}** (ID: {book.book_id})\n"
                
                if book.authors:
                    response += f"   👤 **Authors:** {', '.join(book.authors)}\n"
                
                if book.genres:
                    response += f"   🎭 **Genres:** {', '.join(book.genres[:3])}\n"
                
                if book.rating > 0:
                    response += f"   ⭐ **Rating:** {book.rating:.1f}/5"
                    if book.num_ratings > 0:
                        response += f" ({book.num_ratings} reviews)\n"
                    else:
                        response += "\n"
                
                # Generic recommendation
                response += f"   💡 **Why I recommend it:** Offers engaging reading with relevant and well-reviewed content."
                response += "\n\n"
            
            response += "🔍 **Tip:** To dive deeper into a specific topic, ask me about 'books on [subject]'.\n"
            response += "📖 **Suggested order:** Start with the book that caught your attention the most.\n\n"
            response += "Would you like to know more details about any of these books?"
        
        return response