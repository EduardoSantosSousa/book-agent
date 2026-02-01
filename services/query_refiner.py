# D:\Django\book_agent\services\query_refiner.py
import asyncio
import logging
from typing import Dict, List, Tuple
import json

logger = logging.getLogger(__name__)

class QueryRefinerAgent:
    def __init__(self, groq_service):
        self.groq_service = groq_service
        
    async def refine_search_query(self, user_message: str, language: str = 'pt') -> Dict:
        """
        Refina a query de busca usando IA para:
        1. Corrigir erros de digitação
        2. Expandir sinônimos
        3. Normalizar termos
        4. Extrair intenções
        """
        logger.info(f"🔍 Refinando query: '{user_message}'")
        
        prompt = self._create_refinement_prompt(user_message, language)
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = await self.groq_service.chat(messages)
            
            # Parse a resposta JSON
            refined_data = self._parse_refinement_response(response)
            
            logger.info(f"✅ Query refinada: {refined_data.get('normalized_query', user_message)}")
            logger.info(f"   Sinônimos: {refined_data.get('synonyms', [])[:3]}")
            logger.info(f"   Termos-chave: {refined_data.get('keywords', [])[:3]}")
            
            return refined_data
            
        except Exception as e:
            logger.error(f"❌ Erro ao refinar query: {e}")
            # Fallback básico
            return {
                "original_query": user_message,
                "normalized_query": user_message.lower(),
                "synonyms": [],
                "keywords": user_message.lower().split(),
                "search_intent": "general",
                "corrected_typos": False
            }
    
    def _create_refinement_prompt(self, query: str, language: str) -> str:
        """Cria prompt para refinamento de query"""
        
        if language == 'pt':
            return f"""
            VOCÊ É: Um especialista em refinamento de queries de busca para sistemas de recomendação de livros.
            
            SUA TAREFA: Analisar a query do usuário e retornar um JSON estruturado com:
            1. Query normalizada (corrigida)
            2. Sinônimos relevantes
            3. Termos-chave extraídos
            4. Intenção de busca
            5. Correções aplicadas
            
            REGRAS IMPORTANTES:
            - CORRIGIR erros de digitação comuns (ex: "superhome" → "superman")
            - EXPANDIR sinônimos para personagens de quadrinhos (ex: "homem-aranha" → ["spider-man", "peter parker"])
            - NORMALIZAR variações (ex: "super-homem", "super homem", "superman" → "superman")
            - EXTRAIR termos-chave para busca semântica
            - DETECTAR intenção: comics, author, genre, general, etc.
            
            EXEMPLOS:
            
            INPUT: "livros do superhome"
            OUTPUT: {{
                "original_query": "livros do superhome",
                "normalized_query": "livros do superman",
                "synonyms": ["super-homem", "man of steel", "clark kent", "dc comics"],
                "keywords": ["superman", "dc", "comics", "super-herói"],
                "search_intent": "comics",
                "corrected_typos": true,
                "confidence_score": 0.95
            }}
            
            INPUT: "homem aranha quadrinhos"
            OUTPUT: {{
                "original_query": "homem aranha quadrinhos",
                "normalized_query": "homem-aranha quadrinhos",
                "synonyms": ["spider-man", "peter parker", "marvel comics", "aranha"],
                "keywords": ["spider-man", "marvel", "comics", "super-herói", "quadrinhos"],
                "search_intent": "comics",
                "corrected_typos": true,
                "confidence_score": 0.92
            }}
            
            INPUT: "recomende livros de fantasia"
            OUTPUT: {{
                "original_query": "recomende livros de fantasia",
                "normalized_query": "livros de fantasia",
                "synonyms": ["fantasy", "magia", "aventura", "épico"],
                "keywords": ["fantasia", "fantasy", "aventura", "magia"],
                "search_intent": "genre",
                "corrected_typos": false,
                "confidence_score": 0.98
            }}
            
            RETORNE APENAS JSON. Nenhum texto adicional.
            """
        else:
            return f"""
            YOU ARE: A search query refinement expert for book recommendation systems.
            
            YOUR TASK: Analyze the user's query and return structured JSON with:
            1. Normalized query (corrected)
            2. Relevant synonyms
            3. Extracted keywords
            4. Search intent
            5. Applied corrections
            
            IMPORTANT RULES:
            - CORRECT common typos (ex: "superhome" → "superman")
            - EXPAND synonyms for comic characters (ex: "spider man" → ["spider-man", "peter parker"])
            - NORMALIZE variations (ex: "super man", "super-man", "superman" → "superman")
            - EXTRACT keywords for semantic search
            - DETECT intent: comics, author, genre, general, etc.
            
            EXAMPLES:
            
            INPUT: "books about superhome"
            OUTPUT: {{
                "original_query": "books about superhome",
                "normalized_query": "books about superman",
                "synonyms": ["man of steel", "clark kent", "dc comics"],
                "keywords": ["superman", "dc", "comics", "superhero"],
                "search_intent": "comics",
                "corrected_typos": true,
                "confidence_score": 0.95
            }}
            
            INPUT: "spider man comics"
            OUTPUT: {{
                "original_query": "spider man comics",
                "normalized_query": "spider-man comics",
                "synonyms": ["spider-man", "peter parker", "marvel comics"],
                "keywords": ["spider-man", "marvel", "comics", "superhero"],
                "search_intent": "comics",
                "corrected_typos": true,
                "confidence_score": 0.92
            }}
            
            INPUT: "recommend fantasy books"
            OUTPUT: {{
                "original_query": "recommend fantasy books",
                "normalized_query": "fantasy books",
                "synonyms": ["fantasy", "magic", "adventure", "epic"],
                "keywords": ["fantasy", "magic", "adventure", "epic"],
                "search_intent": "genre",
                "corrected_typos": false,
                "confidence_score": 0.98
            }}
            
            RETURN ONLY JSON. No additional text.
            """
    
    def _parse_refinement_response(self, response: str) -> Dict:
        """Parse a resposta do Groq como JSON"""
        try:
            # Encontrar JSON na resposta (remover possíveis markdown)
            response = response.strip()
            
            # Remover ```json ... ``` se existir
            if response.startswith('```json'):
                response = response[7:-3].strip()
            elif response.startswith('```'):
                response = response[3:-3].strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Garantir estrutura mínima
            if "normalized_query" not in data:
                data["normalized_query"] = data.get("original_query", "")
            
            if "keywords" not in data:
                data["keywords"] = data["normalized_query"].split()
            
            if "synonyms" not in data:
                data["synonyms"] = []
            
            if "search_intent" not in data:
                data["search_intent"] = "general"
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Erro ao parsear JSON do refinamento: {e}")
            logger.error(f"Resposta recebida: {response}")
            
            # Fallback
            return {
                "original_query": "",
                "normalized_query": "",
                "synonyms": [],
                "keywords": [],
                "search_intent": "general",
                "corrected_typos": False,
                "confidence_score": 0.5
            }
    
    async def expand_with_context(self, query: str, context: List[Dict], language: str = 'pt') -> Dict:
        """
        Expande a query com contexto da conversa
        """
        logger.info(f"🔄 Expandindo query com contexto: '{query}'")
        
        # Resumir contexto
        context_summary = self._summarize_context(context)
        
        prompt = f"""
        Você está em uma conversa sobre recomendações de livros.
        
        CONTEXTO DA CONVERSA:
        {context_summary}
        
        NOVA MENSAGEM DO USUÁRIO:
        "{query}"
        
        SUA TAREFA: Criar uma query de busca expandida que considere:
        1. O contexto anterior
        2. A nova mensagem
        3. Termos relacionados
        4. Possíveis continuidades
        
        EXEMPLO:
        Contexto: usuário perguntou sobre "superman"
        Nova mensagem: "e da marvel?"
        Query expandida: "marvel comics spider-man x-men avengers iron man"
        
        RETORNE APENAS a query expandida como string simples.
        """
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            expanded_query = await self.groq_service.chat(messages)
            return {"expanded_query": expanded_query.strip()}
        except Exception as e:
            logger.error(f"❌ Erro ao expandir query: {e}")
            return {"expanded_query": query}
    
    def _summarize_context(self, context: List[Dict]) -> str:
        """Resume o contexto da conversa"""
        if not context:
            return "Sem contexto anterior."
        
        summary = []
        for msg in context[-3:]:  # Últimas 3 mensagens
            role = "Usuário" if msg.get("role") == "user" else "Assistente"
            content = msg.get("content", "")[:100]
            summary.append(f"{role}: {content}")
        
        return "\n".join(summary)