# D:\Django\book_agent\services\conversation_context.py

import json
import redis
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

class ConversationContextManager:
    """
    Gerencia contexto de conversação com persistência em Redis.
    Projetado para agentes LLM (RAG / Recomendação / Continuidade).
    """

    def __init__(
        self,
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        #redis_url: str = "redis://localhost:6379/0",
        #redis_url: str = "redis://redis:6379/0",
        #redis_url: str = "redis://redis-service.book-agent-ns.svc.cluster.local:6379/0",
        max_context_messages: int = 50,
        ttl_hours: int = 24,
    ):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.max_context_messages = max_context_messages
        self.context_ttl_seconds = ttl_hours * 3600

    # ------------------------------------------------------------------
    # 🔹 Helpers de serialização
    # ------------------------------------------------------------------

    # Adicione estes métodos na classe ConversationContextManager:

    def clear_session_data(self, session_id: str) -> Dict[str, any]:
        """Limpa TODOS os dados de uma sessão específica do Redis"""
        logger.info(f"🗑️  Limpando dados da sessão: {session_id}")
        
        try:
            # Lista de todas as chaves relacionadas a esta sessão
            session_keys = [
                f"conversation:{session_id}",
                f"chat:{session_id}:messages",
                f"chat:{session_id}:recommendations",
                f"chat:{session_id}:user_profile",
                f"chat:{session_id}:conversation_context",
                f"chat:{session_id}:history",
                f"chat:{session_id}:topics",
                f"chat:{session_id}:preferences"
            ]
            
            deleted_count = 0
            
            # Deletar cada chave
            for key in session_keys:
                if self.redis.exists(key):
                    self.redis.delete(key)
                    deleted_count += 1
                    logger.info(f"   ✅ Chave removida: {key}")
            
            # Também procurar por padrões de chave
            pattern_keys = self.redis.keys(f"*{session_id}*")
            for key in pattern_keys:
                if key not in session_keys:  # Para evitar duplicação
                    self.redis.delete(key)
                    deleted_count += 1
                    logger.info(f"   ✅ Chave por padrão removida: {key}")
            
            return {
                "success": True,
                "session_id": session_id,
                "deleted_keys": deleted_count,
                "message": f"Dados da sessão {session_id} limpos com sucesso"
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao limpar sessão {session_id}: {e}")
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e),
                "message": f"Erro ao limpar dados da sessão {session_id}"
            }

    def clear_all_sessions(self) -> Dict[str, any]:
        """Limpa TODAS as sessões do Redis (PERIGOSO - usar com cuidado)"""
        logger.warning("⚠️  LIMPANDO TODAS AS SESSÕES DO REDIS")
        
        try:
            # Encontrar todas as chaves de chat/conversa
            all_chat_keys = self.redis.keys("conversation:*")
            all_chat_keys.extend(self.redis.keys("chat:*"))
            
            if not all_chat_keys:
                return {
                    "success": True,
                    "deleted_keys": 0,
                    "message": "Nenhuma sessão encontrada para limpar"
                }
            
            # Remover duplicados
            all_chat_keys = list(set(all_chat_keys))
            
            # Deletar todas as chaves
            deleted_count = 0
            for key in all_chat_keys:
                self.redis.delete(key)
                deleted_count += 1
            
            logger.info(f"🗑️  Todas as sessões limpas: {deleted_count} chaves removidas")
            
            return {
                "success": True,
                "deleted_keys": deleted_count,
                "message": f"{deleted_count} sessões limpas com sucesso"
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao limpar todas as sessões: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Erro ao limpar todas as sessões"
            }

    def _serialize_session(self, session: Dict) -> Dict:
        return {
            **session,
            "created": session["created"].isoformat(),
            "last_activity": session["last_activity"].isoformat(),
            "discussed_books": list(session["discussed_books"]),
        }

    def _deserialize_session(self, data: Dict) -> Dict:
        return {
            **data,
            "created": datetime.fromisoformat(data["created"]),
            "last_activity": datetime.fromisoformat(data["last_activity"]),
            "discussed_books": set(data.get("discussed_books", [])),
        }

    def _session_key(self, session_id: str) -> str:
        return f"conversation:{session_id}"

    # ------------------------------------------------------------------
    # 🔹 Sessão
    # ------------------------------------------------------------------

    def get_or_create_session(self, session_id: str) -> Dict:
        key = self._session_key(session_id)
        raw = self.redis.get(key)

        if raw:
            session = self._deserialize_session(json.loads(raw))
        else:
            session = {
                "created": datetime.now(),
                "last_activity": datetime.now(),
                "conversation_history": [],
                "last_recommendations": None,
                "current_topic": None,
                "discussed_books": set(),
                "book_details_cache": {},
            }

        session["last_activity"] = datetime.now()

        self._save_session(session_id, session)
        return session

    def _save_session(self, session_id: str, session: Dict):
        self.redis.setex(
            self._session_key(session_id),
            self.context_ttl_seconds,
            json.dumps(self._serialize_session(session)),
        )

    # ------------------------------------------------------------------
    # 🔹 Mensagens
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        books: Optional[List[Dict]] = None,
        intent: Optional[str] = None,
    ):
        session = self.get_or_create_session(session_id)

        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "intent": intent,
        }

        if books:
            message["books"] = books
            session["last_recommendations"] = books

        session["conversation_history"].append(message)

        # Limitar histórico
        if len(session["conversation_history"]) > self.max_context_messages:
            session["conversation_history"] = session["conversation_history"][
                -self.max_context_messages :
            ]

        # Marcar livros discutidos
        if books:
            for book in books:
                book_id = book.get("book_id")
                if book_id is not None:
                    session["discussed_books"].add(book_id)

        self._save_session(session_id, session)

    # ------------------------------------------------------------------
    # 🔹 Contexto para Prompt
    # ------------------------------------------------------------------

    def get_conversation_context(
        self, session_id: str, max_messages: int = 4
    ) -> str:
        session = self.get_or_create_session(session_id)

        if not session["conversation_history"]:
            return "Primeira interação com o usuário."

        recent_messages = session["conversation_history"][-max_messages:]

        lines = [
            "CONTEXTO DA CONVERSA (use para manter continuidade e coerência):"
        ]

        for msg in recent_messages:
            role = "Usuário" if msg["role"] == "user" else "Assistente"
            lines.append(f"- {role}: {msg['content'][:200]}")

        if session["discussed_books"]:
            lines.append(
                f"- Livros já discutidos: {len(session['discussed_books'])}"
            )

        if session["last_recommendations"]:
            lines.append("- O assistente já fez recomendações recentemente.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 🔹 Recomendações
    # ------------------------------------------------------------------

    def get_last_recommendations(self, session_id: str) -> Optional[List[Dict]]:
        session = self.get_or_create_session(session_id)
        return session.get("last_recommendations")

    def get_book_from_recommendations(
        self,
        session_id: str,
        book_id: Optional[int] = None,
        book_title: Optional[str] = None,
    ) -> Optional[Dict]:
        recommendations = self.get_last_recommendations(session_id)

        if not recommendations:
            return None

        if book_id is not None:
            for book in recommendations:
                if book.get("book_id") == book_id:
                    return book

        if book_title:
            title_lower = book_title.lower()
            for book in recommendations:
                if title_lower in book.get("title", "").lower():
                    return book

        return None

    # ------------------------------------------------------------------
    # 🔹 Cache de detalhes de livros
    # ------------------------------------------------------------------

    def add_book_details(self, session_id: str, book_id: int, details: Dict):
        session = self.get_or_create_session(session_id)
        session["book_details_cache"][str(book_id)] = {
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
        self._save_session(session_id, session)

    def get_book_details(
        self, session_id: str, book_id: int
    ) -> Optional[Dict]:
        session = self.get_or_create_session(session_id)
        cached = session["book_details_cache"].get(str(book_id))
        return cached.get("details") if cached else None

    # ------------------------------------------------------------------
    # 🔹 Utilidades
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str):
        self.redis.delete(self._session_key(session_id))
