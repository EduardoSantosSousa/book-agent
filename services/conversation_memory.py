import redis
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class ConversationMemoryManager:
    """
    Gerencia contexto de conversação usando Redis
    (versão escalável e persistente)
    """

    def __init__(
        self,
        redis_host: str = "redis",
        #redis_host: str = "redis-service.book-agent-ns.svc.cluster.local",
        redis_port: int = 6379,
        ttl_hours: int = 24,
        max_context_messages: int = 50
    ):
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )

        self.ttl_seconds = ttl_hours * 3600
        self.max_context_messages = max_context_messages

    # -------------------------
    # Helpers internos
    # -------------------------
    def _key(self, session_id: str) -> str:
        return f"conversation:{session_id}"

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    def _save_session(self, key: str, session: Dict):
        self.redis.setex(
            key,
            self.ttl_seconds,
            json.dumps(session, default=str)
        )

    # -------------------------
    # Sessão
    # -------------------------
    def get_or_create_session(self, session_id: str) -> Dict:
        key = self._key(session_id)
        data = self.redis.get(key)

        if data:
            session = json.loads(data)
            session["last_activity"] = self._now()
            self._save_session(key, session)
            return session

        session = {
            "created": self._now(),
            "last_activity": self._now(),
            "conversation_history": [],
            "last_recommendations": None,
            "current_topic": None,
            "discussed_books": [],
            "book_details_cache": {}
        }

        self._save_session(key, session)
        return session

    # -------------------------
    # Mensagens
    # -------------------------
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        books: List[Dict] = None,
        intent: str = None
    ):
        session = self.get_or_create_session(session_id)

        message = {
            "timestamp": self._now(),
            "role": role,
            "content": content,
            "intent": intent
        }

        if books:
            message["books"] = books
            session["last_recommendations"] = books

            for book in books:
                book_id = book.get("book_id")
                if book_id and book_id not in session["discussed_books"]:
                    session["discussed_books"].append(book_id)

        session["conversation_history"].append(message)

        # Limitar histórico
        if len(session["conversation_history"]) > self.max_context_messages:
            session["conversation_history"] = session["conversation_history"][
                -self.max_context_messages:
            ]

        session["last_activity"] = self._now()
        self._save_session(self._key(session_id), session)

    # -------------------------
    # Contexto para LLM
    # -------------------------
    def get_conversation_context(
        self,
        session_id: str,
        max_messages: int = 4
    ) -> str:
        session = self.get_or_create_session(session_id)

        history = session.get("conversation_history", [])
        if not history:
            return "Não há histórico de conversa."

        recent_messages = history[-max_messages:]

        context_lines = []
        for msg in recent_messages:
            role_symbol = "👤" if msg["role"] == "user" else "🤖"
            context_lines.append(f"{role_symbol} {msg['content'][:200]}")

        if session.get("discussed_books"):
            context_lines.append(
                f"\n📚 Livros já mencionados nesta conversa: {len(session['discussed_books'])}"
            )

        return "\n".join(context_lines)

    # -------------------------
    # Recomendações
    # -------------------------
    def get_last_recommendations(self, session_id: str) -> Optional[List[Dict]]:
        session = self.get_or_create_session(session_id)
        return session.get("last_recommendations")

    def get_book_from_recommendations(
        self,
        session_id: str,
        book_id: int = None,
        book_title: str = None
    ) -> Optional[Dict]:
        recommendations = self.get_last_recommendations(session_id)
        if not recommendations:
            return None

        if book_id:
            for book in recommendations:
                if book.get("book_id") == book_id:
                    return book

        if book_title:
            title_lower = book_title.lower()
            for book in recommendations:
                if title_lower in book.get("title", "").lower():
                    return book

        return None

    # -------------------------
    # Cache de detalhes
    # -------------------------
    def add_book_details(self, session_id: str, book_id: int, details: Dict):
        session = self.get_or_create_session(session_id)

        session["book_details_cache"][str(book_id)] = {
            "details": details,
            "timestamp": self._now()
        }

        self._save_session(self._key(session_id), session)

    def get_book_details(self, session_id: str, book_id: int) -> Optional[Dict]:
        session = self.get_or_create_session(session_id)
        cached = session.get("book_details_cache", {}).get(str(book_id))
        return cached.get("details") if cached else None
