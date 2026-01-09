from services.conversation_memory import ConversationMemoryManager

memory = ConversationMemoryManager(
    redis_host="localhost",  # ⚠️ IMPORTANTE
    redis_port=6379
)

session_id = "teste_memoria_1"

memory.add_message(session_id, "user", "Olá Redis!")
memory.add_message(session_id, "assistant", "Oi! Memória funcionando.")

context = memory.get_conversation_context(session_id)
print(context)
