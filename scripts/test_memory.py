# test_memory.py
import asyncio
import redis
import json
import time

async def test_redis_memory():
    """Testa se o Redis está mantendo a memória entre requisições"""
    
    # Conecta ao Redis
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    session_id = "test-session-001"
    
    # Primeira mensagem
    print("📝 Teste 1: Primeira mensagem")
    conversation1 = {
        "messages": [
            {"role": "user", "content": "Olá, preciso de livros de programação"},
            {"role": "assistant", "content": "Recomendo 'Clean Code' e 'The Pragmatic Programmer'"}
        ],
        "timestamp": time.time()
    }
    
    r.set(f"conversation:{session_id}", json.dumps(conversation1))
    print(f"✅ Salvo no Redis: {r.get(f'conversation:{session_id}')[:50]}...")
    
    # Simula segunda requisição
    await asyncio.sleep(1)
    
    print("\n📝 Teste 2: Segunda mensagem (deve ter histórico)")
    stored = r.get(f"conversation:{session_id}")
    if stored:
        history = json.loads(stored)
        print(f"✅ Histórico recuperado: {len(history['messages'])} mensagens")
        
        # Adiciona nova mensagem
        history["messages"].append({"role": "user", "content": "Qual deles é melhor para iniciantes?"})
        history["timestamp"] = time.time()
        
        r.set(f"conversation:{session_id}", json.dumps(history))
        print(f"✅ Atualizado no Redis: {len(history['messages'])} mensagens totais")
    else:
        print("❌ Histórico não encontrado!")
    
    # Limpa
    r.delete(f"conversation:{session_id}")
    print("\n🧹 Teste concluído, dados limpos")

if __name__ == "__main__":
    asyncio.run(test_redis_memory())