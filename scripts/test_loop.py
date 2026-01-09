import asyncio
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ollama_connection():
    """Testa conexão com Ollama"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Teste 1: Verifica se está rodando
            resp = await client.get("http://localhost:11434/api/tags")
            logger.info(f"Ollama está rodando: {resp.status_code}")
            
            # Teste 2: Faz uma chamada simples
            resp = await client.post("http://localhost:11434/api/chat", json={
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            })
            logger.info(f"Chat funcionou: {resp.status_code}")
            
            return True
    except Exception as e:
        logger.error(f"Erro: {e}")
        return False

def main():
    """Testa em loops diferentes"""
    print("Teste 1: Loop normal...")
    asyncio.run(test_ollama_connection())
    
    print("\nTeste 2: Loop fechado e recriado...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(test_ollama_connection())
    loop.close()
    
    print(f"\nResultado: {result}")

if __name__ == "__main__":
    main()