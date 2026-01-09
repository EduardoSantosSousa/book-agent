# services/ollama_service.py - VERSÃO SIMPLIFICADA E SEGURA
import httpx
import logging
import time
import asyncio
import threading
from typing import List, Dict

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        base_url = "http://ollama-service.book-agent-ns.svc.cluster.local:11434",
        #base_url: str = "http://localhost:11434",
        timeout: int = 800,  # Segundos, não milissegundos
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.response_times: List[float] = []
        self._client = None
        self._client_lock = threading.Lock()
        self._loop = None
        
    def _ensure_loop(self):
        """Garante que temos um loop de eventos válido"""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
        
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat com tratamento seguro de event loop"""
        loop = self._ensure_loop()
        
        # Se já estamos no loop correto, use-o
        if asyncio.get_event_loop() == loop:
            return await self._chat_impl(messages)
        else:
            # Executa no loop correto
            return await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    self._chat_impl(messages), 
                    loop
                )
            )
    
    async def _chat_impl(self, messages: List[Dict[str, str]]) -> str:
        """Implementação real do chat"""
        logger.info(f"📤 Enviando para Ollama: {len(messages)} mensagens no histórico")
        
        try:
            # Cria cliente novo para cada requisição (mais seguro)
            timeout = httpx.Timeout(self.timeout)
            async with httpx.AsyncClient(timeout=timeout) as client:
                start_time = time.time()
                
                # Teste de conexão
                try:
                    await client.get(f"{self.base_url}/api/tags", timeout=5.0)
                except Exception as e:
                    logger.error(f"❌ Ollama não está acessível: {e}")
                    return "O serviço Ollama não está disponível no momento."
                
                # Requisição principal
                try:
                    response = await client.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "stream": False,
                            "options": {"temperature": 0.7, "num_predict": 32000}
                        },
                        timeout=800
                    )
                except httpx.TimeoutException:
                    logger.error("❌ Timeout no Ollama")
                    return "O Ollama demorou muito para responder."
                
                elapsed = time.time() - start_time
                logger.info(f"⏱️  Ollama respondeu em {elapsed:.2f}s")
                self.response_times.append(elapsed)
                
                if response.status_code != 200:
                    logger.error(f"❌ Erro {response.status_code}: {response.text[:200]}")
                    return f"Erro {response.status_code} do Ollama."
                
                try:
                    data = response.json()
                    return data.get('message', {}).get('content', 'Sem resposta do Ollama.')
                except:
                    return "Resposta inválida do Ollama."
                    
        except Exception as e:
            logger.error(f"❌ Erro inesperado: {e}")
            return f"Erro: {str(e)[:100]}"
    
    async def health_check(self) -> bool:
        """Verifica saúde do Ollama"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except:
            return False
    
    async def close(self):
        """Fecha recursos"""
        if self._client:
            await self._client.aclose()