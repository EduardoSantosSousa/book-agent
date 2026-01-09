# test_all_routes.py
import requests
import json

BASE_URL = "http://127.0.0.1:8080"

def test_all_endpoints():
    print("🧪 Testando TODOS os endpoints...")
    print("=" * 60)
    
    endpoints = [
        ("GET", "/health", "Health Check"),
        ("GET", "/", "Página inicial"),
        ("GET", "/api/v1/", "Documentação API"),
        ("GET", "/api/v1/consumer/status", "Status do Consumidor"),
        ("GET", "/api/v1/stats", "Estatísticas do Sistema"),
        ("POST", "/api/v1/chat", "Chat (teste rápido)"),
        ("GET", "/api/v1/books/search?query=fantasia", "Busca de Livros"),
    ]
    
    for method, endpoint, description in endpoints:
        try:
            url = f"{BASE_URL}{endpoint}"
            
            if method == "POST" and endpoint == "/api/v1/chat":
                response = requests.post(url, json={
                    "message": "Livros de fantasia",
                    "session_id": "test-user",
                    "language": "pt"
                })
            else:
                response = requests.request(method, url)
            
            status = "✅" if 200 <= response.status_code < 300 else "❌"
            
            print(f"{status} {method} {endpoint}")
            print(f"   {description}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"   Erro: {response.text[:100]}")
            
            print()
            
        except Exception as e:
            print(f"❌ {method} {endpoint}")
            print(f"   {description}")
            print(f"   Erro: {str(e)[:100]}")
            print()
    
    print("=" * 60)
    print("✨ Teste concluído!")

if __name__ == "__main__":
    test_all_endpoints()