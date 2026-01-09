# scripts/test_conversation_api.py
import requests
import json
import time

BASE_URL = "http://localhost:8080/api/v1"
SESSION_ID = "test-conversation-001"

def test_conversation():
    print("💬 Testando conversação via API...")
    print("=" * 50)
    
    # Teste de saúde primeiro
    print("\n🔍 Verificando saúde da API...")
    health_resp = requests.get(f"{BASE_URL}/health")
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        print(f"✅ API saudável: {health_data}")
    else:
        print(f"❌ API não saudável: {health_resp.text}")
        return
    
    # 1. Primeira mensagem
    print("\n👤 Usuário: Recomende livros de fantasia para iniciantes")
    
    payload = {
        "message": "Recomende livros de fantasia para iniciantes",
        "session_id": SESSION_ID,
        "language": "pt"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat", 
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Resposta: {data['data']['response'][:100]}...")
            print(f"📚 Livros encontrados: {data['data']['books_found']}")
            
            if data['data']['books']:
                print("📖 Livros recomendados:")
                for i, book in enumerate(data['data']['books'][:3], 1):
                    print(f"   {i}. {book['title']} - ⭐ {book['rating']}")
        else:
            print(f"❌ Erro: {response.status_code} - {response.text}")
            # Tentar obter mais detalhes
            try:
                error_data = response.json()
                print(f"Detalhes do erro: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Resposta bruta: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Exceção durante requisição: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conversation()