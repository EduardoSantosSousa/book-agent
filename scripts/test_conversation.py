# test_conversation.py
import requests
import json
import time

BASE_URL = "http://localhost:8080/api/v1"
SESSION_ID = "test-chatbot-001"

def test_conversation():
    print("🤖 Teste de Conversação com Histórico")
    print("=" * 50)
    
    # 1. Primeira mensagem
    print("\n1️⃣ Primeira mensagem:")
    response1 = requests.post(f"{BASE_URL}/chat", json={
        "message": "Quero livros sobre liderança para engenheiros",
        "session_id": SESSION_ID,
        "language": "pt"
    })
    
    if response1.status_code == 200:
        data1 = response1.json()
        print(f"✅ Resposta: {data1['data']['response'][:100]}...")
        print(f"📚 Livros recomendados: {len(data1['data']['books'])}")
    else:
        print(f"❌ Erro: {response1.text}")
        return
    
    time.sleep(1)
    
    # 2. Segunda mensagem (deve usar histórico)
    print("\n2️⃣ Segunda mensagem (deve referenciar anterior):")
    response2 = requests.post(f"{BASE_URL}/chat", json={
        "message": "Desses, qual é o mais focado em equipes técnicas?",
        "session_id": SESSION_ID,
        "language": "pt"
    })
    
    if response2.status_code == 200:
        data2 = response2.json()
        print(f"✅ Resposta: {data2['data']['response'][:150]}...")
        
        # Verificar se menciona livros anteriores
        response_text = data2['data']['response'].lower()
        if "anterior" in response_text or "mencionei" in response_text or "disse" in response_text:
            print("🎯 PERFEITO! O agente está usando o histórico!")
        else:
            print("⚠️ O agente pode não estar usando o histórico completamente")
    
    # 3. Verificar histórico salvo
    print("\n3️⃣ Verificando histórico salvo:")
    history_resp = requests.get(f"{BASE_URL}/books/conversation/history/{SESSION_ID}")
    if history_resp.status_code == 200:
        history = history_resp.json()
        print(f"📊 Histórico: {history['data']['message_count']} mensagens")
        print(f"📚 Livros discutidos: {len(history['data']['discussed_books'])}")

if __name__ == "__main__":
    test_conversation()