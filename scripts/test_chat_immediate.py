# scripts/test_chat_immediate.py
import requests
import json
import time
import sys
import os

BASE_URL = "http://127.0.0.1:8080"

def test_chat_immediate():
    print("🚀 Testando chat com agente já inicializado...")
    
    # Primeiro, verificar estrutura do agente
    print("\n1. 🔍 Verificando estrutura do agente...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/debug/agent-structure", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Agente inicializado: {data.get('agent_initialized', False)}")
            
            # Procurar por componentes importantes
            structure = data.get('structure', {})
            
            # Verificar se tem process_message
            if 'process_message' in structure:
                print(f"   ✅ Tem método process_message")
            
            # Verificar embeddings
            for key in structure:
                if 'embedding' in key.lower():
                    print(f"   🔍 Encontrado atributo relacionado a embeddings: {key}")
            
        else:
            print(f"   ❌ Erro: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Erro ao verificar estrutura: {e}")
    
    # Agora testar o chat
    print("\n2. 💬 Testando endpoint /chat...")
    
    payload = {
        "message": "Recomende livros de fantasia para iniciantes",
        "session_id": "test-user-123",
        "language": "pt"
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/v1/chat", 
                               json=payload, 
                               timeout=30)
        elapsed = time.time() - start_time
        
        print(f"   ⏱️  Tempo de resposta: {elapsed:.2f}s")
        print(f"   📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Sucesso!")
            print(f"     Livros encontrados: {data.get('metadata', {}).get('books_found', 0)}")
            print(f"     Intent: {data.get('metadata', {}).get('intent', 'unknown')}")
            
            # Mostrar primeiros livros se existirem
            if 'data' in data and 'books' in data['data']:
                books = data['data']['books']
                if books and len(books) > 0:
                    print(f"\n   📚 Primeiro livro recomendado:")
                    print(f"     Título: {books[0].get('title', 'N/A')}")
                    print(f"     Autor(es): {', '.join(books[0].get('authors', []))}")
                    print(f"     Gêneros: {', '.join(books[0].get('genres', []))}")
                    print(f"     Avaliação: {books[0].get('rating', 'N/A')}")
                    
        elif response.status_code == 500:
            data = response.json()
            print(f"   ❌ Erro interno: {data.get('error', 'Unknown error')}")
            if 'details' in data:
                print(f"     Detalhes: {data['details'][:200]}")
                
        elif response.status_code == 503:
            print("   ⚠️  Agente não pronto - pode ser problema de sincronização")
            
        else:
            print(f"   ❌ Status inesperado: {response.status_code}")
            print(f"     Resposta: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("   ⏰ Timeout - O agente pode estar processando")
    except Exception as e:
        print(f"   ❌ Erro na requisição: {e}")

if __name__ == "__main__":
    test_chat_immediate()