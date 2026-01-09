# diagnose_chat.py
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8080"

def diagnose_chat():
    print("🔍 Diagnóstico do Endpoint /chat")
    print("=" * 60)
    
    # 1. Testar endpoint de debug
    print("\n1. 🐛 Testando endpoint de debug...")
    try:
        payload = {"message": "teste de diagnóstico"}
        response = requests.post(f"{BASE_URL}/api/v1/debug/test-agent", 
                               json=payload, timeout=15)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            debug_info = data.get('debug_info', {})
            test_results = data.get('test_results', {})
            
            print(f"   ✅ Agente inicializado: {debug_info.get('agent_initialized', False)}")
            print(f"   ✅ Tem método process_message: {debug_info.get('has_process_message', False)}")
            
            if 'process_message_error' in test_results:
                print(f"   ❌ Erro no process_message: {test_results['process_message_error']}")
                if 'process_message_traceback' in test_results:
                    print(f"\n   Stack trace:")
                    lines = test_results['process_message_traceback'].split('\n')
                    for line in lines[:10]:  # Mostra só as primeiras 10 linhas
                        print(f"     {line}")
            
            if 'search_error' in test_results:
                print(f"   ❌ Erro na busca: {test_results['search_error']}")
                
            if test_results.get('process_message_works'):
                print(f"   ✅ process_message funciona!")
                print(f"     Resultado tem {len(test_results.get('result_keys', []))} chaves")
            
        else:
            print(f"   ❌ Erro: {response.text[:200]}")
            
    except Exception as e:
        print(f"   ❌ Erro na requisição: {e}")
    
    # 2. Testar endpoint /chat diretamente
    print("\n2. 💬 Testando endpoint /chat diretamente...")
    try:
        payload = {
            "message": "Recomende livros de fantasia",
            "session_id": "diagnose-user",
            "language": "pt"
        }
        
        response = requests.post(f"{BASE_URL}/api/v1/chat", 
                               json=payload, timeout=20)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"   ✅ /chat funcionou!")
            data = response.json()
            books_found = data.get('metadata', {}).get('books_found', 0)
            print(f"     Livros encontrados: {books_found}")
        elif response.status_code == 500:
            data = response.json()
            print(f"   ❌ Erro 500: {data.get('error', 'Erro desconhecido')}")
            if 'traceback' in data:
                print(f"\n   Stack trace (primeiras linhas):")
                lines = data['traceback'].split('\n')
                for line in lines[:15]:
                    print(f"     {line}")
        elif response.status_code == 503:
            print("   ⚠️  Agente não inicializado - aguarde alguns segundos")
        else:
            print(f"   ❌ Status inesperado: {response.status_code}")
            print(f"     Resposta: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("   ⏰ Timeout - O agente pode estar demorando muito")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
    
    print("\n" + "=" * 60)
    print("💡 Possíveis problemas:")
    print("1. Agente não inicializou completamente")
    print("2. Arquivos de embeddings não carregaram")
    print("3. Erro no método process_message do BookAgentService")
    print("4. Problema com o Ollama (se usado)")

if __name__ == "__main__":
    diagnose_chat()