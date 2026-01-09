## **Book Agent API**

### **📚 About the Project**

The **Book Agent API** is an intelligent book recommendation and conversation API built with Flask, integrated with:
  - FAISS for semantic search
  - Ollama for LLM response generation
  - Google Cloud Storage (GCS) for embedding management
  - Redis for conversation context

### **🏗️ Project Architecture**
```
BOOK_AGENT/
├── api/                    # API endpoints
│   ├── book_conversation_routes.py
│   ├── consumer_routes.py
│   ├── middleware.py
│   └── routes.py
├── services/              # Business logic
│   ├── agent_service.py
│   ├── book_conversation_service.py
│   ├── conversation_context.py
│   ├── embedding_service.py
│   ├── gcs_consumer_service.py
│   ├── ollama_service.py
│   ├── response_generator.py
│   ├── search_engine.py
│   └── translation_service.py
├── models/                # Schemas and models
│   └── schemas.py
├── utils/                 # Utilities
│   ├── data_loader.py
│   ├── gcs_loader.py
│   └── validators.py
├── notebook/              # Analysis notebooks
├── scripts/               # Helper scripts
├── doc/                   # Documentation
├── config.py              # Configuration
├── app.py                 # Main application
├── requirements.txt       # Dependencies
└── docker-compose.yml     # Docker configuration
```
