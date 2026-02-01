from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum

class SearchMethod(str, Enum):
    SEMANTIC = "semantic"
    GENRE = "genre"
    AUTHOR = "author"
    POPULARITY = "popularity"

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="Mensagem do usuário")
    session_id: str = Field(default="default", description="ID da sessão do usuário")
    language: str = Field(default="pt", pattern="^(pt|en)$", description="Idioma (pt ou en)")
    
    @validator('message')
    def message_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Mensagem não pode estar vazia")
        return v.strip()

class SearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="Termo de busca semântica")
    genre: Optional[str] = Field(None, description="Gênero para filtro")
    author: Optional[str] = Field(None, description="Autor para filtro")
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="Rating mínimo (0-5)")
    limit: int = Field(default=10, ge=1, le=100, description="Número máximo de resultados")
    method: SearchMethod = Field(default=SearchMethod.SEMANTIC, description="Método de busca")
    
    @validator('method')
    def validate_method_params(cls, v, values):
        if v == SearchMethod.SEMANTIC and not values.get('query'):
            raise ValueError("Método 'semantic' requer um parâmetro 'query'")
        elif v == SearchMethod.GENRE and not values.get('genre'):
            raise ValueError("Método 'genre' requer um parâmetro 'genre'")
        elif v == SearchMethod.AUTHOR and not values.get('author'):
            raise ValueError("Método 'author' requer um parâmetro 'author'")
        return v

class BookResponse(BaseModel):
    book_id: int
    title: str
    authors: List[str]
    genres: List[str]
    rating: float
    num_ratings: int
    description: str
    price: str
    similarity_score: Optional[float] = None
    search_method: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    intent: str
    books_found: int
    processing_time_seconds: float
    session_id: str
    language: str
    books: List[BookResponse]
    original_message: Optional[str] = None  # NOVO
    translated_message: Optional[str] = None  # NOVO
    translation_applied: Optional[bool] = False  # NOVO

class StatsResponse(BaseModel):
    agent: Dict[str, Any]
    search: Dict[str, Any]
    ollama: Dict[str, Any]
    embeddings: Dict[str, Any]
    system: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    checks: Dict[str, bool]
    timestamp: Optional[float]