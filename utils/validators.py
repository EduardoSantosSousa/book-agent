from pydantic import ValidationError
from typing import Type, Any

def validate_request(schema_class: Type, data: dict) -> Any:
    """Valida dados de requisição usando um schema Pydantic"""
    if not data:
        raise ValueError("Dados da requisição não fornecidos")
    
    try:
        return schema_class(**data)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error['loc'])
            errors.append(f"{field}: {error['msg']}")
        
        raise ValueError(f"Erro de validação: {', '.join(errors)}")