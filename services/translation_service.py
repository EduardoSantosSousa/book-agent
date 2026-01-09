import logging
from typing import Optional
from deep_translator import GoogleTranslator
import asyncio
import re

logger = logging.getLogger(__name__)

class TranslationService:
    """Serviço para tradução de texto entre idiomas usando deep-translator"""
    
    def __init__(self):
        self.english_translator = GoogleTranslator(source='auto', target='en')
        self.portuguese_translator = GoogleTranslator(source='auto', target='pt')
        
    async def translate_to_english(self, text: str, source_lang: str = 'auto') -> str:
        """
        Traduz texto para inglês
        
        Args:
            text: Texto a ser traduzido
            source_lang: Idioma de origem (padrão: auto-detect)
            
        Returns:
            Texto traduzido para inglês
        """
        try:
            if not text or not text.strip():
                return text
            
            # Verificar se já está em inglês (heurística simples)
            if self._is_english(text):
                logger.info(f"Texto já está em inglês: {text[:50]}...")
                return text
            
            logger.info(f"Traduzindo para inglês: {text[:50]}...")
            
            # Executar tradução em thread separada para não bloquear
            loop = asyncio.get_event_loop()
            translated = await loop.run_in_executor(
                None, 
                lambda: self.english_translator.translate(text)
            )
            
            if translated and translated.strip():
                logger.info(f"Tradução bem-sucedida: '{text[:30]}...' -> '{translated[:30]}...'")
                return translated
            else:
                logger.warning("Tradução retornou vazia, retornando texto original")
                return text
            
        except Exception as e:
            logger.error(f"Erro na tradução para inglês: {e}")
            return text
    
    async def translate_from_english(self, text: str, target_lang: str = 'pt') -> str:
        """
        Traduz texto do inglês para outro idioma
        """
        try:
            if not text or not text.strip():
                return text
            
            logger.info(f"Traduzindo de inglês para {target_lang}: {text[:50]}...")
            
            # Criar tradutor específico para o idioma de destino
            translator = GoogleTranslator(source='en', target=target_lang)
            
            loop = asyncio.get_event_loop()
            translated = await loop.run_in_executor(
                None,
                lambda: translator.translate(text)
            )
            
            return translated if translated and translated.strip() else text
                
        except Exception as e:
            logger.error(f"Erro na tradução do inglês: {e}")
            return text
    
    def _is_english(self, text: str) -> bool:
        """
        Verifica heuristicamente se o texto está em inglês
        
        Args:
            text: Texto para verificar
            
        Returns:
            True se parece ser inglês, False caso contrário
        """
        # Lista de palavras comuns em inglês
        english_indicators = [
            'the', 'and', 'is', 'in', 'to', 'of', 'that', 'it', 'with', 'for',
            'this', 'are', 'on', 'as', 'be', 'at', 'by', 'an', 'have', 'from',
            'or', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your',
            'can', 'said', 'there', 'use', 'each', 'which', 'she', 'do', 'how',
            'their', 'will', 'other', 'about', 'out', 'many', 'then', 'them',
            'these', 'so', 'some', 'her', 'would', 'make', 'like', 'him', 'into',
            'time', 'has', 'look', 'two', 'more', 'write', 'go', 'see', 'number',
            'no', 'way', 'could', 'people', 'my', 'than', 'first', 'water',
            'been', 'call', 'who', 'oil', 'its', 'now', 'find', 'long', 'down',
            'day', 'did', 'get', 'come', 'made', 'may', 'part'
        ]
        
        # Lista de palavras comuns em português para comparação
        portuguese_indicators = [
            'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da',
            'dos', 'das', 'em', 'no', 'na', 'nos', 'nas', 'por', 'pelo', 'pela',
            'pelos', 'pelas', 'com', 'para', 'é', 'são', 'se', 'que', 'não',
            'mais', 'como', 'mas', 'me', 'eu', 'tu', 'ele', 'ela', 'nós', 'vós',
            'eles', 'elas', 'meu', 'minha', 'teu', 'tua', 'seu', 'sua', 'nosso',
            'vosso', 'deles', 'delas', 'este', 'esta', 'esse', 'essa', 'aquele',
            'aquela', 'isto', 'isso', 'aquilo', 'aquilo', 'quem', 'qual', 'quais',
            'onde', 'quando', 'porque', 'porquê', 'como', 'quanto', 'quantos'
        ]
        
        if not text or len(text) < 10:
            return False
        
        text_lower = text.lower()
        
        # Contar ocorrências de palavras em inglês vs português
        english_count = sum(1 for word in english_indicators if f' {word} ' in f' {text_lower} ')
        portuguese_count = sum(1 for word in portuguese_indicators if f' {word} ' in f' {text_lower} ')
        
        # Se tem mais indicadores de inglês que português, provavelmente é inglês
        if english_count > portuguese_count:
            return True
        
        # Verificar caracteres específicos do português
        portuguese_chars = ['ç', 'ã', 'õ', 'á', 'é', 'í', 'ó', 'ú', 'â', 'ê', 'î', 'ô', 'û']
        if any(char in text for char in portuguese_chars):
            return False
        
        # Verificar palavras comuns em português no início
        portuguese_starts = ['você', 'gostaria', 'recomende', 'livro', 'livros', 'sobre']
        if any(text_lower.startswith(word + ' ') or f' {word} ' in f' {text_lower} ' for word in portuguese_starts):
            return False
        
        # Por padrão, assumir que não é inglês
        return False

# Singleton para o serviço de tradução
_translation_service = None

def get_translation_service():
    """Obtém a instância singleton do serviço de tradução"""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service