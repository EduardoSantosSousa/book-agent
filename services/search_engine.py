# D:\Django\book_agent\services\search_engine.py

import pandas as pd
import numpy as np
import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BookResult:
    book_id: int
    title: str
    authors: List[str]
    description: str
    genres: List[str]
    rating: float
    num_ratings: int
    price: str
    similarity_score: float
    search_method: str

class BookSearchEngine:
    def __init__(self, data: pd.DataFrame, embedding_service):
        self.data = data.reset_index(drop=True)
        self.embedding_service = embedding_service
        self.search_history = []
        
        logger.info(f"Motor de busca inicializado com {len(self.data)} livros")
        
    def search_by_semantic(self, query: str, filters: Dict = None, k: int = 10) -> List[BookResult]:
        """Busca semântica"""
        logger.info(f"Buscando semanticamente: '{query}'")
        
        start_time = time.time()
        
        # Buscar no índice
        indices, distances = self.embedding_service.semantic_search(query, k * 2)
        
        if len(indices) == 0 or indices[0] == -1:
            logger.warning("Busca semântica não retornou resultados")
            return []
        
        results = []
        seen_titles = set()
        
        for idx, dist in zip(indices, distances):
            if idx == -1 or idx >= len(self.data):
                continue
            
            book = self.data.iloc[idx]
            
            # Aplicar filtros
            if filters and not self._check_filters(book, filters):
                continue
            
            title = str(book['title'])
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            # Extrair informações
            authors = self._extract_authors(book)
            genres = self._extract_genres(book)
            description = str(book.get('description', ''))[:200]
            rating = float(book.get('rating', 0))
            num_ratings = int(book.get('numRatings', 0)) if 'numRatings' in book else 0
            price = str(book.get('price', 'N/A'))
            book_id = int(book.get('bookId', idx))
            
            # Calcular similaridade
            similarity = 1 / (1 + dist) if dist > 0 else 1.0
            
            result = BookResult(
                book_id=book_id,
                title=title,
                authors=authors,
                description=description,
                genres=genres,
                rating=rating,
                num_ratings=num_ratings,
                price=price,
                similarity_score=similarity,
                search_method="semantic"
            )
            
            results.append(result)
            
            if len(results) >= k:
                break
        
        search_time = time.time() - start_time
        
        # Registrar no histórico
        self.search_history.append({
            'query': query,
            'filters': filters,
            'results': len(results),
            'method': 'semantic',
            'time_seconds': search_time
        })
        
        logger.info(f"{len(results)} livros encontrados em {search_time:.2f}s")
        return results
    
    def search_by_genre(self, genre: str, limit: int = 10) -> List[BookResult]:
        """Busca por gênero (com traduções português-inglês)"""
        logger.info(f"Buscando livros do gênero: {genre}")
        
        # Mapeamento de traduções
        genre_translations = {
            'fantasia': ['fantasia', 'fantasy'],
            'ficção científica': ['ficção científica', 'science fiction', 'sci-fi', 'scifi', 'ficcao cientifica'],
            'romance': ['romance', 'romantic'],
            'terror': ['terror', 'horror'],
            'mistério': ['mistério', 'mystery', 'suspense'],
            'história': ['história', 'history', 'historical', 'historia'],
            'biografia': ['biografia', 'biography'],
            'autoajuda': ['autoajuda', 'self-help', 'self help'],
            'negócios': ['negócios', 'business'],
            'ciência': ['ciência', 'science'],
            'tecnologia': ['tecnologia', 'technology'],
            'culinária': ['culinária', 'culinaria', 'cooking', 'gastronomy'],
            'poesia': ['poesia', 'poetry'],
            'drama': ['drama', 'dramatic'],
            'comédia': ['comédia', 'comedia', 'comedy'],
        }
        
        # Expandir termos de busca
        search_terms = [genre.lower()]
        if genre.lower() in genre_translations:
            search_terms.extend(genre_translations[genre.lower()])
        
        start_time = time.time()
        results = []
        seen_titles = set()
        
        for idx, book in self.data.iterrows():
            # Verificar no gênero principal
            main_genre = str(book.get('main_genre', '')).lower()
            
            # Verificar em todos os gêneros
            all_genres = book.get('all_genres', [])
            all_genres_lower = []
            
            if isinstance(all_genres, list):
                all_genres_lower = [str(g).lower() for g in all_genres]
            elif pd.notnull(all_genres):
                all_genres_lower = [str(all_genres).lower()]
            
            # Verificar correspondência com qualquer termo de busca
            genre_match = False
            
            for term in search_terms:
                if term in main_genre:
                    genre_match = True
                    break
                if any(term in g for g in all_genres_lower):
                    genre_match = True
                    break
                if any(g in term for g in all_genres_lower):
                    genre_match = True
                    break
            
            if genre_match:
                title = str(book['title'])
                
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                
                authors = self._extract_authors(book)
                genres = self._extract_genres(book)
                
                result = BookResult(
                    book_id=int(book.get('bookId', idx)),
                    title=title,
                    authors=authors,
                    description=str(book.get('description', ''))[:200],
                    genres=genres[:3],
                    rating=float(book.get('rating', 0)),
                    num_ratings=int(book.get('numRatings', 0)) if 'numRatings' in book else 0,
                    price=str(book.get('price', 'N/A')),
                    similarity_score=0.0,
                    search_method="genre"
                )
                
                results.append(result)
                
                if len(results) >= limit:
                    break
        
        search_time = time.time() - start_time
        
        self.search_history.append({
            'query': genre,
            'search_terms': search_terms,
            'results': len(results),
            'method': 'genre',
            'time_seconds': search_time
        })
        
        logger.info(f"{len(results)} livros do gênero '{genre}' em {search_time:.2f}s")
        return results
    
    def search_by_author(self, author_name: str, limit: int = 10) -> List[BookResult]:
        """Busca por autor"""
        logger.info(f"Buscando livros do autor: {author_name}")
        
        start_time = time.time()
        results = []
        seen_titles = set()
        
        author_lower = author_name.lower()
        
        for idx, book in self.data.iterrows():
            authors = book.get('author', [])
            
            # Verificar se o autor está na lista
            author_match = False
            
            if isinstance(authors, list):
                author_match = any(author_lower in str(a).lower() for a in authors)
            elif pd.notnull(authors):
                author_match = author_lower in str(authors).lower()
            
            if author_match:
                title = str(book['title'])
                
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                
                author_list = self._extract_authors(book)
                genres = self._extract_genres(book)
                
                result = BookResult(
                    book_id=int(book.get('bookId', idx)),
                    title=title,
                    authors=author_list,
                    description=str(book.get('description', ''))[:200],
                    genres=genres[:3],
                    rating=float(book.get('rating', 0)),
                    num_ratings=int(book.get('numRatings', 0)) if 'numRatings' in book else 0,
                    price=str(book.get('price', 'N/A')),
                    similarity_score=0.0,
                    search_method="author"
                )
                
                results.append(result)
                
                if len(results) >= limit:
                    break
        
        search_time = time.time() - start_time
        
        self.search_history.append({
            'query': author_name,
            'results': len(results),
            'method': 'author',
            'time_seconds': search_time
        })
        
        logger.info(f"{len(results)} livros do autor '{author_name}' em {search_time:.2f}s")
        return results
    
    def search_by_popularity(self, filters: Dict = None, limit: int = 10) -> List[BookResult]:
        """Busca por popularidade (rating + número de avaliações)"""
        logger.info("Buscando livros populares...")
        
        start_time = time.time()
        
        # Aplicar filtros
        filtered_data = self._apply_filters(self.data, filters) if filters else self.data
        
        if len(filtered_data) == 0:
            logger.warning("Nenhum livro após filtros")
            return []
        
        # Garantir que as colunas existem
        if 'rating' not in filtered_data.columns or 'numRatings' not in filtered_data.columns:
            logger.warning("Colunas de rating não encontradas, ordenando aleatoriamente")
            filtered_data = filtered_data.sample(frac=1)
        else:
            # Ordenar por rating
            filtered_data = filtered_data.sort_values('rating', ascending=False)
        
        results = []
        seen_titles = set()
        
        for _, book in filtered_data.head(limit * 2).iterrows():
            title = str(book['title'])
            
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            authors = self._extract_authors(book)
            genres = self._extract_genres(book)
            
            result = BookResult(
                book_id=int(book.get('bookId', 0)),
                title=title,
                authors=authors,
                description=str(book.get('description', ''))[:200],
                genres=genres[:3],
                rating=float(book.get('rating', 0)),
                num_ratings=int(book.get('numRatings', 0)) if 'numRatings' in book else 0,
                price=str(book.get('price', 'N/A')),
                similarity_score=0.0,
                search_method="popularity"
            )
            
            results.append(result)
            
            if len(results) >= limit:
                break
        
        search_time = time.time() - start_time
        
        self.search_history.append({
            'query': 'popular_books',
            'filters': filters,
            'results': len(results),
            'method': 'popularity',
            'time_seconds': search_time
        })
        
        logger.info(f"{len(results)} livros populares em {search_time:.2f}s")
        return results
    
    def get_book_by_id(self, book_id: int) -> Optional[BookResult]:
        """Busca livro por ID"""
        logger.info(f"Buscando livro com ID: {book_id}")
        
        book_row = self.data[self.data['bookId'] == book_id]
        
        if book_row.empty:
            logger.warning(f"Livro com ID {book_id} não encontrado")
            return None
        
        book = book_row.iloc[0]
        
        authors = self._extract_authors(book)
        genres = self._extract_genres(book)
        
        return BookResult(
            book_id=int(book_id),
            title=str(book['title']),
            authors=authors,
            description=str(book.get('description', ''))[:200],
            genres=genres,
            rating=float(book.get('rating', 0)),
            num_ratings=int(book.get('numRatings', 0)) if 'numRatings' in book else 0,
            price=str(book.get('price', 'N/A')),
            similarity_score=1.0,
            search_method="id_lookup"
        )
    
    def _check_filters(self, book, filters: Dict) -> bool:
        """Verifica se um livro passa nos filtros"""
        if not filters:
            return True
        
        if 'min_rating' in filters:
            rating = float(book.get('rating', 0))
            if rating < filters['min_rating']:
                return False
        
        if 'author' in filters:
            author_filter = filters['author'].lower()
            authors = book.get('author', [])
            
            if isinstance(authors, list):
                if not any(author_filter in str(a).lower() for a in authors):
                    return False
            elif author_filter not in str(authors).lower():
                return False
        
        if 'genre' in filters:
            genre_filter = filters['genre'].lower()
            
            # Verificar gênero principal
            main_genre = str(book.get('main_genre', '')).lower()
            if genre_filter not in main_genre:
                # Verificar todos os gêneros
                all_genres = book.get('all_genres', [])
                if isinstance(all_genres, list):
                    if not any(genre_filter in str(g).lower() for g in all_genres):
                        return False
                elif genre_filter not in str(all_genres).lower():
                    return False
        
        return True
    
    def _apply_filters(self, data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Aplica filtros ao dataframe"""
        filtered = data.copy()
        
        if 'min_rating' in filters:
            if 'rating' in filtered.columns:
                filtered = filtered[filtered['rating'] >= filters['min_rating']]
        
        if 'author' in filters:
            author_filter = filters['author'].lower()
            
            def contains_author(row):
                authors = row.get('author', [])
                if isinstance(authors, list):
                    return any(author_filter in str(a).lower() for a in authors)
                elif pd.notnull(authors):
                    return author_filter in str(authors).lower()
                return False
            
            filtered = filtered[filtered.apply(contains_author, axis=1)]
        
        if 'genre' in filters:
            genre_filter = filters['genre'].lower()
            
            def contains_genre(row):
                # Verificar gênero principal
                main_genre = str(row.get('main_genre', '')).lower()
                if genre_filter in main_genre:
                    return True
                
                # Verificar todos os gêneros
                all_genres = row.get('all_genres', [])
                if isinstance(all_genres, list):
                    return any(genre_filter in str(g).lower() for g in all_genres)
                elif pd.notnull(all_genres):
                    return genre_filter in str(all_genres).lower()
                return False
            
            filtered = filtered[filtered.apply(contains_genre, axis=1)]
        
        return filtered
    
    def _extract_authors(self, book) -> List[str]:
        """Extrai autores de um livro"""
        authors = book.get('author', [])
        
        if isinstance(authors, list):
            return [str(a) for a in authors[:2]]  # Limitar a 2 autores
        elif pd.notnull(authors):
            return [str(authors)]
        
        return []
    
    def _extract_genres(self, book) -> List[str]:
        """Extrai gêneros de um livro"""
        genres = []
        
        # Adicionar gênero principal
        main_genre = book.get('main_genre')
        if pd.notnull(main_genre):
            genres.append(str(main_genre))
        
        # Adicionar outros gêneros
        all_genres = book.get('all_genres', [])
        if isinstance(all_genres, list):
            genres.extend([str(g) for g in all_genres[:2]])
        elif pd.notnull(all_genres):
            genres.append(str(all_genres))
        
        return list(set(genres))[:3]  # Remover duplicados e limitar
    
    def get_search_stats(self):
        """Retorna estatísticas das buscas"""
        if not self.search_history:
            return {"total_searches": 0, "avg_results": 0, "avg_time": 0}
        
        total_searches = len(self.search_history)
        avg_results = np.mean([h['results'] for h in self.search_history])
        avg_time = np.mean([h.get('time_seconds', 0) for h in self.search_history])
        
        return {
            "total_searches": total_searches,
            "avg_results": avg_results,
            "avg_time_seconds": avg_time,
            "history": self.search_history[-10:]  # Últimas 10 buscas
        }