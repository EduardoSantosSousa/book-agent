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
            #book_id = int(book.get('bookId', idx))
            book_id = int(book.get('book_id', book.get('bookid', idx + 1)))

            
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
                    book_id=int(book.get('book_id', book.get('bookid', idx + 1))),
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
                    #book_id=int(book.get('bookId', idx)),
                    book_id=int(book.get('book_id', book.get('bookid', idx + 1))),
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
                #book_id=int(book.get('bookId', 0)),
                book_id=int(book.get('book_id', book.get('bookid', 0))),
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
        
        #book_row = self.data[self.data['bookId'] == book_id]
        if 'book_id' in self.data.columns:
            book_row = self.data[self.data['book_id'] == book_id]
        elif 'bookid' in self.data.columns:
            book_row = self.data[self.data['bookid'] == book_id]
        else:
            logger.warning("Nenhuma coluna de ID encontrada (book_id/bookid)")
            return None
        
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
    
    # Adicione este método à classe BookSearchEngine:

    def search(self, query: str, search_type: str = "hybrid", filters: Dict = None, k: int = 8) -> List[BookResult]:
        """Busca híbrida: combina semântica e textual"""
        logger.info(f"Buscando '{query}' com método: {search_type}")
        
        all_results = []
        
        # Busca semântica (sempre)
        semantic_results = self.search_by_semantic(query, filters, k=k*2)
        all_results.extend(semantic_results)
        
        # Busca textual (se habilitada ou híbrida)
        if search_type in ["textual", "hybrid"]:
            textual_results = self.search_by_textual(query, filters, k=k*2)
            all_results.extend(textual_results)
        
        # Remover duplicatas por book_id
        unique_results = self._remove_duplicates(all_results)
        
        # Ordenar por similaridade/relevância
        unique_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return unique_results[:k]
    
    # Em search_engine.py, adicione esta função de normalização:

    def _normalize_text_search(self, text: str) -> str:
        """Normaliza texto para busca"""
        text = text.lower().strip()
        
        # Remover acentos (simplificado)
        replacements = {
            'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a',
            'é': 'e', 'ê': 'e', 'è': 'e',
            'í': 'i', 'î': 'i', 'ì': 'i',
            'ó': 'o', 'ô': 'o', 'ò': 'o', 'õ': 'o',
            'ú': 'u', 'û': 'u', 'ù': 'u',
            'ç': 'c'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Normalizar variações de personagens
        normalizations = {
            'superhome': 'superman',
            'super home': 'superman',
            'super-homem': 'superman',
            'super homem': 'superman',
            'homem aranha': 'homem-aranha',
            'aranha': 'spider',
            'spider man': 'spider-man',
            'spiderman': 'spider-man',
            'marvel': 'marvel comics',
            'dc': 'dc comics',
            'quadrinhos': 'comics',
            'hq': 'comics'
        }
        
        for wrong, correct in normalizations.items():
            if wrong in text:
                text = text.replace(wrong, correct)
        
        return text



    def search_by_textual(self, query: str, filters: Dict = None, k: int = 16) -> List[BookResult]:
        """Busca textual simples nos campos de texto"""
        logger.info(f"Buscando textualmente: '{query}'")

        # Normalizar a query
        normalized_query = self._normalize_text_search(query)

        # Expandir com sinônimos básicos
        search_terms = [normalized_query]

        # Sinônimos simples
        synonyms = {
            'superman': ['man of steel', 'clark kent', 'super-homem'],
            'spider-man': ['homem-aranha', 'peter parker'],
            'batman': ['dark knight', 'bruce wayne'],
            'comics': ['quadrinhos', 'graphic novel', 'hq']
        }

        for term, syn_list in synonyms.items():
            if term in normalized_query:
                search_terms.extend(syn_list)
        
        start_time = time.time()
        results = []
        seen_titles = set()
        
        query_lower = query.lower()
        
        for idx, book in self.data.iterrows():
            # Aplicar filtros
            if filters and not self._check_filters(book, filters):
                continue
            
            # Verificar em vários campos
            score = 0
            
            # Título (maior peso)
            title = str(book.get('title', '')).lower()
            if query_lower in title:
                score += 3
            elif any(word in title for word in query_lower.split()):
                score += 2
            
            # Descrição
            description = str(book.get('description', '')).lower()
            if query_lower in description:
                score += 2
            elif any(word in description for word in query_lower.split()):
                score += 1
            
            # Personagens
            characters = str(book.get('characters', '')).lower()
            if 'homem-aranha' in query_lower and ('spider-man' in characters or 'homem-aranha' in characters):
                score += 5
            elif 'spider-man' in query_lower and ('spider-man' in characters or 'homem-aranha' in characters):
                score += 5
            
            # Gêneros
            genres_text = str(book.get('genres', '')).lower() + ' ' + str(book.get('maingenre', '')).lower()
            if 'comic' in query_lower or 'quadrinhos' in query_lower or 'hq' in query_lower:
                if 'comic' in genres_text or 'graphic' in genres_text or 'quadrinhos' in genres_text:
                    score += 2
            
            # Autor
            author_text = str(book.get('author', '')).lower()
            if 'marvel' in query_lower and 'marvel' in author_text:
                score += 2
            
            if score > 0:
                title = str(book['title'])
                
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                
                authors = self._extract_authors(book)
                genres = self._extract_genres(book)
                
                # Garantir book_id
                if 'book_id' in book:
                    book_id = int(book['book_id'])
                elif 'bookid' in book:
                    book_id = int(book['bookid'])
                else:
                    book_id = idx + 1
                
                result = BookResult(
                    book_id=book_id,
                    title=title,
                    authors=authors,
                    description=str(book.get('description', ''))[:200],
                    genres=genres,
                    rating=float(book.get('rating', 0)),
                    num_ratings=int(book.get('numRatings', 0)) if 'numRatings' in book else 0,
                    price=str(book.get('price', 'N/A')),
                    similarity_score=score / 10.0,  # Normalizar para 0-1
                    search_method="textual"
                )
                
                results.append(result)
                
                if len(results) >= k:
                    break
        
        # Ordenar por score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        search_time = time.time() - start_time
        logger.info(f"Busca textual: {len(results)} livros em {search_time:.2f}s")
        
        return results

    def _remove_duplicates(self, results: List[BookResult]) -> List[BookResult]:
        """Remove duplicatas por book_id"""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.book_id not in seen_ids:
                seen_ids.add(result.book_id)
                unique_results.append(result)
        
        return unique_results

    
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