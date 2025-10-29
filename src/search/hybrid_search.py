import os
import bm25s
from bm25s.tokenization import Tokenizer
from embeddings.embedding_manager import EmbeddingManager
import gzip
import joblib
import logging
from models import SearchQuery, SearchResult, SearchResponse, SearchType
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation
from typing import List, Dict, Any, Optional, Tuple, Set

# New imports for enhanced features
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from rapidfuzz import fuzz, process
from collections import defaultdict

from storage.csv_tables import LyricsTable, SongsTable


# Download required NLTK data (do this once in your setup) --> TO MOVE TO setup.py?
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Hybrid search engine combining keyword and semantic search with enhanced text matching"""
    
    def __init__(self, embedding_manager: EmbeddingManager, 
                 enable_lemmatization: bool = True,
                 enable_fuzzy_matching: bool = True,
                 enable_phrase_matching: bool = True,
                 data_dir: str = './data', 
                 persist_directory: str = './keyword_index'):
        """
        Initialize the search engine with optional enhanced features.
        
        Args:
            embedding_manager: Manager for semantic embeddings
            enable_lemmatization: Enable word lemmatization for better variant matching
            enable_fuzzy_matching: Enable fuzzy matching for typos and variations
            enable_phrase_matching: Enable exact phrase matching
            data_dir: Directory where data are stored
        """
        
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory, exist_ok=True)
        self.bm25_retriever_path = os.path.join(persist_directory, 'BM25') 
        self.tfidf_path = os.path.join(persist_directory, 'tfidf')
        if not os.path.exists(self.tfidf_path):
            os.makedirs(self.tfidf_path, exist_ok=True)
        self.phrase_index_path = os.path.join(persist_directory, 'phrase_index.pkl.gz')
        self.word_index_path = os.path.join(persist_directory, 'word_index.joblib')
        
        self.embedding_manager = embedding_manager
        
        # Enhanced feature flags
        self.enable_lemmatization = enable_lemmatization
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.enable_phrase_matching = enable_phrase_matching
        
        # Initialize lemmatizer if enabled
        self.lemmatizer = WordNetLemmatizer() if enable_lemmatization else None
        
        # Original TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=10000,
            tokenizer=self._custom_tokenizer if enable_lemmatization else None
        )

        # BM25 retriever
        self.BM25_retriever = bm25s.BM25()
        self.BM25_tokenizer = Tokenizer(
            lower=True, 
            splitter=self._custom_tokenizer if enable_lemmatization else None, 
            stopwords="en"
        )

        # Tables
        self.lyrics = LyricsTable(f'{data_dir}/lyrics.csv')
        self.songs = SongsTable(f'{data_dir}/songs.csv')
        
        self.tfidf_matrix = None
        self.song_ids = []
        self.lyrics_texts = []
        # New indices for enhanced features
        self.phrase_index = defaultdict(list) if enable_phrase_matching else None
        self.word_to_songs = defaultdict(set)  # For fuzzy matching

        # init keyword indexes
        all_lyrics = self.lyrics._read_all()

        if len(all_lyrics) > 0:
            self.lyrics_texts, self.song_ids = map(list, zip(*[(el.get('lyrics_text'), el.get('song_id')) for el in all_lyrics if el.get('lyrics_text')]))

            if self.enable_phrase_matching and os.path.exists(self.phrase_index_path):
                # using pickle and gzip as otherwise we get a huge file
                with gzip.open(self.phrase_index_path, 'rb') as f:
                    self.phrase_index = pickle.load(f)
            if self.enable_fuzzy_matching and os.path.exists(self.word_index_path):
                with open(self.word_index_path, 'rb') as f:
                    self.word_to_songs = joblib.load(f)
            if (self.enable_phrase_matching and not os.path.exists(self.phrase_index_path)) or (self.enable_fuzzy_matching and not os.path.exists(self.word_index_path)):
                self.build_phrase_and_word_index() 
       
            if os.path.exists(os.path.join(self.tfidf_path, 'tfidf_matrix.joblib')) and os.path.exists(os.path.join(self.tfidf_path, 'tfidf_vectorizer.joblib')):
                with open(os.path.join(self.tfidf_path, 'tfidf_matrix.joblib'), 'rb') as f:
                    self.tfidf_matrix = joblib.load(f)
                with open(os.path.join(self.tfidf_path, 'tfidf_vectorizer.joblib'), 'rb') as f:
                    self.tfidf_vectorizer = joblib.load(f)
                    # re-set the custom tokenizer
                    self.tfidf_vectorizer.set_params(**{'tokenizer': self._custom_tokenizer if enable_lemmatization else None})
            else:
                self.build_keyword_index(algo='tf-idf')

            if os.path.exists(self.bm25_retriever_path):
                self.BM25_retriever = bm25s.BM25.load(self.bm25_retriever_path)
                self.BM25_tokenizer.load_vocab(self.bm25_retriever_path)
            else:
                self.build_keyword_index(algo='bm25')

        
    def _custom_tokenizer(self, text: str) -> List[str]:
        """Custom tokenizer with lemmatization support"""
        tokens = word_tokenize(text.lower()) # Convert to lowercase and extract words
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def _extract_phrases(self, text: str, max_phrase_length: int = 4) -> List[Tuple[str, int]]:
        """Extract all n-gram phrases from text with their positions"""
        phrases = []
        words = ''.join(c for c in text if c not in punctuation).lower().split()
        
        for n in range(1, min(max_phrase_length + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                phrases.append((phrase, i))
        
        return phrases

    def build_phrase_and_word_index(self):
        """Build phrase and word indexes for enhanced search"""
        try:
            
            # Clear existing indices
            if self.phrase_index is not None:
                self.phrase_index = defaultdict(list) if self.enable_phrase_matching else None
            if self.word_to_songs is not None:
                self.word_to_songs = defaultdict(set) if self.enable_fuzzy_matching else None
 
            if self.enable_phrase_matching or self.enable_fuzzy_matching:

                for lyrics_data in zip(self.lyrics_texts, self.song_ids):
                    lyrics, song_id = lyrics_data
                    
                    # Build phrase index if enabled
                    if self.enable_phrase_matching:
                        phrases = self._extract_phrases(lyrics)
                        for phrase, position in phrases:
                            self.phrase_index[phrase].append({
                                'song_id': song_id,
                                'position': position,
                                'context': self._get_context(lyrics, position)
                            })
                    
                    # Build word index for fuzzy matching
                    if self.enable_fuzzy_matching:
                        words = set(lyrics.lower().split())
                        for word in words:
                            self.word_to_songs[word].add(song_id)

                if self.enable_phrase_matching:
                    logger.info(f"Built phrase index with {len(self.phrase_index)} unique phrases")
                    # save index (using pickle and gzip as otherwise we get a huge file)
                    with gzip.open(self.phrase_index_path, 'wb', compresslevel=6) as f:
                        pickle.dump(dict(self.phrase_index), f, protocol=pickle.HIGHEST_PROTOCOL)
                if self.enable_fuzzy_matching:
                    logger.info(f"Built word index with {len(self.word_to_songs)} words for fuzzy matching")
                    # save index
                    with open(self.word_index_path, 'wb') as f:
                        joblib.dump(self.word_to_songs, f) 
           
                
        except Exception as e:
            logger.error(f"Error building phrase/word index: {e}")

    
    def build_keyword_index(self, algo: str='tf-idf'):
        """Build TF-IDF index and additional indices for enhanced search"""
        try:
            self.lyrics_texts = []
            self.song_ids = []
            
            # Clear existing indices
            if self.phrase_index is not None:
                self.phrase_index.clear()
            self.word_to_songs.clear()

            # get lyrics data from DB
            all_lyrics = self.lyrics._read_all()
            self.lyrics_texts, self.song_ids = map(list, zip(*[(el.get('lyrics_text'), el.get('song_id')) 
                                                    for el in all_lyrics if el.get('lyrics_text')]))
          
            if self.lyrics_texts:
                # build additional indexes
                self.build_phrase_and_word_index() 

                # build keyword indexes
                if algo == 'tf-idf':
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.lyrics_texts)
                    logger.info(f"Built keyword index for {len(self.lyrics_texts)} songs for TF-IDF")
                    with open(os.path.join(self.tfidf_path, 'tfidf_matrix.joblib'), 'wb') as f:
                        joblib.dump(self.tfidf_matrix, f)
                    with open(os.path.join(self.tfidf_path, 'tfidf_vectorizer.joblib'), 'wb') as f:
                        # persisting the vectorizer with the _custom_tokenizer set will fail, so we force it to None
                        # before saving it and will re-set it correctly at loading time
                        self.tfidf_vectorizer.set_params(**{'tokenizer': None})
                        joblib.dump(self.tfidf_vectorizer, f)    
                elif algo == 'bm25':
                    corpus_tokens = self.BM25_tokenizer.tokenize(self.lyrics_texts)
                    self.BM25_retriever.index(corpus_tokens)
                    logger.info(f"Built keyword index for {len(self.lyrics_texts)} songs for BM25")
                    self.BM25_retriever.save(self.bm25_retriever_path)
                    self.BM25_tokenizer.save_vocab(self.bm25_retriever_path)

                else:
                    logger.warning(f"Unknown algo option: {algo}, cannot build index")

            else:
                logger.warning("No lyrics available for keyword indexing")
                
        except Exception as e:
            logger.error(f"Error building keyword index: {e}")
    
    def _get_context(self, text: str, position: int, window: int = 10) -> str:
        """Get context around a position in text"""
        words = text.split()
        start = max(0, position - window)
        end = min(len(words), position + window + 1)
        return ' '.join(words[start:end])
    
    def exact_phrase_search(self, phrase: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for exact phrase matches"""
        if not self.enable_phrase_matching or self.phrase_index is None:
            return []
        
        try:
            phrase_lower = phrase.lower().strip()
            results = []
            
            # Check if phrase exists in index
            if phrase_lower in self.phrase_index:
                matches = self.phrase_index[phrase_lower]
                
                # Group by song and count occurrences
                song_matches = defaultdict(list)
                for match in matches:
                    song_matches[match['song_id']].append(match)
                
                # Score based on frequency and create results
                for song_id, matches in song_matches.items():
                    idx = self.song_ids.index(song_id)
                    song_data = self.songs.get(self.song_ids[idx])
                    results.append({
                        'song_id': song_id,
                        'title': song_data.get('title', ''),
                        'artist': song_data.get('artist', ''),
                        'album': song_data.get('album', ''),
                        'lyrics_text': self.lyrics_texts[idx],
                        'lyrics_excerpt': matches[0]['context'],
                        'exact_match_score': 1, #len(matches)/ 10.0,  # Normalize by arbitrary factor
                        'match_count': len(matches),
                        'match_type': 'exact_phrase'
                    })
                
                # Sort by match count
                results.sort(key=lambda x: x['match_count'], reverse=True)
                results = results[:n_results]
            
            logger.info(f"Exact phrase search found {len(results)} results for: '{phrase}'")
            return results
            
        except Exception as e:
            logger.error(f"Error in exact phrase search: {e}")
            return []
    
    def fuzzy_search(self, query: str, n_results: int = 10, threshold: int = 85) -> List[Dict[str, Any]]:
        """Perform fuzzy search for handling typos and variations"""
        if not self.enable_fuzzy_matching:
            return []

        try:
            query_words = query.lower().split()
            song_scores = defaultdict(float)
            song_matches =  defaultdict(list)
            
            for query_word in query_words:
                # Find fuzzy matches for this word
                if len(query_word) > 2:  # Skip very short words
                    # Search against all words in our index
                    all_words = list(self.word_to_songs.keys())
                    fuzzy_matches = process.extract(
                        query_word, 
                        all_words, 
                        scorer=fuzz.ratio,
                        limit=10
                    )
                    # Add scores for songs containing fuzzy-matched words
                    for matched_word, score, _ in fuzzy_matches:
                        if score >= threshold:
                            for song_id in self.word_to_songs[matched_word]:
                                # Weight by fuzzy match score
                                song_scores[song_id] += score / 100.0
                                song_matches[song_id].append(matched_word)
            
            # Get top scoring songs
            if song_scores:
                sorted_songs = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
                
                results = []
                for song_id, score in sorted_songs:
                    idx = self.song_ids.index(song_id)
                    song_data = self.songs.get(song_id)
                    lyrics = self.lyrics_texts[idx]
                    results.append({
                        'song_id': song_id,
                        'title': song_data.get('title', ''),
                        'artist': song_data.get('artist', ''),
                        'album': song_data.get('album', ''),
                        'lyrics_text': lyrics, 
                        'lyrics_excerpt': self._get_context(lyrics, lyrics.lower().find(song_matches[song_id][0])),
                        'fuzzy_score': score / len(query_words),  # Normalize by query length
                        'match_type': 'fuzzy',
                        'matched_terms': song_matches[song_id]
                    })
                
                logger.info(f"Fuzzy search found {len(results)} results for query: {query}")
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"Error in fuzzy search: {e}")
            return []
    
    def keyword_search(self, query: str, algo: str='bm25', n_results: int = 10) -> List[Dict[str, Any]]:
        """Enhanced keyword-based search with optional exact phrase detection"""
        try:
            results = []

            # Check for exact matches first
            if self.enable_phrase_matching:
                exact_results = self.exact_phrase_search(query, n_results)
                
                # Combine with regular TF-IDF search
                if algo == 'tf-idf' and self.tfidf_matrix is not None:
                    tfidf_results = self._tfidf_search(query, n_results)
                    
                    # Merge results, prioritizing exact matches
                    seen_ids = {r['song_id'] for r in exact_results}
                    for r in tfidf_results:
                        if r['song_id'] not in seen_ids:
                            results.append(r)
                    
                    # Put exact matches first
                    results = exact_results + results[:n_results - len(exact_results)]

                elif algo == 'bm25' and hasattr(self.BM25_retriever, 'vocab_dict'):
                    bm25_results = self._bm25_search(query, n_results)
                    
                    # Merge results, prioritizing exact matches
                    seen_ids = {r['song_id'] for r in exact_results}
                    for r in bm25_results:
                        if r['song_id'] not in seen_ids:
                            results.append(r)
                    
                    # Put exact matches first
                    results = exact_results + results[:n_results - len(exact_results)]
                else:
                    results = exact_results
            else:
                if algo == 'tf-idf':
                    # Regular TF-IDF search
                    results = self._tfidf_search(query, n_results)
                elif algo == 'bm25':
                    results = self._bm25_search(query, n_results)
                else:
                    logger.warning(f"Unknown algo option: {algo}, cannot perform keyword search")

            
            # If few results and fuzzy matching enabled, supplement with fuzzy search
            if len(results) < n_results // 2 and self.enable_fuzzy_matching:
                fuzzy_results = self.fuzzy_search(query, n_results - len(results))
                seen_ids = {r['song_id'] for r in results}
                for r in fuzzy_results:
                    if r['song_id'] not in seen_ids:
                        results.append(r)
            
            # combine scores (currently, for each song only one score will be nonzero, so the combined lexical score is just that one score)
            for r in results:
                r['keyword_score'] = r.get('keyword_score', 0.0)
                r['exact_match_score'] = r.get('exact_match_score', 0.0)
                r['fuzzy_score'] = r.get('fuzzy_score', 0.0)
                r['combined_lexical_score'] = r['exact_match_score'] + r['fuzzy_score'] + r['keyword_score']

            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    def _tfidf_search(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Original TF-IDF search (extracted for clarity)"""
        if self.tfidf_matrix is None or not self.lyrics_texts:
            logger.warning("Keyword index not built, returning empty results")
            return []
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include results with some similarity
                song_data = self.songs.get(self.song_ids[idx])
                results.append({
                    'song_id': self.song_ids[idx],
                    'title': song_data.get('title', ''),
                    'artist': song_data.get('artist', ''),
                    'album': song_data.get('album', ''),
                    'lyrics_text': self.lyrics_texts[idx],
                    'keyword_score': float(similarities[idx]),
                    'match_type': 'keyword'
                })
        
        return results
        
        
    def _bm25_search(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """BM25 search"""
        if not hasattr(self.BM25_retriever, 'vocab_dict') or not self.lyrics_texts:
            logger.warning("Keyword index not built, returning empty results")
            return []
        
        # Transform query
        query_tokens = self.BM25_tokenizer.tokenize([query])
        
        # Get top results
        top_indices, scores = self.BM25_retriever.retrieve(query_tokens, k=n_results)

        # get maximum score to scale values to [0, 1]
        max_score = max(scores[0])
        if max_score < 1:
            max_score = 1 # do not scale if all scores are low to avoid magnifying them
        
        results = []
        for idx, score in zip(top_indices[0], scores[0]):
            if score > 0:  # Only include results with some similarity
                song_data = self.songs.get(self.song_ids[idx])
                results.append({
                    'song_id': self.song_ids[idx],
                    'title': song_data.get('title', ''),
                    'artist': song_data.get('artist', ''),
                    'album': song_data.get('album', ''),
                    'lyrics_text': self.lyrics_texts[idx],
                    'keyword_score': float(score)/max_score,
                    'match_type': 'keyword'
                })
        
        return results
    
    def semantic_search(self, query: str, n_results: int = 10, 
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings (unchanged)"""
        try:
            results = self.embedding_manager.search_similar(query, n_results, filters)
            
            # Add match type
            for result in results:
                result['match_type'] = 'semantic'
                result['semantic_score'] = result.pop('similarity_score', 0.0)
            
            logger.info(f"Semantic search found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def hybrid_search(self, query: str, keyword_algo: str='bm25', n_results: int = 10, 
                     semantic_weight: float = 0.7,
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Enhanced hybrid search combining all search methods"""
        try:
            # Perform all searches
            keyword_results = self.keyword_search(query, keyword_algo, n_results)
            semantic_results = self.semantic_search(query, n_results, filters)
            
            # Create lookup dictionaries with all scores
            all_scores = defaultdict(lambda: {
                'keyword_score': 0.0,
                'semantic_score': 0.0,
                'exact_match_score': 0.0,
                'fuzzy_score': 0.0,
                'combined_lexical_score': 0.0
            })
            
            # Collect all song IDs and scores
            for r in keyword_results:
                song_id = r['song_id']
                all_scores[song_id]['keyword_score'] = r.get('keyword_score', 0.0)
                all_scores[song_id]['exact_match_score'] = r.get('exact_match_score', 0.0)
                all_scores[song_id]['fuzzy_score'] = r.get('fuzzy_score', 0.0)
                all_scores[song_id]['combined_lexical_score'] = r.get('combined_lexical_score', 0.0)
                all_scores[song_id]['data'] = r
            
            for r in semantic_results:
                song_id = r['song_id']
                all_scores[song_id]['semantic_score'] = r.get('semantic_score', 0.0)
                if 'data' not in all_scores[song_id]:
                    all_scores[song_id]['data'] = r
            
            # Calculate hybrid scores with boost for exact matches
            results = []
            for song_id, scores in all_scores.items():

                hybrid_score =  (semantic_weight * scores['semantic_score']) + \
                            ((1 - semantic_weight) * scores['combined_lexical_score'])
               
                result_data = scores['data']
                result = {
                    'song_id': song_id,
                    'title': result_data.get('title', ''),
                    'artist': result_data.get('artist', ''),
                    'album': result_data.get('album', ''),
                    'lyrics_text': result_data.get('lyrics_text', ''),
                    'lyrics_excerpt': result_data.get('lyrics_excerpt', ''),
                    'hybrid_score': hybrid_score,
                    'exact_match_score': scores['exact_match_score'],
                    'fuzzy_score': scores['fuzzy_score'],
                    'keyword_score': scores['keyword_score'],
                    'combined_lexical_score': scores['combined_lexical_score'],
                    'semantic_score': scores['semantic_score'],
                    'match_type': 'hybrid'
                }
                results.append(result)
            
            # Sort by hybrid score
            results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            results = results[:n_results]
            
            logger.info(f"Hybrid search found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def search(self, search_query: SearchQuery, keyword_algo: str='bm25') -> SearchResponse:
        """Main search method that routes to appropriate search type (mostly unchanged)"""
        import time
        start_time = time.time()
        
        try:
            if search_query.search_type == SearchType.KEYWORD:
                results = self.keyword_search(search_query.query, keyword_algo, search_query.max_results)
            elif search_query.search_type == SearchType.SEMANTIC:
                results = self.semantic_search(search_query.query, search_query.max_results, search_query.filters)
            else:  # HYBRID
                semantic_weight = float(os.getenv("HYBRID_SEARCH_WEIGHT", 0.7))
                results = self.hybrid_search(
                    search_query.query,
                    keyword_algo,  
                    search_query.max_results, 
                    semantic_weight,
                    search_query.filters
                )
            
            # Convert to SearchResult objects
            search_results = []
            for result in results:
                # Get the best available score
                relevance_score = result.get('hybrid_score', 
                                            result.get('combined_lexical_score',
                                                      result.get('semantic_score', 0.0)))
                
                song_data = self.songs.get(result['song_id'])
                search_result = SearchResult(
                    song={
                        'song_id': result['song_id'],
                        'title': result.get('title', ''),
                        'artist': result.get('artist', ''),
                        'album': result.get('album'),
                        'original_id': song_data.get('original_id'),
                        'isrc_id': song_data.get('isrc_id', ''),
                        'duration_ms': song_data.get('duration_ms', None),
                        'popularity': song_data.get('popularity', None) if song_data.get('popularity') !='' else None,
                        'release_date': song_data.get('release_date', ''),
                        'source': song_data.get('source'),
                    },
                    lyrics_text=result.get('lyrics_text', ''),
                    lyrics_excerpt=result.get('lyrics_excerpt', ''),
                    relevance_score=relevance_score,
                    match_type=result['match_type'],
                    matched_terms=result.get('matched_terms', self._extract_matched_terms(search_query.query, result.get('lyrics_text', ''))) #lyrics_excerpt
                )
                search_results.append(search_result)
            
            processing_time = time.time() - start_time
            
            return SearchResponse(
                results=search_results,
                total_found=len(search_results),
                query=search_query.query,
                search_type=search_query.search_type,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return SearchResponse(
                results=[],
                total_found=0,
                query=search_query.query,
                search_type=search_query.search_type,
                processing_time=time.time() - start_time
            )
    
    def _extract_matched_terms(self, query: str, lyrics_excerpt: str) -> List[str]:
        """Enhanced term extraction with lemmatization support"""
        try:
            # Extract query terms
            query_terms = set(re.findall(r'\b\w+\b', query.lower()))
            
            # Apply lemmatization if enabled
            if self.lemmatizer:
                query_terms = {self.lemmatizer.lemmatize(term) for term in query_terms}
            
            # Extract lyrics terms
            lyrics_terms = set(re.findall(r'\b\w+\b', lyrics_excerpt.lower()))
            
            if self.lemmatizer:
                lyrics_terms = {self.lemmatizer.lemmatize(term) for term in lyrics_terms}
            
            matched = query_terms.intersection(lyrics_terms)
            return list(matched)[:5]  # Limit to 5 terms
            
        except Exception as e:
            logger.error(f"Error extracting matched terms: {e}")
            return []