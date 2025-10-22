import logging
import asyncio
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastEmbeddingService:
    def __init__(self):
        # Use the better embedding model from old system
        self.model_name = "BAAI/bge-small-en-v1.5"
        self.dimension = 384
        self._model = None
        # Optimize for speed - preload model
        self._ensure_model_loaded()
        logger.info("FastEmbeddingService initialized with BAAI/bge-small-en-v1.5")

    def _ensure_model_loaded(self):
        """Load model on initialization for faster subsequent calls"""
        if self._model is None:
            logger.info(f"Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            # Optimize for speed but maintain quality
            self._model.max_seq_length = 512
            # Use faster inference settings
            self._model._target_device = 'cpu'
            logger.info("BAAI/bge-small-en-v1.5 loaded and optimized")

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding generation with better model"""
        if not texts:
            return []

        try:
            # Better text processing for quality
            processed_texts = [' '.join(text.split()[:500]) for text in texts]
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._encode_batch_quality,
                processed_texts
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings with BAAI/bge-small-en-v1.5")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            # Return simple fallback embeddings
            return [[0.1] * self.dimension for _ in texts]

    def _encode_batch_quality(self, texts: List[str]) -> List[List[float]]:
        """Quality-focused encoding"""
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=min(len(texts), 256),  # Smaller batches for stability
            convert_to_numpy=True,
            convert_to_tensor=False,
            device='cpu'
        )
        
        return embeddings.tolist()

    async def get_embedding(self, text: str) -> List[float]:
        """Get single embedding"""
        embeddings = await self.get_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.1] * self.dimension

# Global instance
fast_embedding_service = FastEmbeddingService()