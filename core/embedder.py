"""
Shared embedding helpers (one model instance for the whole app).
 
Why chunking exists: all-MiniLM-L6-v2 only reads the first 256 word-pieces of
whatever it is given (~190 words). A real CV is 500-1500 words, so embedding
it in one piece would ignore everything after the first paragraph — skills and
experience further down would never influence the ranking. Instead every CV
(and the JD) is split into overlapping ~150-word chunks, each embedded on its
own.
"""
 
import numpy as np
 
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_WORDS = 150
CHUNK_OVERLAP = 25
MAX_CHUNKS = 30  # ~3,500 words; longer documents are cut here
 
_model = None
 
 
def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model
 
 
def embedding_dim() -> int:
    return get_model().get_embedding_dimension()
 
 
def chunk_text(text: str) -> list:
    words = text.split()
    if not words:
        return [""]
    step = CHUNK_WORDS - CHUNK_OVERLAP
    starts = list(range(0, len(words), step))
    # a final chunk made only of overlap adds nothing new
    if len(starts) > 1 and len(words) - starts[-1] <= CHUNK_OVERLAP:
        starts.pop()
    return [" ".join(words[s:s + CHUNK_WORDS]) for s in starts[:MAX_CHUNKS]]
 
 
def embed_chunks(chunks: list) -> np.ndarray:
    vecs = get_model().encode(
        chunks, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype=np.float32)