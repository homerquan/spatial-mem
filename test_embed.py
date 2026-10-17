"""Quick test: load google/embeddinggemma-2 and encode sample texts."""
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")
texts = [
    "a +y wall section at (-10.7, -0.5, 0.4) in the room",
    "the floor at (-10.0, -0.7, 0.2)",
    "the ceiling at height 5.0 m at (0.0, 2.0)",
    "a window at (2.0, 4.0, 1.5)",
]
emb = model.encode(texts, batch_size=4, convert_to_numpy=True, normalize_embeddings=True)
print("model:", model)
print("embedding shape:", emb.shape)
print("dtype:", emb.dtype)
print("norm of first:", float(np.linalg.norm(emb[0])))
print("sim wall-floor:", float(emb[0] @ emb[1]))
print("sim wall-wall-ish:", float(emb[0] @ emb[2]))
