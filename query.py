"""Text lookup over the 3D spatial-memory DuckDB index.

Embeds a free-text query with google/embeddinggemma-2 and returns the top-k
most similar spatial units, each with its 3D location.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer

from build_index import DB_PATH, TABLE, load_units

MODEL_NAME = "google/embeddinggemma-2"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def lookup(query: str, k: int = 5, db_path: str = DB_PATH,
           surface_filter: str = None) -> List[Tuple]:
    model = _get_model()
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    q_emb = q_emb[0].astype(np.float64)

    con = duckdb.connect(db_path, read_only=True)
    if surface_filter:
        sql = f"""
            SELECT unit_id, center_x, center_y, center_z, surface_type, facing,
                   description, area, embedding
            FROM {TABLE} WHERE surface_type = ?
        """
        rows = con.execute(sql, [surface_filter]).fetchall()
    else:
        sql = f"""
            SELECT unit_id, center_x, center_y, center_z, surface_type, facing,
                   description, area, embedding
            FROM {TABLE}
        """
        rows = con.execute(sql).fetchall()
    con.close()

    if not rows:
        return []

    embs = np.array([np.asarray(r[-1], dtype=np.float64) for r in rows])
    # stored embeddings are L2-normalized; q_emb is normalized too
    sims = embs @ q_emb
    order = np.argsort(-sims)[:k]
    out = []
    for i in order:
        r = rows[i]
        out.append((r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], float(sims[i])))
    return out


def fmt(rows) -> str:
    lines = []
    lines.append(f"{'rank':<5} {'unit_id':<10} {'sim':>6}   {'x':>7} {'y':>7} {'z':>6}  {'type':<9} description")
    lines.append("-" * 100)
    for i, (uid, cx, cy, cz, st, facing, desc, area, sim) in enumerate(rows, 1):
        lines.append(f"{i:<5} {uid:<10} {sim:>6.3f}   {cx:>7.2f} {cy:>7.2f} {cz:>6.2f}  {st:<9} {desc}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your text query\" [k] [surface_type]")
        sys.exit(2)
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    surface_filter = sys.argv[3] if len(sys.argv) > 3 else None
    rows = lookup(query, k=k, surface_filter=surface_filter)
    print(f"\nQuery: {query!r}  (top {k}{' | surface=' + surface_filter if surface_filter else ''})\n")
    print(fmt(rows))


if __name__ == "__main__":
    main()
