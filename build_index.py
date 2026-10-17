"""Build a DuckDB vector index from embedded spatial-memory units.

Table: spatial_memory(unit_id, voxel, center, bbox_min, bbox_max, surface_type,
facing, normal, triangle_count, area, description, embedding)

Lookup is cosine similarity over the stored (L2-normalized) 768-D embeddings
using DuckDB's built-in array_cosine_similarity — no vector extension needed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import List

import duckdb
import numpy as np
from scene_metadata import load_index_metadata

DB_PATH = "spatial_memory.duckdb"
TABLE = "spatial_memory"


def load_units(npz_prefix: str = "index_data"):
    embeddings = np.load(f"{npz_prefix}.npy")
    with open(f"{npz_prefix}.json") as f:
        units = json.load(f)
    return embeddings, units


def build_index(npz_prefix: str = "index_data", db_path: str = DB_PATH,
                recreate: bool = True):
    embeddings, units = load_units(npz_prefix)
    if len(embeddings) != len(units):
        raise ValueError("Embedding and spatial-unit counts do not match")
    dim = embeddings.shape[1]

    con = duckdb.connect(db_path)
    con.execute("BEGIN TRANSACTION")
    if recreate:
        con.execute(f"DROP TABLE IF EXISTS {TABLE}")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            unit_id VARCHAR PRIMARY KEY,
            voxel_x INTEGER, voxel_y INTEGER, voxel_z INTEGER,
            center_x DOUBLE, center_y DOUBLE, center_z DOUBLE,
            bbox_min_x DOUBLE, bbox_min_y DOUBLE, bbox_min_z DOUBLE,
            bbox_max_x DOUBLE, bbox_max_y DOUBLE, bbox_max_z DOUBLE,
            cell_min_x DOUBLE, cell_min_y DOUBLE, cell_min_z DOUBLE,
            cell_max_x DOUBLE, cell_max_y DOUBLE, cell_max_z DOUBLE,
            orientation INTEGER,
            surface_type VARCHAR,
            facing VARCHAR,
            normal_x DOUBLE, normal_y DOUBLE, normal_z DOUBLE,
            triangle_count INTEGER,
            area DOUBLE,
            description VARCHAR,
            embedding DOUBLE[]
        )
    """)

    schema = con.execute(f"DESCRIBE {TABLE}").fetchall()
    names = [column[0] for column in schema]
    # DuckDB's recursive Python binding is very slow for millions of nested
    # embedding values. Stream records to its native JSON reader instead.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", dir=Path(db_path).parent) as batch:
        for u, emb in zip(units, embeddings):
            row = (
                u["unit_id"],
                *map(int, u["voxel"]), *map(float, u["center"]),
                *map(float, u["bbox_min"]), *map(float, u["bbox_max"]),
                *map(float, u.get("cell_min", u["bbox_min"])),
                *map(float, u.get("cell_max", u["bbox_max"])),
                int(u.get("orientation", -1)), u["surface_type"], u["facing"],
                *map(float, u["normal"]), int(u["triangle_count"]),
                float(u["area"]), u["description"], emb.tolist(),
            )
            batch.write(json.dumps(dict(zip(names, row)), separators=(",", ":"))+"\n")
        batch.flush()
        if units:
            projections = ", ".join(f'CAST("{column[0]}" AS {column[1]})' for column in schema)
            con.execute(f"INSERT INTO {TABLE} SELECT {projections} FROM read_json_auto(?)", [batch.name])
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_surface ON {TABLE}(surface_type)")
    con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE}_metadata (metadata VARCHAR)")
    con.execute(f"DELETE FROM {TABLE}_metadata")
    con.execute(f"INSERT INTO {TABLE}_metadata VALUES (?)", [json.dumps(load_index_metadata(npz_prefix))])
    con.execute("COMMIT")
    n = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    print(f"DuckDB index built: {db_path}, table {TABLE}, {n} rows, embedding dim {dim}", flush=True)
    con.close()
    return con


def main():
    build_index()


if __name__ == "__main__":
    main()
