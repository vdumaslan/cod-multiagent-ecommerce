from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--id-col", default="product_id")
    parser.add_argument("--text-col", default="product_document")
    parser.add_argument("--model-id", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--output-dir", default="seller-copilot/artifacts/retrieval")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_parquet).dropna(subset=[args.id_col, args.text_col]).copy()
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=42).reset_index(drop=True)
    texts = df[args.text_col].astype(str).tolist()

    model = SentenceTransformer(args.model_id)
    emb = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    emb = emb.astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(out_dir / "retrieval_index.faiss"))
    df[[args.id_col, args.text_col]].reset_index(drop=True).to_parquet(
        out_dir / "retrieval_lookup.parquet", index=False
    )
    np.save(out_dir / "embeddings.npy", emb)
    print(f"Saved retrieval artifacts to {out_dir}")


if __name__ == "__main__":
    main()


