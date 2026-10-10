"""Build the cached LaBSE dense index for the active corpus (CPU-friendly).

Run after prepare_data.py. The index stores L2-normalised embeddings (float16),
the document IDs in corpus order, and the model name, so the app can verify the
index still matches data/comments_clean.csv before using it.

    python build_labse_index.py              # builds only if missing/stale
    python build_labse_index.py --force      # always rebuild
"""
import argparse
import sys
import time

import numpy as np

from rag_service.retrieval.engine import LABSE_INDEX, LABSE_MODEL, load_corpus, load_labse_model


def index_is_current(ids, model_name):
    if not LABSE_INDEX.exists():
        return False
    try:
        archive = np.load(LABSE_INDEX, allow_pickle=False)
        stored_model = str(archive['model_name']) if 'model_name' in archive.files else LABSE_MODEL
        return archive['document_ids'].astype(str).tolist() == ids and stored_model == model_name
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--force', action='store_true', help='rebuild even if a matching index exists')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--model', default=LABSE_MODEL)
    args = ap.parse_args(argv)

    corpus = load_corpus()
    texts = [d['comment_text'] for d in corpus]
    ids = [d['document_id'] for d in corpus]
    if not args.force and index_is_current(ids, args.model):
        print(f'Index at {LABSE_INDEX} already matches the corpus ({len(ids):,} docs). Use --force to rebuild.')
        return 0

    print(f'Loading {args.model} on CPU (first run downloads ~1.8 GB of weights)...')
    try:
        model = load_labse_model(args.model)
    except RuntimeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    print(f'Embedding {len(texts):,} comments (batch size {args.batch_size})...')
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                              show_progress_bar=True, convert_to_numpy=True)
    tmp = LABSE_INDEX.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, embeddings=embeddings.astype(np.float16),
                        document_ids=np.array(ids), model_name=np.array(args.model))
    tmp.replace(LABSE_INDEX)  # atomic: a crash never leaves a half-written index
    print(f'Indexed {len(ids):,} comments in {time.time() - t0:.0f}s -> {LABSE_INDEX}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
