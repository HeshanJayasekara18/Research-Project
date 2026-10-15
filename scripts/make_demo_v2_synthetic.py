"""Write the SYNTHETIC Corpus V2 test fixture to a folder for the PP1 frontend demo.

The 14 rows are the invented comments from tests/api/test_corpus_v2.py (no real data).
Real Corpus V2 must not be used until research-use / ethics authorisation is confirmed.

    python scripts/make_demo_v2_synthetic.py          # -> data/demo_v2_synthetic/comments_v2.csv
    $env:RAG_CORPUS_V2_DIR = "data/demo_v2_synthetic"; python run.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from tests.api.test_corpus_v2 import FIELDS, SYNTHETIC  # noqa: E402  (single source of the fixture)

out = ROOT / 'data' / 'demo_v2_synthetic'
out.mkdir(parents=True, exist_ok=True)
with (out / 'comments_v2.csv').open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(SYNTHETIC)
(out / 'README.txt').write_text('SYNTHETIC test fixture (invented comments). Not real YouTube data.\n', encoding='utf-8')
print(f'wrote {len(SYNTHETIC)} synthetic rows to {out.relative_to(ROOT)}')
