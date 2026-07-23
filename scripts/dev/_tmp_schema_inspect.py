import pyarrow.parquet as pq
from pathlib import Path

base = Path(r"c:/Users/[USERNAME]/Desktop/MINOR PROJECT/DataSets")
files = [
    base / "nsl_kdd_stage1_output.parquet",
    base / "unsw_nb15_stage1_output.parquet",
    base / "cicids2017_stage1_output.parquet",
]

for p in files:
    table = pq.read_table(p)
    cols = table.column_names
    print(p.name, "columns:", len(cols))
    print("  sample cols:", cols[:10])
    label_cols = [c for c in cols if c.lower() in {"label", "attack_cat", "class", "category", "type", "target"}]
    print("  label candidates:", label_cols)
