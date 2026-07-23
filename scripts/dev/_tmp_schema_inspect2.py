import pyarrow.parquet as pq
from pathlib import Path

p = Path(r"c:/Users/[USERNAME]/Desktop/MINOR PROJECT/Stage_2_Intermediate_Files/_unified_aligned.parquet")

table = pq.read_table(p)
cols = table.column_names
print(p.name, "columns:", len(cols))
print("  sample cols:", cols[:15])
label_cols = [c for c in cols if c.lower() in {"label", "attack_cat", "class", "category", "type", "target", "y"}]
print("  label candidates:", label_cols)
