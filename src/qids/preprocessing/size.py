import pyarrow.parquet as pq

table = pq.read_table("../DATA_stage_2_without_near_zero/stage2_sentinel_mask_train.parquet")
num_columns = table.num_columns
num_rows = table.num_rows
print("Number of columns:", num_columns)
print("Number of rows:", num_rows)