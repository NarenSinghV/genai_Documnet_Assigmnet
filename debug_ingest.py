from src.ingest import ingest_file_to_vectordb
import traceback

p = r"uploads\9c460336-04ae-4ceb-bf9a-eb2a6bc6ec46_bb9b1485-7db9-43ce-9c7e-e88d3aaa023a_Assignment_DCA1110_Set 1 & Set 2_ QP_ MarchApril 2026.pdf"

print("ingesting:", p)

try:
    db = ingest_file_to_vectordb(p)
    print("vectordb:", bool(db))
except Exception:
    traceback.print_exc()
