from pathlib import Path
import sys

ROOT = Path(__file__).parent
# ensure project root on path for imports
sys.path.insert(0, str(ROOT))

try:
    from src.ingest import ingest_file_to_vectordb
except Exception as e:
    print("Failed to import ingest_file_to_vectordb:", e)
    raise

UPLOADS = ROOT / "uploads"
if not UPLOADS.exists():
    print("No uploads directory found at", UPLOADS)
    raise SystemExit(1)

for p in sorted(UPLOADS.iterdir()):
    # skip status dir and directories
    if p.is_dir():
        print("Skipping directory:", p.name)
        continue
    # skip json status files if any
    if p.suffix.lower() == ".json":
        print("Skipping json file:", p.name)
        continue
    try:
        print("Ingesting:", p.name)
        res = ingest_file_to_vectordb(str(p))
        print("Ingest result for", p.name, "=>", bool(res))
    except Exception as e:
        print("Error ingesting", p.name, "->", repr(e))

print("Reingest complete.")
