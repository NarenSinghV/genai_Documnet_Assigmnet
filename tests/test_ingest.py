import tempfile
import os
from src.ingest import load_text_from_file, chunk_text


def test_load_text_from_file_txt():
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    try:
        t.write(b"Hello world")
        t.flush()
        t.close()
        s = load_text_from_file(t.name)
        assert "Hello world" in s
    finally:
        os.unlink(t.name)


def test_chunk_text():
    text = "\n".join(["line" * 200])
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 0

# Note: vector store tests that use OpenAI embeddings or Chroma are integration tests and
# may be skipped in CI unless API keys and services are available.
