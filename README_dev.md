Development notes

- API endpoints:
  - POST /upload (requires API_KEY header if configured)
  - GET /status/{file_id}
  - POST /query
  - POST /query_rag
  - POST /query_agent

- To run tests:
  python -m pytest -q

- Docker build:
  docker build -t genai-docs .

- CI runs pytest.
