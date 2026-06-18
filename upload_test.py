import os
import requests

p = r"uploads\bb9b1485-7db9-43ce-9c7e-e88d3aaa023a_Assignment_DCA1110_Set 1 & Set 2_ QP_ MarchApril 2026.pdf"

h = {}
if os.getenv("API_KEY"):
    h["x-api-key"] = os.getenv("API_KEY")

with open(p, "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/upload",
        files={"file": (os.path.basename(p), f)},
        headers=h
    )

print(r.status_code)
print(r.text)
