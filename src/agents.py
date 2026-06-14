from typing import List

class Agent:
    def __init__(self, name: str):
        self.name = name

    def plan(self, query: str) -> List[str]:
        # very small planning stub
        return ["retrieve", "read", "answer"]

    def run(self, query: str, retriever) -> str:
        # simple pipeline: retrieve top context and return a placeholder answer
        docs = retriever(query)
        context = "\n".join([d.page_content if hasattr(d, 'page_content') else str(d) for d in docs])
        return f"Agent {self.name} answer based on context: {context[:500]}"
