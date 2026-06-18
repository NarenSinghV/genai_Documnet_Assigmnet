from typing import Callable, List
import os

class Agent:
    def __init__(self, name: str):
        self.name = name

    def plan(self, query: str) -> List[str]:
        # simple planner returns ordered steps
        return ["retrieve", "reason", "verify"]

    def retrieve(self, query: str, retriever: Callable):
        # retriever is a function that returns top docs
        return retriever(query)

    def reason(self, query: str, context: str) -> str:
        # call LLM to generate an answer from context
        try:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            prompt = f"Use the context to answer the question. Context:\n{context}\nQuestion: {query}\nAnswer:"
            resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=512)
            return resp.choices[0].text.strip()
        except Exception:
            return ""

    def verify(self, answer: str, context: str) -> bool:
        # basic verification: check that answer words appear in context (very naive)
        for w in answer.split()[:10]:
            if w.lower().strip('.,') not in context.lower():
                return False
        return True

    def run(self, query: str, retriever: Callable):
        steps = self.plan(query)
        docs = []
        for s in steps:
            if s == "retrieve":
                docs = self.retrieve(query, retriever)
            elif s == "reason":
                context = "\n\n".join([getattr(d, "page_content", None) or str(d) for d in docs])
                answer = self.reason(query, context)
            elif s == "verify":
                ok = self.verify(answer, context)
                if not ok:
                    return {"answer": answer, "verified": False}
        return {"answer": answer, "verified": True}
