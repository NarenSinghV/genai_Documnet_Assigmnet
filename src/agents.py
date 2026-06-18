from typing import Callable, List, Any, Dict
import os

class RetrievalAgent:
    def __init__(self, name: str, retriever: Callable[[str], List[Any]]):
        self.name = name
        self.retriever = retriever

    def retrieve(self, query: str, k: int = 5):
        return self.retriever(query)


class ReasoningAgent:
    def __init__(self, name: str):
        self.name = name

    def reason(self, query: str, context: str) -> str:
        # Use OpenAI to generate answer from context; fall back to a simple heuristic if not available
        try:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            prompt = f"Use the context to answer the question. Context:\n{context}\nQuestion: {query}\nAnswer:"
            resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=512)
            return resp.choices[0].text.strip()
        except Exception:
            # fallback: return short extract from context
            return context[:400]


class VerifierAgent:
    def __init__(self, name: str):
        self.name = name

    def verify(self, answer: str, context: str) -> Dict[str, Any]:
        # Basic verification: check overlap of key tokens; return score and verdict
        tokens = [t for t in answer.split() if len(t) > 3][:20]
        if not tokens:
            return {"verified": False, "score": 0.0}
        found = sum(1 for t in tokens if t.lower().strip('.,') in context.lower())
        score = found / len(tokens)
        return {"verified": score >= 0.6, "score": score}


# Simple tool stubs
def tool_calculator(expression: str) -> str:
    # very limited safe eval: allow digits and +-*/().
    import re
    if not re.fullmatch(r"[0-9+\-*/(). \t]+", expression):
        return "unsupported expression"
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception:
        return "error"


def tool_web_search(query: str) -> str:
    # stub: in real system integrate Bing/Google or internal search
    return f"[web_search_stub] results for: {query}"


def tool_python(code: str) -> str:
    # VERY DANGEROUS in prod; here a stub that echoes code
    return f"[python_stub] executed: {code[:200]}"


class AgentManager:
    def __init__(self, retriever: Callable[[str], List[Any]]):
        self.retriever = retriever
        self.retrieval_agent = RetrievalAgent("retriever", self.retriever)
        self.reasoning_agent = ReasoningAgent("reasoner")
        self.verifier_agent = VerifierAgent("verifier")

    def run(self, query: str) -> Dict[str, Any]:
        # Step 1: retrieve
        docs = self.retrieval_agent.retrieve(query)
        context = "\n\n".join([getattr(d, "page_content", None) or str(d) for d in docs])
        # Step 2: reason
        answer = self.reasoning_agent.reason(query, context)
        # Step 3: verify
        verification = self.verifier_agent.verify(answer, context)
        # If verification low, attempt tool-based augmentation
        tools_used = []
        if not verification.get("verified"):
            # Try web search for extra context
            web = tool_web_search(query)
            tools_used.append({"web_search": web})
            # Re-run reasoning with appended web results
            answer2 = self.reasoning_agent.reason(query, context + "\n\n" + web)
            verification2 = self.verifier_agent.verify(answer2, context + "\n\n" + web)
            if verification2.get("verified"):
                answer = answer2
                verification = verification2
        return {"answer": answer, "verified": verification.get("verified"), "score": verification.get("score"), "tools": tools_used}
