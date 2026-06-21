import os
import logging
from typing import List, Any, Dict
from google import genai
from google.genai import errors

logger = logging.getLogger("genai")

class RetrievalAgent:
    def __init__(self, name: str, vectordb: Any):
        self.name = name
        self.vectordb = vectordb

    def retrieve(self, query: str, source_name: str = None, k: int = 5) -> List[Any]:
        if source_name:
            logger.info(f"Filtering vector lookups exclusively for resource: {source_name}")
            return self.vectordb.similarity_search(query, k=k, filter={"source": str(source_name)})
        return self.vectordb.similarity_search(query, k=k)

class ReasoningAgent:
    def __init__(self, name: str):
        self.name = name
        self.model_name = "gemini-2.5-flash"

    def reason(self, query: str, context: str) -> str:
        # Fetches key directly inside request to prevent FastAPI startup timing bugs
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("System Environment Variable 'GEMINI_API_KEY' is missing or unreadable.")
            return "Configuration Error: Please verify that GEMINI_API_KEY is properly declared inside your .env file."

        clean_key = str(api_key).strip().strip('"').strip("'")
        
        try:
            client = genai.Client(api_key=clean_key)
        except Exception as init_err:
            logger.error(f"Failed to instantiate official GenAI Client framework: {str(init_err)}")
            return "Configuration Error: The provided Gemini API Key string layout is invalid."

        # High-density, constraint prompt to guarantee 10 clean single-sentence rows
        prompt = (
            "You are an expert research assistant agent. Use the context below to answer the question.\n"
            "Task: Summarise the text in exactly 10 short, simple, single-sentence points.\n"
            "Constraint: Every single point must be less than 15 words long. Do not write paragraphs.\n"
            "Format: Start immediately with '1. ' and end at '10. '. No introductory text.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "List Output:"
        )

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "max_output_tokens": 1500,
                    "temperature": 0.1
                }
            )
            
            if response.text:
                return response.text.strip()
            return "Error: The reasoning model generated an empty text response."
                
        except errors.APIError as api_err:
            logger.error(f"Google Server Error: {api_err.message} (Code: {api_err.code})")
            return f"Google API Error [{api_err.code}]: {api_err.message}"
        except Exception as e:
            logger.exception("Unexpected reasoning pipeline crash.")
            return f"Processing Error [SDK Debug: {type(e).__name__}]"

class VerifierAgent:
    def __init__(self, name: str):
        self.name = name

    def verify(self, answer: str, context: str) -> Dict[str, Any]:
        logger.info("[VerifierAgent] Beginning hallucination verification loop...")
        tokens = [t.lower().strip(".,!?()\"'") for t in answer.split() if len(t) > 4]
        if not tokens or "google api error" in answer.lower() or "processing error" in answer.lower():
            return {"verified": False, "score": 0.0}
        
        cleaned_context = context.lower()
        found = sum(1 for t in tokens if t in cleaned_context)
        score = found / len(tokens)
        logger.info(f"[VerifierAgent] Completed verification. Score: {round(score, 2)}")
        return {"verified": score >= 0.35, "score": score}

class AgentManager:
    def __init__(self, vectordb: Any):
        self.retrieval_agent = RetrievalAgent("retriever", vectordb)
        self.reasoning_agent = ReasoningAgent("reasoner")
        self.verifier_agent = VerifierAgent("verifier")

    def run(self, query: str, source_name: str = None) -> Dict[str, Any]:
        logger.info(f"[AgentManager] Starting Multi-Agent Pipeline Execution for query: '{query}'")
        docs = self.retrieval_agent.retrieve(query, source_name=source_name, k=5)
        if not docs:
            return {"answer": "No matching document entries found.", "verified": False, "sources": []}
            
        context = "\n\n".join([d.page_content for d in docs])
        answer = self.reasoning_agent.reason(query, context)
        verification = self.verifier_agent.verify(answer, context)
        
        sources_meta = []
        for d in docs:
            meta = getattr(d, "metadata", {})
            sources_meta.append({
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "snippet": d.page_content[:200] + "..."
            })

        return {
            "answer": answer,
            "verified": verification.get("verified"),
            "verification_score": round(verification.get("score", 0.0), 2),
            "sources": sources_meta
        }