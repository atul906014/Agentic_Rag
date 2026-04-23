from dataclasses import dataclass
from typing import Dict, List

from src.llm import HuggingFaceChatModel
from src.vector_store import VectorStore


ROUTER_PROMPT = """You are a routing agent for a chatbot.
Classify the user request into exactly one label:
- direct: general knowledge or conversational request that does not require local documents
- retrieve: request depends on local knowledge base or should use retrieval
- clarify: question is too vague to answer reliably

Return only one word: direct, retrieve, or clarify."""

QUERY_REWRITE_PROMPT = """Rewrite the user question into a concise retrieval query.
Keep important entities and intent.
Return only the rewritten query."""

ANSWER_WITH_CONTEXT_PROMPT = """You are a careful RAG assistant.
Use the provided context to answer the user question.
If the context is insufficient, say what is missing.
Prefer concise, grounded answers and mention source file names when relevant."""

DIRECT_ANSWER_PROMPT = """You are a helpful AI assistant.
Answer clearly and concisely."""


@dataclass
class AgentResult:
    answer: str
    route: str
    rewritten_query: str
    sources: List[Dict[str, str]]


class AgenticRAG:
    def __init__(self, llm: HuggingFaceChatModel, vector_store: VectorStore, top_k: int = 4):
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k

    def route(self, question: str) -> str:
        label = self.llm.generate(ROUTER_PROMPT, question, max_tokens=20).strip().lower()
        if label not in {"direct", "retrieve", "clarify"}:
            return "retrieve"
        return label

    def rewrite_query(self, question: str) -> str:
        return self.llm.generate(QUERY_REWRITE_PROMPT, question, max_tokens=80)

    def answer_direct(self, question: str) -> AgentResult:
        answer = self.llm.generate(DIRECT_ANSWER_PROMPT, question)
        return AgentResult(answer=answer, route="direct", rewritten_query="", sources=[])

    def answer_with_retrieval(self, question: str) -> AgentResult:
        rewritten_query = self.rewrite_query(question)
        sources = self.vector_store.search(rewritten_query, top_k=self.top_k)
        context = "\n\n".join(
            f"Source: {item['source']}\nContent: {item['content']}" for item in sources
        )
        prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context or 'No relevant context found.'}"
        )
        answer = self.llm.generate(ANSWER_WITH_CONTEXT_PROMPT, prompt)
        return AgentResult(
            answer=answer,
            route="retrieve",
            rewritten_query=rewritten_query,
            sources=sources,
        )

    def clarify(self, question: str) -> AgentResult:
        answer = (
            "I need a bit more detail before I answer that. "
            "Please mention the document, topic, or goal you want me to focus on."
        )
        return AgentResult(answer=answer, route="clarify", rewritten_query="", sources=[])

    def run(self, question: str) -> AgentResult:
        route = self.route(question)
        if route == "direct":
            return self.answer_direct(question)
        if route == "clarify":
            return self.clarify(question)
        return self.answer_with_retrieval(question)
