import re
from typing import List, Dict, Optional

def prompt_builder(question: str, context_docs: List[dict], chat_summary: Optional[str] = None, mode: str = "tutor", history: Optional[List[dict]] = None) -> str:
    """
    Assemble a prompt for the LLM, including system instructions, chat summary, context, and optional history.
    """
    system_prompt = {
        "tutor": "You are a knowledgeable tutor and academic guide. Use the provided context and, when helpful, your broader knowledge to teach the student clearly. Provide concise explanations, step-by-step reasoning, and encourage learning with follow-up questions.",
        "citations": "You are an academic assistant. Use ONLY the provided context. Cite facts with [C1], [C2], etc. Include References section at the end. Be concise.",
        "concise": "You are a helpful assistant. Answer concisely and clearly, using the provided context."
    }.get(mode, "You are a helpful assistant.")

    prompt_parts = [system_prompt]
    if chat_summary:
        prompt_parts.append(f"Chat summary: {chat_summary}")
    if history:
        for h in history[-3:]:
            prompt_parts.append(f"Previous Q: {h.get('question','')}\nA: {h.get('answer','')}")
    context_lines = []
    for idx, d in enumerate(context_docs, start=1):
        snippet = (d.get('content','') or '')[:250].strip()
        context_lines.append(f"[C{idx}] Source: {d.get('filename')} page {d.get('page')} chunk {d.get('chunk_index')}\n{snippet}")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("Context:\n" + "\n\n".join(context_lines))
    prompt_parts.append("Answer:")
    return "\n\n".join(prompt_parts)

# Simple post-processor for answer quality and citation normalization
def postprocess_answer(answer: str, context_docs: List[dict], min_words: int = 20, max_words: int = 300) -> str:
    # Normalize citation markers
    answer = re.sub(r"\[C\s*(\d+)\]", r"[C\1]", answer)
    answer = re.sub(r"\(C(\d+)\)", r"[C\1]", answer)
    # Remove hallucinated citations
    valid_ids = set(str(i+1) for i in range(len(context_docs)))
    answer = re.sub(r"\[C(\d+)\]", lambda m: m.group(0) if m.group(1) in valid_ids else "", answer)
    # Enforce length bounds
    words = answer.split()
    if len(words) < min_words:
        answer += "\n\n(Please expand your answer with one clarifying sentence and cite sources.)"
    elif len(words) > max_words:
        answer = " ".join(words[:max_words]) + "..."
    # Add minimal citation list if missing
    if "References" not in answer and context_docs:
        refs = "\nReferences:\n" + "\n".join([f"[C{i+1}] {d.get('filename')} page {d.get('page')}" for i, d in enumerate(context_docs[:2])])
        answer += refs
    return answer
