import requests
import time
from typing import List, Optional, Tuple
from .prompt_builder import prompt_builder, postprocess_answer

def generate_answer_engine(question: str, context_docs: List[dict], session_id: Optional[str] = None, chat_summary: Optional[str] = None, mode: str = "tutor", history: Optional[List[dict]] = None) -> Tuple[str, List[dict]]:
    """
    Unified answer generation engine: builds prompt, calls LLM, applies post-processing, returns answer and citations.
    """
    # Assign citation ids
    for idx, d in enumerate(context_docs, start=1):
        d['citation_id'] = idx
    prompt = prompt_builder(question, context_docs, chat_summary, mode, history)
    max_retries = 3
    initial_timeout = 90
    min_timeout = 30
    timeout_reduction_factor = 0.7
    timeout_count = 0
    answer = ""
    for attempt in range(max_retries + 1):
        current_timeout = max(min_timeout, initial_timeout * (timeout_reduction_factor ** attempt))
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3:latest",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7
                },
                timeout=current_timeout
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get('response', '')
            if not answer.strip():
                if attempt < max_retries:
                    timeout_count += 1
                    time.sleep((attempt + 1) * 2)
                    continue
                else:
                    answer = "Error: Received empty response from Ollama. Please try again with a simpler question."
            break
        except requests.exceptions.Timeout:
            timeout_count += 1
            if attempt < max_retries:
                time.sleep((attempt + 1) * 2)
            else:
                answer = f"Error: Request to Ollama timed out after {max_retries+1} attempts. Try asking a simpler question or check if Ollama is running properly."
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                time.sleep((attempt + 1) * 3)
            else:
                answer = "Error: Cannot connect to Ollama. Please check if Ollama is running and restart if necessary."
        except Exception as e:
            if attempt < max_retries:
                time.sleep((attempt + 1) * 2)
                timeout_count += 1
            else:
                answer = f"Error communicating with Ollama: {str(e)}. Please try again or check if Ollama is running properly."
                break
    answer = postprocess_answer(answer, context_docs)
    citations = [
        {
            'id': d['citation_id'],
            'filename': d.get('filename'),
            'page': d.get('page'),
            'chunk_index': d.get('chunk_index')
        }
        for d in context_docs[:max(3, min(len(context_docs), 6) - timeout_count)]
    ]
    return answer, citations
