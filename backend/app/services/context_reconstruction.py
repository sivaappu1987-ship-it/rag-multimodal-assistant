"""
context_reconstruction.py — Resolves short follow-up answers during a clarification loop.
"""
import json
from typing import Tuple
from app.main import call_llm

def reconstruct_query(
    original_query: str,
    clarification_question: str,
    user_followup: str,
    product_hint: str = None
) -> Tuple[str, str]:
    """
    Returns (resolved_query, resolution_confidence).
    resolution_confidence is HIGH, MEDIUM, or LOW.
    """
    # Deterministic bypass for extremely long follow-ups that look like entirely new queries
    if len(user_followup.split()) > 20:
        # We will still try to merge, but we know it's robust.
        pass

    prompt = f"""You are a conversational AI context reconstruction module.
The user asked an original question, you asked a clarification question, and the user provided a follow-up answer.
Your job is to merge the original question and the follow-up answer into a single, fully resolved query that can be sent to a search engine.

Original Query: "{original_query}"
Clarification Question Asked: "{clarification_question}"
User Follow-up Answer: "{user_followup}"
Known Product Context: "{product_hint or 'Unknown'}"

Rules:
1. Preserve original user intent; use the clarification question as context to understand the follow-up.
2. Interpret short follow-ups semantically (e.g., if user says "too fast", combine it with the fan issue).
3. Preserve known product/model info.
4. Never invent technical facts not stated by the user.

Return ONLY a valid JSON object matching this schema exactly (do not wrap in markdown or backticks):
{{
  "resolved_query": "The complete, standalone sentence incorporating the user's answer into their original intent.",
  "resolution_confidence": "HIGH" (if perfectly clear), "MEDIUM" (if mostly clear), or "LOW" (if the follow-up makes no sense or changes the subject entirely)
}}
"""
    try:
        response_text = call_llm(prompt)
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_text)
        
        confidence = data.get("resolution_confidence", "LOW")
        if confidence not in ["HIGH", "MEDIUM", "LOW"]:
            confidence = "LOW"
            
        resolved_query = data.get("resolved_query", user_followup)
        return resolved_query, confidence
        
    except Exception as e:
        print(f"[ContextReconstruction] Failed: {str(e)}")
        return user_followup, "LOW"
