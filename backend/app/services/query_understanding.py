"""
query_understanding.py — Pre-retrieval query understanding and normalization layer.
Extracts intent, entities, ambiguities, and classifies input confidence for routing.
"""
import json
from typing import Dict, Any

def understand_query(raw_query: str) -> Dict[str, Any]:
    """
    Analyzes the raw user query and returns a structured understanding payload.
    Returns:
      original_query: str
      normalized_query: str
      intent: str
      entities: dict
      technical_terms: list[str]
      product_hint: str | None
      issue_hint: str | None
      ambiguities: list[str]
      input_confidence: Literal["HIGH", "MEDIUM", "LOW"]
    """
    from app.main import call_llm

    # Defensive fallback defaults
    fallback = {
        "original_query": raw_query,
        "normalized_query": raw_query,
        "intent": "unknown",
        "entities": {},
        "technical_terms": [],
        "product_hint": None,
        "issue_hint": None,
        "ambiguities": [],
        "input_confidence": "LOW"
    }
    
    prompt = f"""You are a query understanding module for a technical support assistant.
Analyze the following user query. Handle vague queries, incomplete sentences, spelling/grammar mistakes, colloquial language, technical term variants, partially incorrect terms, and noisy speech-to-text artifacts gracefully.

Return ONLY a valid JSON object matching this schema exactly (do not wrap in markdown or backticks):
{{
  "original_query": "{raw_query}",
  "normalized_query": "the corrected, clear, and standardized version of the query",
  "intent": "the core user intent (e.g., troubleshoot, question, greeting)",
  "entities": {{"key": "value"}},
  "technical_terms": ["list", "of", "terms"],
  "product_hint": "the specific product model if mentioned, otherwise null",
  "issue_hint": "the specific problem or symptom if mentioned, otherwise null",
  "ambiguities": ["If ambiguous, provide a concrete either/or clarification question. e.g. 'Do you mean the fan is running too fast, too slow, or not turning on at all?'"],
  "input_confidence": "HIGH" (if clear and actionable), "MEDIUM" (if partially unclear or ambiguous product), or "LOW" (if extremely vague, off-topic, or fragment)
}}

User Query: "{raw_query}"
"""

    try:
        response_text = call_llm(prompt)
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_text)
        
        # Ensure confidence is one of the allowed values
        confidence = data.get("input_confidence", "LOW")
        if confidence not in ["HIGH", "MEDIUM", "LOW"]:
            data["input_confidence"] = "LOW"
            
        return {
            "original_query": data.get("original_query", raw_query),
            "normalized_query": data.get("normalized_query", raw_query),
            "intent": data.get("intent", "unknown"),
            "entities": data.get("entities", {}),
            "technical_terms": data.get("technical_terms", []),
            "product_hint": data.get("product_hint"),
            "issue_hint": data.get("issue_hint"),
            "ambiguities": data.get("ambiguities", []),
            "input_confidence": data.get("input_confidence", "LOW")
        }
        
    except Exception as e:
        print(f"[QueryUnderstanding] Failed to parse LLM response: {str(e)}")
        return fallback
