"""
session_store.py — Simple in-memory session store for tracking troubleshooting diagnostic histories.
"""
from typing import Dict, Any


class SessionStore:
    """In-memory session registry for agentic state retention across turns."""

    _instance = None

    def __new__(cls):
        # Singleton: share sessions database across API endpoints
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.sessions = {}
            print("[SessionStore] Diagnostic session database initialized.")
        return cls._instance

    def get(self, session_id: str) -> Dict[str, Any]:
        """Fetch an active session state by session_id, initializing it if empty."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "product": None,
                "issue": None,
                "step": 0,
                "status": "START",
                "history": [],
                "context": [],
                # Phase 2 Additions
                "last_valid_user_query": None,
                "last_normalized_query": None,
                "pending_clarification": False,
                "clarification_question": None,
                "clarification_context": None,
                "clarification_attempts": 0,
                "last_valid_response": None,
                "input_confidence": None,
                "retrieval_confidence": None,
            }
        return self.sessions[session_id].copy()

    def save(self, session_id: str, data: Dict[str, Any]):
        """Save/overwrite a session state."""
        self.sessions[session_id] = data.copy()

    def clear(self, session_id: str):
        """Delete a session's history."""
        if session_id in self.sessions:
            del self.sessions[session_id]
