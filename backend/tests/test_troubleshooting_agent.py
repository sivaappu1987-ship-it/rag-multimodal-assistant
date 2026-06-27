
from unittest.mock import patch, MagicMock
import sys
import os
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.troubleshooting_agent import diagnose_and_propose


def test_troubleshooting_agent_logic():
    async def run_test():
        # Test fallback diagnostic question when no history is present
        with patch("app.services.troubleshooting_agent.LLM_PROVIDER", "none"):
            res = await diagnose_and_propose(
                context_chunks=[],
                history=[],
                last_message="I have error E105 on my X100"
            )
            assert res["decision"] == "QUESTION"
            assert "fan" in res["text"].lower()

            # Test action advice when history shows user answered "no"
            history = [{"question": "Is the cooling fan spinning at all?", "answer": "No"}]
            res2 = await diagnose_and_propose(
                context_chunks=[],
                history=history,
                last_message="No"
            )
            assert res2["decision"] == "ACTION"
            assert "rear panel" in res2["text"].lower()

    asyncio.run(run_test())
