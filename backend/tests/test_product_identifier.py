
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Retrieve globally mocked modules
mock_groq = sys.modules["groq"]

from app.services.product_identifier import identify_product


def test_product_identifier_fallback():
    # Explicitly test regex fallback code path
    with patch("app.services.product_identifier.LLM_PROVIDER", "none"):
        result = identify_product("My printer X100 displays Error E105 because the cooling fan failed")
        
        assert result["product"] == "X100"
        assert result["model"] == "X100"
        assert result["category"] == "Printer"
        assert result["error_code"] == "E105"
        assert result["component"] == "Cooling Fan"
        assert result["product_family"] == "X-Series"


@patch("app.services.product_identifier.LLM_PROVIDER", "groq")
@patch("app.services.product_identifier.GROQ_API_KEY", "fakekey")
def test_product_identifier_llm():
    # Mock LLM API response returning structured JSON
    mock_client = MagicMock()
    mock_groq.Groq.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = """
    {
      "product": "A200",
      "model": "A200",
      "category": "Router",
      "error_code": "E202",
      "component": "Power Supply",
      "product_family": "A-Series",
      "version": "v1.2",
      "section": "Diagnostics",
      "page": 12
    }
    """
    mock_client.chat.completions.create.return_value = mock_response

    result = identify_product("Router model A200 showing E202 error on page 12")
    
    assert result["product"] == "A200"
    assert result["error_code"] == "E202"
    assert result["category"] == "Router"
    assert result["page"] == 12
