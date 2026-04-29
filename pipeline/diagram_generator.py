"""
Diagram generator — generates PlantUML (via Ollama or Gemini) and renders to PNG.
"""

import base64
import zlib
import requests

from . import gemini_client, ollama_client


DEFAULT_DIAGRAMS = [
    "Use Case Diagram",
    "Class Diagram",
    "Sequence Diagram",
    "Activity Diagram",
    "Component Diagram",
    "Deployment Diagram",
]


def _encode_for_kroki(uml_code: str) -> str:
    compressed = zlib.compress(uml_code.encode("utf-8"))
    return base64.urlsafe_b64encode(compressed).decode("ascii")


def _encode_for_plantuml(uml_code: str) -> str:
    compressed = zlib.compress(uml_code.encode("utf-8"))[2:-4]
    plantuml_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    base64_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    standard = base64.b64encode(compressed).decode("ascii").rstrip("=")
    translation = str.maketrans(base64_alphabet, plantuml_alphabet)
    return standard.translate(translation)


def render_plantuml(uml_code: str):
    """Try multiple endpoints to render PlantUML to PNG. Returns (png_bytes, status)."""
    headers = {"User-Agent": "Mozilla/5.0 (AutoDocs)"}
    errors = []
    
    try:
        encoded = _encode_for_kroki(uml_code)
        url = f"https://kroki.io/plantuml/png/{encoded}"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200 and r.content[:4] == b"\x89PNG":
            return r.content, "rendered via kroki.io GET"
        errors.append(f"kroki GET: HTTP {r.status_code}")
    except Exception as e:
        errors.append(f"kroki GET: {type(e).__name__}")
    
    try:
        r = requests.post(
            "https://kroki.io/plantuml/png",
            data=uml_code.encode("utf-8"),
            headers={**headers, "Content-Type": "text/plain"},
            timeout=30,
        )
        if r.status_code == 200 and r.content[:4] == b"\x89PNG":
            return r.content, "rendered via kroki.io POST"
        errors.append(f"kroki POST: HTTP {r.status_code}")
    except Exception as e:
        errors.append(f"kroki POST: {type(e).__name__}")
    
    try:
        encoded = _encode_for_plantuml(uml_code)
        url = f"https://www.plantuml.com/plantuml/png/~1{encoded}"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200 and r.content[:4] == b"\x89PNG":
            return r.content, "rendered via plantuml.com"
        errors.append(f"plantuml.com: HTTP {r.status_code}")
    except Exception as e:
        errors.append(f"plantuml.com: {type(e).__name__}")
    
    return None, " | ".join(errors)


def generate_diagrams(structure: dict, profile: dict, llm: str = "ollama") -> list:
    """
    Generate UML diagrams using the specified LLM for PlantUML code.
    
    Args:
        structure: parsed code structure
        profile: project profile
        llm: "ollama" or "gemini"
    """
    results = []
    
    if llm == "ollama":
        llm_module = ollama_client
    else:
        llm_module = gemini_client
    
    for diagram_type in DEFAULT_DIAGRAMS:
        result = {
            "name": diagram_type,
            "plantuml": "",
            "image_bytes": None,
            "status": "",
        }
        
        try:
            plantuml = llm_module.generate_plantuml(diagram_type, structure, profile)
            result["plantuml"] = plantuml
        except Exception as e:
            result["status"] = f"PlantUML generation failed ({llm}): {e}"
            results.append(result)
            print(f"[diagram_generator] {diagram_type} ({llm}): {e}")
            continue
        
        try:
            image_bytes, status = render_plantuml(plantuml)
            result["image_bytes"] = image_bytes
            result["status"] = status if image_bytes else f"render failed: {status}"
            if image_bytes:
                print(f"[diagram_generator] {diagram_type}: ✓ {status}")
            else:
                print(f"[diagram_generator] {diagram_type}: ✗ {status}")
        except Exception as e:
            result["status"] = f"render exception: {e}"
        
        results.append(result)
    
    return results