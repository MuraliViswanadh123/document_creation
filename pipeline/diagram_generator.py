"""
Diagram generator — Gemini writes PlantUML, multiple renderers as fallbacks.
"""

import base64
import zlib
import requests
import traceback

from . import gemini_client


# Default diagrams to generate for any project
DEFAULT_DIAGRAMS = [
    "Use Case Diagram",
    "Class Diagram",
    "Sequence Diagram",
    "Activity Diagram",
    "Component Diagram",
    "Deployment Diagram",
]

# Multiple rendering endpoints — try in order
RENDER_ENDPOINTS = [
    {"name": "kroki.io (GET)", "type": "kroki_get", "url": "https://kroki.io"},
    {"name": "kroki.io (POST)", "type": "kroki_post", "url": "https://kroki.io/plantuml/png"},
    {"name": "PlantUML official", "type": "plantuml_official", "url": "https://www.plantuml.com/plantuml/png"},
]


def _encode_for_kroki(uml_code: str) -> str:
    """Encode PlantUML for Kroki GET URL (deflate + base64)."""
    compressed = zlib.compress(uml_code.encode("utf-8"))
    return base64.urlsafe_b64encode(compressed).decode("ascii")


def _encode_for_plantuml(uml_code: str) -> str:
    """Encode for PlantUML official server (uses ~1 prefix + their custom base64)."""
    compressed = zlib.compress(uml_code.encode("utf-8"))[2:-4]  # raw deflate, no zlib wrapper
    
    # PlantUML's custom base64 alphabet
    plantuml_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    base64_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    
    standard = base64.b64encode(compressed).decode("ascii").rstrip("=")
    translation = str.maketrans(base64_alphabet, plantuml_alphabet)
    return standard.translate(translation)


def render_plantuml(uml_code: str) -> tuple[bytes | None, str]:
    """
    Try multiple endpoints to render PlantUML to PNG.
    Returns (png_bytes_or_None, status_message).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (AutoDocs) AppleWebKit/537.36",
    }
    
    errors = []
    
    # Method 1: Kroki GET with URL-encoded payload
    try:
        encoded = _encode_for_kroki(uml_code)
        url = f"https://kroki.io/plantuml/png/{encoded}"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200 and r.content[:4] == b"\x89PNG":
            return r.content, "rendered via kroki.io GET"
        errors.append(f"kroki GET: HTTP {r.status_code}")
    except Exception as e:
        errors.append(f"kroki GET: {type(e).__name__}: {e}")
    
    # Method 2: Kroki POST
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
        errors.append(f"kroki POST: {type(e).__name__}: {e}")
    
    # Method 3: PlantUML official server
    try:
        encoded = _encode_for_plantuml(uml_code)
        url = f"https://www.plantuml.com/plantuml/png/~1{encoded}"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200 and r.content[:4] == b"\x89PNG":
            return r.content, "rendered via plantuml.com"
        errors.append(f"plantuml.com: HTTP {r.status_code}")
    except Exception as e:
        errors.append(f"plantuml.com: {type(e).__name__}: {e}")
    
    return None, " | ".join(errors)


def generate_diagrams(structure: dict, profile: dict) -> list:
    """
    Generate all UML diagrams for the project.
    Returns list of {name, image_bytes, plantuml, status}.
    """
    results = []
    for diagram_type in DEFAULT_DIAGRAMS:
        result = {
            "name": diagram_type,
            "plantuml": "",
            "image_bytes": None,
            "status": "",
        }
        
        # Step 1: Generate PlantUML code via Gemini
        try:
            plantuml = gemini_client.generate_plantuml(diagram_type, structure, profile)
            result["plantuml"] = plantuml
        except Exception as e:
            result["status"] = f"PlantUML generation failed: {e}"
            results.append(result)
            print(f"[diagram_generator] {diagram_type}: PlantUML generation failed: {e}")
            continue
        
        # Step 2: Render to PNG
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
            print(f"[diagram_generator] {diagram_type}: exception: {traceback.format_exc()}")
        
        results.append(result)
    
    return results