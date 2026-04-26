"""
Diagram generator — Gemini writes PlantUML, Kroki renders to PNG.
"""

import base64
import zlib
import requests

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


def render_plantuml(uml_code: str) -> bytes | None:
    """
    Render PlantUML code to PNG bytes via Kroki.io.
    Returns None if rendering fails.
    """
    try:
        compressed = zlib.compress(uml_code.encode("utf-8"))
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
        url = f"https://kroki.io/plantuml/png/{encoded}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"[diagram_generator] Kroki rendering failed: {e}")
        return None


def generate_diagrams(structure: dict, profile: dict) -> list:
    """
    Generate all UML diagrams for the project.
    Returns list of {name, image_bytes, plantuml}.
    """
    results = []
    for diagram_type in DEFAULT_DIAGRAMS:
        try:
            plantuml = gemini_client.generate_plantuml(diagram_type, structure, profile)
            image_bytes = render_plantuml(plantuml)
            results.append({
                "name": diagram_type,
                "plantuml": plantuml,
                "image_bytes": image_bytes,
            })
        except Exception as e:
            print(f"[diagram_generator] Failed to generate {diagram_type}: {e}")
            results.append({
                "name": diagram_type,
                "plantuml": "",
                "image_bytes": None,
            })
    return results
