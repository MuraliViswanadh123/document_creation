"""
Gemini API client — handles all LLM calls.
"""

import json
import google.generativeai as genai


def configure(api_key: str):
    """Configure the Gemini SDK with your API key."""
    genai.configure(api_key=api_key)


def _get_model(temperature: float = 0.7, json_mode: bool = False):
    """Get a configured Gemini 2.5 Flash model instance."""
    config = {
        "temperature": temperature,
        "max_output_tokens": 4096,
    }
    if json_mode:
        config["response_mime_type"] = "application/json"
    return genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config=config,
    )


# ============================================================
# 1. PROJECT ANALYSIS — code → project profile JSON
# ============================================================
def analyze_project(structure: dict, abstract: str = "", paper_text: str = "") -> dict:
    """Analyze codebase structure and produce a project profile."""
    code_summary = _summarize_structure(structure)

    prompt = f"""You are analyzing a software project to generate academic documentation.
Infer the project's purpose, domain, and technique from the evidence below.

## Code Structure
{code_summary}

## Abstract (if provided)
{abstract or "(not provided — infer from code)"}

## Base Paper Excerpt (if provided)
{paper_text[:2000] if paper_text else "(not provided)"}

## Task
Return a JSON object with these exact keys:
{{
  "title": "concise project title (5-8 words)",
  "domain": "application domain (e.g. 'computer vision', 'fintech')",
  "project_type": "one of: web_application, ml_pipeline, cli_tool, mobile_app, library, game, data_pipeline, desktop_app, api_service",
  "primary_language": "main programming language",
  "tech_stack": ["frameworks", "and", "libraries"],
  "detected_techniques": ["algorithms or methods used"],
  "key_papers": [{{"title": "...", "authors": "...", "year": 2020, "venue": "..."}}],
  "inferred_purpose": "2-3 sentence description",
  "architecture_pattern": "MVC, microservices, monolith, pipeline, etc.",
  "target_users": "who would use this"
}}

Return ONLY valid JSON. No markdown fences, no commentary."""

    model = _get_model(temperature=0.3, json_mode=True)
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {
            "title": "Software Project",
            "domain": "Software Engineering",
            "project_type": "library",
            "primary_language": (structure.get("languages") or ["Unknown"])[0],
            "tech_stack": [],
            "detected_techniques": [],
            "key_papers": [],
            "inferred_purpose": "A software project (analysis failed to parse).",
            "architecture_pattern": "modular",
            "target_users": "general users",
        }


# ============================================================
# 2. SECTION GENERATION
# ============================================================
SECTION_PROMPTS = {
    "Abstract": "Write a 4-paragraph academic abstract (~250 words): problem, approach, implementation, outcomes. End with 5-6 keywords in italics.",
    "Introduction": "Write a 6-paragraph Introduction (~700 words): broad context, problem area, existing approaches, motivation, project overview, document structure.",
    "Problem Statement": "Write a single dense paragraph (~200 words) on the specific problem, why current solutions are inadequate, and the scope.",
    "Objectives": "Write 6-8 bullet points starting with action verbs (Design, Develop, Implement, Evaluate). Each 1-2 sentences.",
    "Literature Review": "Write a Literature Review (~600 words) covering 4-6 related works. For each: technique, scope, contribution, limitation. Use [Author, Year] inline. End with comparison.",
    "Existing System": "Write 2-3 paragraphs (~400 words) describing existing solutions and their specific limitations.",
    "Proposed System": "Write 3-4 paragraphs (~500 words) on the proposed solution: principle, innovations, advantages, benefits.",
    "System Architecture": "Write a System Architecture section (~500 words) describing 5-6 numbered components, each with name, responsibility, technology, connections.",
    "Methodology": "Write 7-9 numbered methodology steps. Each has an action title and 2-3 sentence explanation covering input to output.",
    "Functional Requirements": "Write 5-7 functional requirement sub-sections with ### headings (e.g. UI, Data Processing, Result Display). Each has 5-6 specific bullets.",
    "Non-Functional Requirements": "Write 6-8 sub-sections with ### headings: Performance, Reliability, Scalability, Usability, Security, Maintainability, Compatibility. Each with 3-5 bullets.",
    "Implementation": "Write Implementation section (~700 words): development environment, core approach, technical decisions, component integration. Match the project's tech stack.",
    "Testing": "Write Testing section (~400 words) describing 3 testing categories (Functional, Performance, Validation) with methodology and outcomes.",
    "Conclusion": "Write Conclusion (~300 words, 3 paragraphs): summary of work, key contributions, final reflection.",
    "Future Scope": "Write 8-10 numbered Future Scope items. Each: bold heading + 1-2 sentences on the enhancement and benefit.",
    "References": "Generate 8-12 plausible IEEE-format references for this domain. Format: [N] A. Author, \"Title,\" Venue, vol. X, no. Y, pp. ZZ-ZZ, Year. Use real well-known papers when possible.",
    "UI Requirements": "Write UI Requirements section (~400 words): layout, key screens, interactions, accessibility.",
    "Model Architecture": "Write Model Architecture section (~600 words): input format, layers, output format, training, loss function, metrics.",
}


def generate_section(section_name: str, profile: dict, structure: dict) -> dict:
    """Generate a single section of the document."""
    instructions = SECTION_PROMPTS.get(
        section_name,
        f"Write a comprehensive {section_name} section (~500 words) appropriate for an academic project report."
    )

    prompt = f"""You are writing the **{section_name}** section of an academic engineering project report.

## Project Context
- Title: {profile.get('title', 'Untitled')}
- Domain: {profile.get('domain', 'Software Engineering')}
- Type: {profile.get('project_type', 'software')}
- Tech Stack: {', '.join(profile.get('tech_stack', []))}
- Techniques: {', '.join(profile.get('detected_techniques', []))}
- Purpose: {profile.get('inferred_purpose', '')}
- Architecture: {profile.get('architecture_pattern', '')}

## Section Instructions
{instructions}

## Writing Style
- Formal academic tone, third person
- Specific to this project (use context above), not generic boilerplate
- No filler phrases like "in this paper" or "we will discuss"
- Clean markdown format

Generate the section content now:"""

    model = _get_model(temperature=0.7)
    response = model.generate_content(prompt)
    content = response.text.strip()
    return {
        "name": section_name,
        "content": content,
        "word_count": len(content.split()),
    }


# ============================================================
# 3. PLANTUML GENERATION
# ============================================================
def generate_plantuml(diagram_type: str, structure: dict, profile: dict) -> str:
    """Generate PlantUML code for a specific diagram type."""
    classes_summary = json.dumps(structure.get("classes", [])[:20], indent=2)[:1500]
    functions_summary = json.dumps(structure.get("functions", [])[:20], indent=2)[:1000]

    prompt = f"""Generate a {diagram_type} for this project as PlantUML code.

## Project Context
- Title: {profile.get('title')}
- Type: {profile.get('project_type')}
- Tech Stack: {', '.join(profile.get('tech_stack', []))}
- Purpose: {profile.get('inferred_purpose', '')}

## Code Structure
Classes: {classes_summary}
Functions: {functions_summary}

## Requirements
- Output ONLY PlantUML code from @startuml to @enduml
- No markdown fences, no commentary
- Use real class/function names from above when relevant
- Keep visually clean — 5-10 main elements

Generate the PlantUML code:"""

    model = _get_model(temperature=0.4)
    response = model.generate_content(prompt)
    code = response.text.strip()

    # Clean up markdown fences if present
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[-1].startswith("```"):
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        code = "\n".join(lines)

    if "@startuml" not in code:
        code = f"@startuml\n{code}\n@enduml"
    return code


def _summarize_structure(structure: dict) -> str:
    """Build a compact text summary of code structure for prompts."""
    parts = [
        f"Languages: {', '.join(structure.get('languages', ['Unknown']))}",
        f"File count: {structure.get('file_count', 0)}",
        f"Total lines: {structure.get('total_lines', 0)}",
    ]
    if structure.get("classes"):
        parts.append(f"\nClasses ({len(structure['classes'])}):")
        for c in structure["classes"][:15]:
            parts.append(f"  - {c}")
    if structure.get("functions"):
        parts.append(f"\nFunctions ({len(structure['functions'])}):")
        for f in structure["functions"][:20]:
            parts.append(f"  - {f}")
    if structure.get("imports"):
        parts.append(f"\nImports ({len(structure['imports'])}):")
        for i in structure["imports"][:30]:
            parts.append(f"  - {i}")
    if structure.get("file_list"):
        parts.append(f"\nFiles ({len(structure['file_list'])}):")
        for f in structure["file_list"][:30]:
            parts.append(f"  - {f}")
    return "\n".join(parts)
