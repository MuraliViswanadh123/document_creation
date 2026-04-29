"""
Ollama client — handles local LLM calls via qwen2.5-coder.

Used for bulk work where API quota is precious:
- Code analysis / project profiling
- Most narrative sections
- PlantUML diagram code generation
"""

import json
import ollama


# Default model - change here to swap models globally
DEFAULT_MODEL = "qwen2.5-coder:7b"

# Increase context window so Ollama can read large code files
# Default is 2048 which is too small. 8192 is sweet spot for 16GB RAM.
DEFAULT_NUM_CTX = 8192


def is_available() -> bool:
    """Check if Ollama is running and model is available."""
    try:
        models = ollama.list()
        # Handle both dict and ListResponse formats from different ollama-python versions
        if hasattr(models, "models"):
            model_list = models.models
        else:
            model_list = models.get("models", [])
        
        for m in model_list:
            name = m.get("name") if isinstance(m, dict) else getattr(m, "model", "")
            if DEFAULT_MODEL.split(":")[0] in name:
                return True
        return False
    except Exception as e:
        print(f"[ollama] Not available: {e}")
        return False


def _chat(prompt: str, json_mode: bool = False, temperature: float = 0.3, num_predict: int = 4096) -> str:
    """Single chat call to Ollama with sensible defaults."""
    options = {
        "num_ctx": DEFAULT_NUM_CTX,
        "temperature": temperature,
        "num_predict": num_predict,
    }
    
    response = ollama.chat(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=options,
        format="json" if json_mode else "",
    )
    return response["message"]["content"]


# ============================================================
# 1. PROJECT ANALYSIS
# ============================================================
def analyze_project(structure: dict, abstract: str = "", paper_text: str = "") -> dict:
    """Analyze codebase and produce a project profile via Ollama."""
    code_summary = _summarize_structure(structure)

    prompt = f"""Analyze this software project and return a structured JSON profile.

## Code Structure
{code_summary}

## Abstract (if provided)
{abstract or "(not provided — infer from code)"}

## Base Paper Excerpt
{paper_text[:1500] if paper_text else "(not provided)"}

## Required JSON Output Format
Return ONLY a valid JSON object with these exact keys:
{{
  "title": "concise project title (5-8 words)",
  "domain": "application domain",
  "project_type": "web_application | ml_pipeline | cli_tool | mobile_app | library | game | data_pipeline | desktop_app | api_service",
  "primary_language": "main programming language",
  "tech_stack": ["frameworks", "and", "libraries"],
  "detected_techniques": ["algorithms or methods used"],
  "key_papers": [{{"title": "...", "authors": "...", "year": 2020, "venue": "..."}}],
  "inferred_purpose": "2-3 sentence description",
  "architecture_pattern": "MVC, microservices, monolith, pipeline, etc.",
  "target_users": "who would use this"
}}

Return ONLY the JSON. No markdown fences, no commentary."""

    try:
        response_text = _chat(prompt, json_mode=True, temperature=0.2, num_predict=2048)
        return json.loads(response_text)
    except Exception as e:
        print(f"[ollama] analyze_project failed: {e}")
        # Fallback profile
        return {
            "title": "Software Project",
            "domain": "Software Engineering",
            "project_type": "library",
            "primary_language": (structure.get("languages") or ["Unknown"])[0],
            "tech_stack": [],
            "detected_techniques": [],
            "key_papers": [],
            "inferred_purpose": "A software project (analysis fallback).",
            "architecture_pattern": "modular",
            "target_users": "general users",
        }


# ============================================================
# 2. SECTION GENERATION
# ============================================================
SECTION_PROMPTS = {
    "Problem Statement": "Write a comprehensive Problem Statement (~600-800 words, 4-5 paragraphs): broader context, specific problem, why current solutions fail, scope/constraints, research questions.",
    
    "Objectives": "Write 8-10 detailed objectives. Each: action verb in **bold**, clear deliverable, 2-3 sentences. Format as markdown bullets.",
    
    "Literature Review": """Write a comprehensive Literature Review (~1800-2200 words):
- Open with introductory paragraph
- Cover 8-10 related works in detail. For EACH: [Author et al., Year], technique used, scope, contribution, limitation, relation to this project (6-8 sentences each)
- Group thematically
- End with 2-3 paragraph comparative analysis""",
    
    "Feasibility Study": """Write a comprehensive Feasibility Study (~1500-1800 words):
Open with 2-paragraph introduction.
Then 5 sub-sections with ### headings:
### Technical Feasibility (~350 words)
### Economic Feasibility (~350 words)
### Operational Feasibility (~300 words)
### Schedule Feasibility (~250 words)
### Legal and Ethical Feasibility (~250 words)
End with 2-paragraph overall conclusion.""",
    
    "Existing System": "Write detailed Existing System (~800-1000 words, 5-6 paragraphs): overview of current approaches, 2-3 representative systems described in detail, specific limitations, why they matter, gap that motivates this project.",
    
    "Proposed System": "Write comprehensive Proposed System (~1000-1200 words, 6-7 paragraphs): overview, working principle, key innovations, component breakdown, comparison with existing, expected benefits, scalability/extensibility.",
    
    "System Architecture": """Write detailed System Architecture (~1200-1500 words):
- 2 paragraphs of architectural overview
- Describe 6-8 numbered components, each with 2-3 paragraphs (responsibility, technology, I/O, internal logic, interactions, performance)
- 2 paragraphs on cross-cutting concerns and deployment
Reference real classes/modules from the codebase.""",
    
    "Methodology": """Write Methodology (~1200-1500 words) as 10-12 numbered steps. Each step: **bold action title**, technical approach, inputs/outputs, connection to next step, algorithms applied (3-4 sentences each).
Open with introduction paragraph, end with validation strategy paragraph.""",
    
    "Functional Requirements": """Write 7-9 functional requirement sub-sections.
Open with 2-paragraph introduction.
Each sub-section:
- ### heading with descriptive name
- 2-3 sentence intro
- 7-10 specific bullets
Cover: UI/Interaction, Input Handling, Core Processing, Data Storage, Output Generation, Error Handling, Configuration, Reporting, Admin Features.
Total ~1800-2200 words.""",
    
    "Non-Functional Requirements": """Write 8-10 non-functional requirement sub-sections.
Open with 2-paragraph introduction.
Each sub-section:
- ### heading
- 2-3 sentence intro on importance
- 6-8 specific measurable bullets with quantitative targets
Cover: Performance, Reliability, Scalability, Usability, Security, Maintainability, Compatibility, Portability, Auditability, Resource Efficiency.
Total ~1800-2200 words.""",
    
    "Implementation": """Write comprehensive Implementation (~2000-2500 words) with sub-sections:
### Development Environment (~300 words)
### Core Implementation Approach (~500 words)
### Detailed Module Implementation (~800 words) - cover 4-6 key modules
### Technical Decisions (~400 words)
### Integration and Workflow (~400 words)""",
    
    "Testing": """Write comprehensive Testing (~1200-1500 words):
2-paragraph introduction, then sub-sections:
### Functional Testing (~300 words)
### Performance Testing (~250 words)
### Integration Testing (~250 words)
### Validation/Acceptance Testing (~250 words)
### Security Testing (~200 words)
End with 2-paragraph testing outcomes summary.""",
    
    "Limitations": """Write Limitations section (~1200-1500 words):
2-paragraph introduction, then 6-8 sub-sections (150-200 words each) with ### headings:
### Technical Limitations
### Scope Limitations
### Data Limitations
### Scalability Limitations
### Performance Limitations
### Security Limitations
### Usability Limitations
### Deployment Limitations
End with 2-paragraph reflection on impact and how future work addresses them.""",
    
    "Future Scope": """Write 12-15 detailed Future Scope items.
Each: **bold heading**, 2-4 sentences explaining what, why, expected benefit.
Cover diverse categories: algorithm improvements, scalability, new features, integrations, performance, security, UX, deployment, analytics, multi-platform, internationalization.
Total ~1500-1800 words.""",
    
    "References": """Generate 15-20 IEEE-format references relevant to this project.
Format strictly:
[N] A. Author, "Title," *Venue*, vol. X, no. Y, pp. ZZ-ZZ, Year.
Mix 60% papers, 25% books, 15% online resources.
Use real, well-known papers. Cover foundational and recent (2020-2024) works.""",
    
    "UI Requirements": """Write UI Requirements (~1000-1200 words):
### Layout and Navigation (~250 words)
### Key Screens (~400 words) - detail 5-7 screens
### User Interactions (~250 words)
### Accessibility (~200 words)
### Visual Design (~150 words)""",
    
    "Model Architecture": """Write Model Architecture (~1500-1800 words):
### Overview (~200 words)
### Input Pipeline (~250 words)
### Network Architecture (~500 words) - layer-by-layer
### Output and Loss (~300 words)
### Training Procedure (~400 words)
### Evaluation Metrics (~200 words)""",
}


def generate_section(section_name: str, profile: dict, structure: dict) -> dict:
    """Generate a single section using Ollama."""
    instructions = SECTION_PROMPTS.get(
        section_name,
        f"Write a comprehensive {section_name} section (~800 words) for an academic project report."
    )
    instructions = instructions.replace("{tech_stack}", ", ".join(profile.get("tech_stack", [])))

    prompt = f"""You are writing the **{section_name}** section of an academic engineering project report.
Target: 60-70 page comprehensive report. This section should be detailed and substantive.

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

## Writing Guidelines
- Formal academic tone, third person throughout
- Specific to THIS project, avoid generic boilerplate
- Substantive paragraphs (8-12 sentences each)
- Use technical terminology appropriate to the domain
- Reference actual code elements when relevant
- Output in clean markdown format
- No filler phrases like "in this paper" or "we will discuss"

Generate the {section_name} section content now:"""

    try:
        content = _chat(prompt, temperature=0.7, num_predict=6000).strip()
    except Exception as e:
        content = f"_Section generation failed: {e}_"
    
    return {
        "name": section_name,
        "content": content,
        "word_count": len(content.split()),
    }


# ============================================================
# 3. PLANTUML GENERATION
# ============================================================

DIAGRAM_INSTRUCTIONS = {
    "Use Case Diagram": """Create a Use Case Diagram. Example structure:
@startuml
left to right direction
actor User
actor Admin
rectangle System {
  usecase "Login" as UC1
  usecase "View Data" as UC2
  User --> UC1
  Admin --> UC2
}
@enduml""",

    "Class Diagram": """Create a Class Diagram. Example:
@startuml
class User {
  -id: int
  -name: string
  +login()
}
class Admin extends User {
  +manageUsers()
}
User --> Database
@enduml""",

    "Sequence Diagram": """Create a Sequence Diagram. Example:
@startuml
actor User
participant "UI" as UI
participant "Controller" as C
database DB
User -> UI: Submit
UI -> C: Process
C -> DB: Query
DB --> C: Result
C --> UI: Response
UI --> User: Display
@enduml""",

    "Activity Diagram": """Create an Activity Diagram. Example:
@startuml
start
:User submits input;
:Validate input;
if (Valid?) then (yes)
  :Process data;
  :Save result;
else (no)
  :Show error;
endif
:Return response;
stop
@enduml""",

    "Component Diagram": """Create a Component Diagram. Example:
@startuml
package "Frontend" { [Web UI] }
package "Backend" { [API]
[Service] }
database "DB" as DB
[Web UI] --> [API]
[API] --> [Service]
[Service] --> DB
@enduml""",

    "Deployment Diagram": """Create a Deployment Diagram. Example:
@startuml
node "Client" { [Browser] }
node "App Server" { [Web App]
[API] }
database "DB Server" { [Postgres] }
[Browser] --> [Web App]
[Web App] --> [API]
[API] --> [Postgres]
@enduml""",
}


def generate_plantuml(diagram_type: str, structure: dict, profile: dict) -> str:
    """Generate PlantUML code for a diagram via Ollama."""
    classes_summary = json.dumps(structure.get("classes", [])[:10], indent=2)[:800]
    functions_summary = json.dumps(structure.get("functions", [])[:10], indent=2)[:600]
    
    specific = DIAGRAM_INSTRUCTIONS.get(diagram_type, "Create a clean valid PlantUML diagram.")

    prompt = f"""Generate a {diagram_type} as PlantUML code.

## Project
- Title: {profile.get('title')}
- Type: {profile.get('project_type')}
- Tech Stack: {', '.join(profile.get('tech_stack', []))}
- Purpose: {profile.get('inferred_purpose', '')}

## Code Structure
Classes: {classes_summary}
Functions: {functions_summary}

## Format & Example for {diagram_type}
{specific}

## CRITICAL Rules
- Output ONLY valid PlantUML from @startuml to @enduml
- NO markdown fences (no ```)
- NO commentary before or after
- Use real names from the code structure where relevant
- Keep visually clean (5-10 main elements)

Generate the PlantUML code:"""

    try:
        code = _chat(prompt, temperature=0.4, num_predict=1024).strip()
    except Exception as e:
        return f"@startuml\nnote \"Diagram generation failed: {e}\" as N1\n@enduml"

    # Clean markdown fences
    if code.startswith("```"):
        lines = code.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)

    code = code.strip()

    if "@startuml" not in code:
        code = f"@startuml\n{code}\n@enduml"
    elif "@enduml" not in code:
        code = f"{code}\n@enduml"

    return code


# ============================================================
# HELPERS
# ============================================================
def _summarize_structure(structure: dict) -> str:
    """Build compact text summary of code structure."""
    parts = [
        f"Languages: {', '.join(structure.get('languages', ['Unknown']))}",
        f"File count: {structure.get('file_count', 0)}",
        f"Total lines: {structure.get('total_lines', 0)}",
    ]
    if structure.get("classes"):
        parts.append(f"\nClasses ({len(structure['classes'])}):")
        for c in structure["classes"][:12]:
            parts.append(f"  - {c}")
    if structure.get("functions"):
        parts.append(f"\nFunctions ({len(structure['functions'])}):")
        for f in structure["functions"][:15]:
            parts.append(f"  - {f}")
    if structure.get("imports"):
        parts.append(f"\nImports ({len(structure['imports'])}):")
        for i in structure["imports"][:25]:
            parts.append(f"  - {i}")
    if structure.get("file_list"):
        parts.append(f"\nFiles ({len(structure['file_list'])}):")
        for f in structure["file_list"][:25]:
            parts.append(f"  - {f}")
    return "\n".join(parts)