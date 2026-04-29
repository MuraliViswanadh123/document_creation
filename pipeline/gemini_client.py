"""
Gemini API client — handles all LLM calls.
"""

import json
import google.generativeai as genai


def configure(api_key: str):
    """Configure the Gemini SDK with your API key."""
    genai.configure(api_key=api_key)


def _get_model(temperature: float = 0.7, json_mode: bool = False, max_tokens: int = 8192):
    """Get a configured Gemini 2.5 Flash model instance."""
    config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
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

    model = _get_model(temperature=0.3, json_mode=True, max_tokens=2048)
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
# 2. SECTION GENERATION — LONGER, MORE DETAILED, WITH FEASIBILITY
# ============================================================
SECTION_PROMPTS = {
    "Abstract": """Write a substantive 4-5 paragraph academic abstract (~350-400 words):
Para 1: Problem context and significance (broad → specific)
Para 2: Existing approaches and their limitations
Para 3: Proposed approach, methodology, and key innovations
Para 4: Implementation highlights and technical contributions
Para 5: Expected outcomes, impact, and significance
End with 6-8 keywords formatted in italics.""",

    "Introduction": """Write a comprehensive 8-10 paragraph Introduction (~1500-1800 words):
1. Broad domain context and recent technological trends (2 paragraphs)
2. Specific problem area and its importance (1-2 paragraphs)
3. Detailed analysis of existing approaches (2 paragraphs)
4. Limitations and gaps in current solutions (1 paragraph)
5. Motivation and rationale for this project (1 paragraph)
6. Project overview, objectives, and contributions (1-2 paragraphs)
7. Document structure preview (1 paragraph)
Each paragraph should be substantive (8-12 sentences), with specific technical details.""",

    "Problem Statement": """Write a comprehensive Problem Statement (~600-800 words, 4-5 paragraphs):
Para 1: Set the broader context and why this domain matters
Para 2: Identify the specific problem with concrete examples
Para 3: Detail why existing solutions are inadequate (3-5 specific limitations)
Para 4: Articulate the scope and constraints of this project
Para 5: Define the research questions or technical challenges being addressed""",

    "Objectives": """Write 8-10 detailed objectives. Each objective:
- Starts with a strong action verb (Design, Develop, Implement, Evaluate, etc.)
- Has a clear deliverable
- Spans 2-3 sentences explaining what will be achieved and why
- Uses **bold** for the action verb""",

    "Literature Review": """Write a comprehensive Literature Review section (~1800-2200 words):

Start with a 1-paragraph introduction.

Then cover 8-10 related works in detail. For EACH work, write a full paragraph (6-8 sentences):
- Authors and year in [Author et al., Year] format
- Technique/approach used
- Dataset, scope, or application context
- Key contribution
- Specific limitations
- Relation to this project

Group thematically (e.g., "Foundational approaches", "Recent advances").

End with a 2-3 paragraph comparative analysis.""",

    "Feasibility Study": """Write a comprehensive Feasibility Study section (~1500-1800 words) covering 5 dimensions of feasibility.

Open with a 2-paragraph introduction explaining the importance of feasibility analysis in this project context.

Then provide 5 detailed sub-sections with ### headings:

### Technical Feasibility (~350 words)
Analyze whether the project is technically achievable:
- Availability of required technologies and tools
- Maturity of the chosen tech stack ({tech_stack})
- Compatibility of components
- Required technical expertise and skills
- Hardware/software requirements
- Integration complexity
Conclude with a feasibility verdict.

### Economic Feasibility (~350 words)
Analyze whether the project is financially viable:
- Development costs (tools, licenses, infrastructure)
- Operational costs (hosting, maintenance, third-party services)
- Cost-benefit analysis
- Return on investment potential
- Free/open-source alternatives leveraged
- Long-term financial sustainability
Conclude with a feasibility verdict.

### Operational Feasibility (~300 words)
Analyze whether the system can be operated effectively:
- User acceptance and adoption likelihood
- Training requirements for users
- Integration with existing workflows
- Operational support requirements
- Scalability of operations
Conclude with a feasibility verdict.

### Schedule Feasibility (~250 words)
Analyze whether the project can be completed within timeline:
- Major project phases and milestones
- Estimated time for each phase
- Critical path and dependencies
- Risk factors that could delay
- Resource availability
Conclude with a feasibility verdict.

### Legal and Ethical Feasibility (~250 words)
Analyze legal and ethical considerations:
- Data privacy and protection (GDPR, etc.)
- Licensing of dependencies
- Intellectual property considerations
- Ethical implications of the system
- Compliance requirements
Conclude with a feasibility verdict.

End with a 2-paragraph overall feasibility conclusion summarizing all dimensions.""",

    "Limitations": """Write a comprehensive Limitations section (~1200-1500 words) covering project constraints and limitations.

Open with a 2-paragraph introduction explaining the importance of acknowledging limitations transparently.

Then provide 6-8 detailed sub-sections with ### headings, each 150-200 words:

### Technical Limitations
Inherent technical constraints (performance bounds, accuracy limits, technology gaps).

### Scope Limitations
What this project does NOT cover, intentional exclusions, boundary conditions.

### Data Limitations
Constraints on data availability, quality, volume, or representativeness.

### Scalability Limitations
Bounds on concurrent users, data volume, geographic reach, growth capacity.

### Performance Limitations
Latency, throughput, resource consumption constraints under specific conditions.

### Security Limitations
Known security boundaries, threat models not fully addressed, trust assumptions.

### Usability Limitations
User experience constraints, accessibility gaps, learning curve issues.

### Deployment Limitations
Infrastructure dependencies, platform restrictions, environmental requirements.

End with a 2-paragraph honest reflection on how these limitations affect the project's value and how future work can address them.""",

    "Existing System": """Write a detailed Existing System section (~800-1000 words, 5-6 paragraphs):

Para 1: Overview of how this problem is currently addressed
Para 2-3: Detailed description of 2-3 representative existing approaches
Para 4: Analysis of limitations — be specific (performance, scalability, accuracy, usability, security)
Para 5: Discussion of why these limitations matter for stakeholders
Para 6: Summary of the gap that motivates the proposed system""",

    "Proposed System": """Write a comprehensive Proposed System section (~1000-1200 words, 6-7 paragraphs):

Para 1: High-level overview of the proposed solution
Para 2: Working principle and core technical approach
Para 3: Key innovations and novel aspects
Para 4: Detailed component breakdown
Para 5: Comparison with existing systems (specific advantages)
Para 6: Expected benefits (technical, operational, user-facing)
Para 7: Scalability and extensibility considerations""",

    "System Architecture": """Write a detailed System Architecture section (~1200-1500 words):

Start with 2 paragraphs of architectural overview describing:
- The overall pattern (layered, microservices, MVC, pipeline, etc.)
- High-level data flow

Describe 6-8 numbered components in depth. For EACH, write 2-3 paragraphs covering:
- Component name and primary responsibility
- Technology/framework used
- Inputs and outputs
- Internal logic and key algorithms
- Interactions with other components
- Performance and reliability considerations

End with 2 paragraphs on cross-cutting concerns and deployment topology.

Reference actual classes/modules from the codebase.""",

    "Methodology": """Write a comprehensive Methodology section (~1200-1500 words) as 10-12 numbered steps.

For EACH step (3-4 sentences):
- Action being performed (clear title in **bold**)
- Technical approach used
- Inputs and outputs
- Connection to subsequent steps
- Algorithms/patterns applied

Start with introductory paragraph explaining the overall methodology.
End with paragraph on validation strategy.""",

    "Functional Requirements": """Write 7-9 detailed functional requirement sub-sections.

Open with a 2-paragraph introduction explaining the functional scope of the system.

For EACH sub-section:
- Use ### heading with descriptive name
- Start with 2-3 sentence introduction
- Provide 7-10 specific bullet points with concrete details

Cover (adapting names to project context):
- User Interface and Interaction
- Input Handling and Validation
- Core Processing Logic
- Data Storage and Persistence
- Output Generation and Display
- Error Handling and Recovery
- Configuration and Settings
- Reporting (if applicable)
- Administrative Features

Ground each requirement in the actual code structure. Total ~1800-2200 words.""",

    "Non-Functional Requirements": """Write 8-10 detailed non-functional requirement sub-sections.

Open with a 2-paragraph introduction explaining the importance of NFRs for system quality.

For EACH:
- ### heading
- 2-3 sentence introduction explaining why this NFR matters
- 6-8 specific, measurable bullet points with quantitative targets

Cover:
- Performance (response times, throughput targets)
- Reliability (uptime, error recovery)
- Scalability (concurrent users, data volume)
- Usability (learning curve, accessibility)
- Security (authentication, data protection, attack resistance)
- Maintainability (code quality, testability, documentation)
- Compatibility (platforms, browsers, integrations)
- Portability (deployment options)
- Auditability and Logging
- Resource Efficiency

Total ~1800-2200 words.""",

    "Implementation": """Write a comprehensive Implementation section (~2000-2500 words) covering:

### Development Environment (~300 words)
- Hardware specifications
- Operating systems supported
- Software prerequisites with versions
- IDE/tooling
- Setup procedure

### Core Implementation Approach (~500 words)
- Programming paradigm
- Architectural decisions and rationale
- Design patterns
- Dependency management
- Code organization

### Detailed Module Implementation (~800 words)
For 4-6 key modules:
- Purpose and responsibility
- Implementation approach
- Key classes/functions (use real names)
- Algorithms and data structures
- Integration points

### Technical Decisions (~400 words)
- Library/framework choices and tradeoffs
- Algorithm selection rationale
- Performance optimizations
- Security considerations

### Integration and Workflow (~400 words)
- How components connect
- Data flow
- Error handling
- Logging and monitoring""",

    "Testing": """Write a comprehensive Testing section (~1200-1500 words).

Start with 2-paragraph introduction.

Cover 5 testing categories with ### sub-headings:

### Functional Testing (~300 words)
Methodology, test cases, tools, outcomes.

### Performance Testing (~250 words)
Metrics, scenarios, benchmarking, results.

### Integration Testing (~250 words)
Integration points, approach, issues found.

### Validation/Acceptance Testing (~250 words)
Validation criteria, methodology, scenarios.

### Security Testing (~200 words)
Aspects evaluated, vulnerabilities checked, mitigations.

End with 2-paragraph summary of overall testing outcomes.""",

    "Conclusion": """Write a comprehensive Conclusion (~700-900 words, 5-6 paragraphs):
Para 1: Restate problem and motivation
Para 2: Summarize what was built and the technical approach
Para 3: Highlight key contributions
Para 4: Discuss findings and validation outcomes
Para 5: Reflect on the project's significance
Para 6: Final thoughts and lessons learned""",

    "Future Scope": """Write 12-15 detailed Future Scope items.

For EACH:
- **Bold heading** describing the enhancement
- 2-4 sentences explaining what, why, expected benefit, prerequisites

Cover diverse categories:
- Algorithm/model improvements
- Architecture and scalability
- New features and functionality
- Integration with external systems
- Performance optimizations
- Security enhancements
- UX improvements
- Deployment improvements
- Advanced analytics
- Multi-platform support
- Internationalization
- Mobile/edge deployment

Total ~1500-1800 words.""",

    "References": """Generate 15-20 plausible IEEE-format references.

Format strictly as IEEE:
[N] A. Author, B. Author, and C. Author, "Title of paper," *Journal/Conference*, vol. X, no. Y, pp. ZZ-ZZ, Month Year.

For books:
[N] A. Author, *Book Title*, Edition. Publisher, Year.

For online resources:
[N] A. Author, "Title," Source, Year. [Online]. Available: URL

Mix: 60% papers, 25% books, 15% technical reports/online.
Use real, well-known papers and authors when possible.
Cover both foundational and recent works (2020-2024).""",

    "UI Requirements": """Write a comprehensive UI Requirements section (~1000-1200 words):

### Layout and Navigation (~250 words)
### Key Screens (~400 words) - detail 5-7 main screens
### User Interactions (~250 words)
### Accessibility (~200 words)
### Visual Design (~150 words)""",

    "Model Architecture": """Write a comprehensive Model Architecture section (~1500-1800 words):

### Overview (~200 words)
### Input Pipeline (~250 words)
### Network Architecture (~500 words) - layer-by-layer
### Output and Loss (~300 words)
### Training Procedure (~400 words)
### Evaluation Metrics (~200 words)""",
}


def generate_section(section_name: str, profile: dict, structure: dict) -> dict:
    """Generate a single section of the document."""
    instructions = SECTION_PROMPTS.get(
        section_name,
        f"Write a comprehensive {section_name} section (~800-1000 words) appropriate for an academic project report."
    )
    
    # Substitute project context into prompts that need it (e.g. Feasibility uses tech_stack)
    instructions = instructions.replace("{tech_stack}", ", ".join(profile.get("tech_stack", [])))

    prompt = f"""You are writing the **{section_name}** section of an academic engineering project report.
This is a substantial 60-70 page report. Each section should be detailed and substantive.

## Project Context
- Title: {profile.get('title', 'Untitled')}
- Domain: {profile.get('domain', 'Software Engineering')}
- Type: {profile.get('project_type', 'software')}
- Tech Stack: {', '.join(profile.get('tech_stack', []))}
- Techniques: {', '.join(profile.get('detected_techniques', []))}
- Purpose: {profile.get('inferred_purpose', '')}
- Architecture: {profile.get('architecture_pattern', '')}
- Target Users: {profile.get('target_users', '')}

## Section Instructions
{instructions}

## Writing Guidelines
- Formal academic tone, third person throughout
- Specific to THIS project — use the context above, avoid generic boilerplate
- Substantive paragraphs (8-12 sentences each)
- Use technical terminology appropriate to the domain
- Reference actual code elements when relevant
- Include specific examples, metrics, and concrete details
- No filler phrases — be direct
- Output in clean markdown format

Generate the section content now. Aim for the target word count specified in the section instructions:"""

    model = _get_model(temperature=0.7, max_tokens=8192)
    response = model.generate_content(prompt)
    content = response.text.strip()
    return {
        "name": section_name,
        "content": content,
        "word_count": len(content.split()),
    }


# ============================================================
# 3. PLANTUML GENERATION — IMPROVED PROMPTS
# ============================================================

DIAGRAM_SPECIFIC_INSTRUCTIONS = {
    "Use Case Diagram": """Create a Use Case Diagram with:
- 2-3 actors (e.g., User, Admin, System)
- 6-10 use cases relevant to the project's main features
- Use 'actor', 'usecase', and arrows
Example structure:
@startuml
left to right direction
actor User
actor Admin
rectangle System {
  usecase "Login" as UC1
  usecase "View Data" as UC2
  User --> UC1
  User --> UC2
  Admin --> UC1
}
@enduml""",

    "Class Diagram": """Create a Class Diagram with:
- 5-8 main classes from the codebase
- Their key attributes and methods
- Inheritance, association, composition relationships
Example structure:
@startuml
class User {
  -id: int
  -name: string
  +login()
  +logout()
}
class Admin extends User {
  +manageUsers()
}
User --> Database
@enduml""",

    "Sequence Diagram": """Create a Sequence Diagram showing one key user flow:
- 3-5 participants (user, controllers, services, database)
- 8-12 numbered messages between them
- Use ->, -->, activation boxes
Example structure:
@startuml
actor User
participant "Web UI" as UI
participant "Controller" as C
database DB
User -> UI: Submit Form
UI -> C: Process
C -> DB: Query
DB --> C: Result
C --> UI: Response
UI --> User: Display
@enduml""",

    "Activity Diagram": """Create an Activity Diagram showing the main workflow:
- Start node, end node
- 6-10 activities
- 1-2 decision points (if/else)
- Clear flow from start to finish
Example structure:
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

    "Component Diagram": """Create a Component Diagram showing system modules:
- 5-8 components
- Their interfaces (provided/required)
- Dependencies between components
Example structure:
@startuml
package "Frontend" {
  [Web UI]
}
package "Backend" {
  [API]
  [Service]
}
database "Database" as DB
[Web UI] --> [API]
[API] --> [Service]
[Service] --> DB
@enduml""",

    "Deployment Diagram": """Create a Deployment Diagram showing infrastructure:
- 3-5 nodes (server, database, client)
- Components deployed on each node
- Network connections
Example structure:
@startuml
node "Client Device" {
  [Web Browser]
}
node "Application Server" {
  [Web App]
  [API Server]
}
database "Database Server" {
  [PostgreSQL]
}
[Web Browser] --> [Web App]
[Web App] --> [API Server]
[API Server] --> [PostgreSQL]
@enduml""",
}


def generate_plantuml(diagram_type: str, structure: dict, profile: dict) -> str:
    """Generate PlantUML code for a specific diagram type."""
    classes_summary = json.dumps(structure.get("classes", [])[:15], indent=2)[:1200]
    functions_summary = json.dumps(structure.get("functions", [])[:15], indent=2)[:800]
    
    specific_instructions = DIAGRAM_SPECIFIC_INSTRUCTIONS.get(
        diagram_type,
        "Create a clean, valid PlantUML diagram."
    )

    prompt = f"""Generate a {diagram_type} for this project as PlantUML code.

## Project Context
- Title: {profile.get('title')}
- Type: {profile.get('project_type')}
- Tech Stack: {', '.join(profile.get('tech_stack', []))}
- Purpose: {profile.get('inferred_purpose', '')}

## Code Structure
Classes: {classes_summary}
Functions: {functions_summary}

## Specific Instructions for {diagram_type}
{specific_instructions}

## CRITICAL Requirements
- Output ONLY valid PlantUML code from @startuml to @enduml
- NO markdown fences (no ```)
- NO commentary or explanation before or after
- Use real names from the code structure when relevant
- Keep it visually clean and uncluttered (5-10 main elements)
- Make sure the syntax is valid PlantUML

Output the PlantUML code now:"""

    model = _get_model(temperature=0.4, max_tokens=2048)
    response = model.generate_content(prompt)
    code = response.text.strip()

    # Clean up markdown fences if present
    if code.startswith("```"):
        lines = code.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)
    
    code = code.strip()

    # Ensure proper start/end tags
    if "@startuml" not in code:
        code = f"@startuml\n{code}\n@enduml"
    elif "@enduml" not in code:
        code = f"{code}\n@enduml"
    
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