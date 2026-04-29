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
# 2. SECTION GENERATION — LONGER, MORE DETAILED
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
Para 2: Identify the specific problem with concrete examples and statistics where applicable
Para 3: Detail why existing solutions are inadequate (3-5 specific limitations)
Para 4: Articulate the scope and constraints of this project
Para 5: Define the research questions or technical challenges being addressed
Be specific, technical, and grounded in the project's domain.""",

    "Objectives": """Write 8-10 detailed objectives. Each objective:
- Starts with a strong action verb (Design, Develop, Implement, Evaluate, Validate, Optimize, Integrate, Analyze, Demonstrate, Establish)
- Has a clear deliverable
- Spans 2-3 sentences explaining what will be achieved and why
- Uses **bold** for the action verb
Format as a markdown bulleted list with substantive content.""",

    "Literature Review": """Write a comprehensive Literature Review section (~1800-2200 words):

Start with a 1-paragraph introduction to the section.

Then cover 8-10 related works in detail. For EACH work, write a full paragraph (6-8 sentences) covering:
- Authors and year in [Author et al., Year] format
- The technique/approach used
- The dataset, scope, or application context
- Key contribution and findings
- Specific limitations or gaps
- How it relates to or differs from this project

Group works thematically (e.g., "Foundational approaches", "Recent advances", "Domain-specific applications").

End with a comprehensive 2-3 paragraph comparative analysis showing:
- How the proposed work builds upon or differs from prior art
- Gaps in existing literature this project addresses
- Positioning relative to state-of-the-art

Use specific, plausible academic citations throughout.""",

    "Existing System": """Write a detailed Existing System section (~800-1000 words, 5-6 paragraphs):

Para 1: Overview of how this problem is currently addressed in industry/academia
Para 2-3: Detailed description of 2-3 representative existing approaches (architecture, methodology, technology stack)
Para 4: Analysis of limitations — be specific (performance bottlenecks, scalability issues, accuracy gaps, usability problems, security concerns)
Para 5: Discussion of why these limitations matter for stakeholders
Para 6: Summary of the gap that motivates the proposed system

Use specific technical details, mention real systems/tools where applicable, and quantify limitations when possible.""",

    "Proposed System": """Write a comprehensive Proposed System section (~1000-1200 words, 6-7 paragraphs):

Para 1: High-level overview of the proposed solution
Para 2: Working principle and core technical approach
Para 3: Key innovations and novel aspects
Para 4: Detailed component breakdown
Para 5: Comparison with existing systems (specific advantages)
Para 6: Expected benefits (technical, operational, user-facing)
Para 7: Scalability and extensibility considerations

Include specific technologies, algorithms, and architectural decisions. Reference the actual tech stack of this project.""",

    "System Architecture": """Write a detailed System Architecture section (~1200-1500 words):

Start with 2 paragraphs of architectural overview describing:
- The overall pattern (layered, microservices, MVC, pipeline, etc.)
- High-level data flow from input to output

Then describe 6-8 numbered components in depth. For EACH component, write 2-3 paragraphs covering:
- Component name and primary responsibility
- Technology/framework used
- Inputs accepted and outputs produced
- Internal logic and key algorithms
- How it interacts with other components
- Performance and reliability considerations

End with 2 paragraphs on:
- Cross-cutting concerns (logging, error handling, security)
- Deployment topology

Reference actual classes/modules from the codebase.""",

    "Methodology": """Write a comprehensive Methodology section (~1200-1500 words) as 10-12 numbered steps.

For EACH step, write 3-4 sentences covering:
- The action being performed (clear title in **bold**)
- The technical approach used
- Inputs required and outputs produced
- How it connects to subsequent steps
- Any algorithms, formulas, or design patterns applied

Start with an introductory paragraph (4-5 sentences) explaining the overall methodology approach.
End with a paragraph on validation strategy and quality assurance.

Cover the full workflow from data input through processing, storage, computation, and output.""",

    "Functional Requirements": """Write 7-9 detailed functional requirement sub-sections.

For EACH sub-section:
- Use ### heading with descriptive name (e.g. "User Interface and Interaction", "Data Processing Pipeline", "Result Visualization")
- Start with 2-3 sentence introduction explaining the scope
- Provide 7-10 specific bullet points, each 1-2 sentences with concrete details

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

Ground each requirement in the actual code structure and project context. Total ~1500-1800 words.""",

    "Non-Functional Requirements": """Write 8-10 detailed non-functional requirement sub-sections.

For EACH:
- Use ### heading
- Open with 2-3 sentence introduction explaining why this NFR matters for this project
- Provide 6-8 specific, measurable bullet points with quantitative targets where possible

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

Total ~1500-1800 words. Include specific metrics where possible.""",

    "Implementation": """Write a comprehensive Implementation section (~2000-2500 words) covering:

### Development Environment (~300 words)
- Hardware specifications (typical dev machine)
- Operating systems supported
- Software prerequisites with version numbers
- IDE/tooling recommendations
- Setup procedure

### Core Implementation Approach (~500 words)
- Programming paradigm chosen
- Architectural decisions and rationale
- Key design patterns applied
- Dependency management
- Code organization (folder structure)

### Detailed Module Implementation (~800 words)
For 4-6 key modules/components, describe:
- Purpose and responsibility
- Implementation approach
- Key classes/functions (use real names from codebase)
- Algorithms and data structures used
- Integration points

### Technical Decisions (~400 words)
- Library/framework choices and tradeoffs
- Algorithm selection rationale
- Performance optimizations applied
- Security considerations implemented

### Integration and Workflow (~400 words)
- How components connect end-to-end
- Data flow through the system
- Error handling strategy
- Logging and monitoring approach

Use technical detail consistent with the project's actual tech stack and code structure.""",

    "Testing": """Write a comprehensive Testing section (~1200-1500 words) describing the testing approach.

Start with a 2-paragraph introduction to the testing philosophy and methodology.

Then cover 4-5 testing categories with ### sub-headings:

### Functional Testing (~300 words)
- Methodology
- Test case categories (5-7 specific scenarios)
- Tools and frameworks used
- Outcomes and findings

### Performance Testing (~250 words)
- Performance metrics measured
- Test scenarios and load profiles
- Benchmarking methodology
- Results summary

### Integration Testing (~250 words)
- Integration points tested
- Test approach
- Issues discovered and resolved

### Validation/Acceptance Testing (~250 words)
- Validation criteria
- Test methodology
- User acceptance scenarios

### Security Testing (~200 words)
- Security aspects evaluated
- Vulnerabilities checked
- Mitigations applied

End with a 2-paragraph summary of overall testing outcomes and quality assurance.""",

    "Conclusion": """Write a comprehensive Conclusion (~700-900 words, 5-6 paragraphs):

Para 1: Restate the problem and motivation (briefly)
Para 2: Summarize what was built and the technical approach
Para 3: Highlight key contributions and achievements
Para 4: Discuss findings, results, and validation outcomes
Para 5: Reflect on the project's significance and broader implications
Para 6: Final thoughts on the value delivered and lessons learned

Be specific to this project, avoid generic conclusions.""",

    "Future Scope": """Write 12-15 detailed Future Scope items covering enhancements and extensions.

For EACH item:
- Use **bold heading** describing the enhancement
- Write 2-4 sentences explaining:
  * What the enhancement involves technically
  * Why it would be valuable
  * Expected benefit or impact
  * Any prerequisites or dependencies

Cover diverse categories:
- Algorithm/model improvements
- Architecture and scalability enhancements
- New features and functionality
- Integration with external systems
- Performance optimizations
- Security and privacy enhancements
- User experience improvements
- Deployment and operational improvements
- Advanced analytics or reporting
- Multi-platform support
- Internationalization
- Mobile/edge deployment

Total ~1500-1800 words. Be specific to the project's domain.""",

    "References": """Generate 15-20 plausible IEEE-format references relevant to this project's domain, techniques, and tech stack.

Format strictly as IEEE:
[N] A. Author, B. Author, and C. Author, "Title of paper," *Journal/Conference Name*, vol. X, no. Y, pp. ZZ-ZZ, Month Year.

For books:
[N] A. Author, *Book Title*, Edition. Publisher, Year.

For online resources:
[N] A. Author, "Title," Source, Year. [Online]. Available: URL

Mix: 60% journal/conference papers, 25% books, 15% technical reports/online resources.
Use real, well-known papers and authors in the domain when possible.
Cover both foundational works and recent advances (2020-2024).""",

    "UI Requirements": """Write a comprehensive UI Requirements section (~1000-1200 words):

### Layout and Navigation (~250 words)
- Overall layout structure
- Navigation patterns
- Information hierarchy
- Responsive design considerations

### Key Screens (~400 words)
Detail 5-7 main screens with:
- Screen purpose
- Key UI elements
- User interactions
- Data displayed

### User Interactions (~250 words)
- Input methods
- Feedback mechanisms
- Loading states and progress indicators
- Error displays

### Accessibility (~200 words)
- WCAG compliance considerations
- Keyboard navigation
- Screen reader support
- Color contrast and visual design

### Visual Design (~150 words)
- Design system or style guide
- Typography
- Color palette
- Iconography""",

    "Model Architecture": """Write a comprehensive Model Architecture section (~1500-1800 words) covering:

### Overview (~200 words)
- Type of model
- High-level architecture
- Theoretical foundation

### Input Pipeline (~250 words)
- Input format and dimensions
- Preprocessing steps
- Augmentation (if applicable)
- Batching strategy

### Network Architecture (~500 words)
Layer-by-layer description:
- Layer types and configurations
- Activation functions
- Skip connections, attention, etc.
- Parameter counts

### Output and Loss (~300 words)
- Output format
- Loss function with mathematical formulation
- Loss components if multi-task

### Training Procedure (~400 words)
- Optimizer choice and hyperparameters
- Learning rate schedule
- Regularization techniques
- Hardware and training time
- Convergence criteria

### Evaluation Metrics (~200 words)
- Primary metrics with formulas
- Secondary metrics
- Validation strategy""",
}


def generate_section(section_name: str, profile: dict, structure: dict) -> dict:
    """Generate a single section of the document."""
    instructions = SECTION_PROMPTS.get(
        section_name,
        f"Write a comprehensive {section_name} section (~800-1000 words) appropriate for an academic project report."
    )

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
- Substantive paragraphs (8-12 sentences each, not 2-3)
- Use technical terminology appropriate to the domain
- Reference actual code elements (classes, functions, modules) from the structure when relevant
- Include specific examples, metrics, and concrete details
- No filler phrases like "in this paper" or "we will discuss" — be direct
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

    model = _get_model(temperature=0.4, max_tokens=2048)
    response = model.generate_content(prompt)
    code = response.text.strip()

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