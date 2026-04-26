# Prompt Templates

Jinja2 templates for each section of the generated document.

Each template receives:
- `profile`: ProjectProfile dict (domain, technique, tech_stack, etc.)
- `structure`: code structure (classes, functions, imports)
- `previous_sections`: dict of already-generated sections (for consistency)

## Required Templates

- `abstract.j2` ✅ (example provided)
- `introduction.j2`
- `problem_statement.j2`
- `objectives.j2`
- `literature_review.j2`
- `existing_system.j2`
- `proposed_system.j2`
- `system_architecture.j2`
- `methodology.j2`
- `functional_requirements.j2`
- `non_functional_requirements.j2`
- `implementation.j2`
- `testing.j2`
- `conclusion.j2`
- `future_scope.j2`
- `references.j2`

## Tips

1. Use **flood.docx as a few-shot example** in your prompts to teach Gemini the expected style and depth
2. Keep instructions specific (word count, structure, tone)
3. Always pass `profile` so output stays consistent with project context
4. Return only the section content — no headers, no commentary
