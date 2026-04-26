"""
Pydantic data models.

Defines structured schemas for LLM outputs to ensure validated JSON.
"""

from pydantic import BaseModel, Field


class Paper(BaseModel):
    title: str
    authors: str
    year: int
    venue: str = ""


class ProjectProfile(BaseModel):
    """Output of the project understanding stage."""
    title: str = Field(description="Inferred project title")
    domain: str = Field(description="Application domain (e.g. healthcare, finance)")
    project_type: str = Field(description="web_app | ml_pipeline | cli_tool | mobile_app | library | game | data_pipeline")
    primary_language: str
    tech_stack: list[str]
    detected_techniques: list[str]
    key_papers: list[Paper] = []
    inferred_purpose: str
    architecture_pattern: str = ""
    target_users: str = ""


class Section(BaseModel):
    """A single generated section."""
    name: str
    content: str
    word_count: int


class Diagram(BaseModel):
    """A generated UML diagram."""
    name: str
    plantuml: str
    image_bytes: bytes | None = None
