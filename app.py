"""
AutoDocs — AI Documentation Synthesizer
Generates academic reports from source code using Gemini.
"""

import streamlit as st
import time
import zipfile
import io
import os
from datetime import datetime

from pipeline import gemini_client, parser, diagram_generator, docx_builder

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AutoDocs — AI Documentation Synthesizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================
def init_session_state():
    defaults = {
        "generation_complete": False,
        "generated_docx": None,
        "generated_filename": None,
        "project_profile": None,
        "sections": [],
        "diagrams": [],
        "stats": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================
# CONFIG
# ============================================================
def get_gemini_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("GEMINI_API_KEY")


def extract_zip(uploaded_file) -> dict:
    """Extract uploaded ZIP into in-memory file dict."""
    files = {}
    uploaded_file.seek(0)
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            # Skip common junk
            if "__MACOSX" in name or name.startswith("."):
                continue
            try:
                files[name] = z.read(name)
            except Exception:
                continue
    return files


def read_pdf(pdf_file) -> str:
    """Extract text from a PDF file."""
    try:
        from pypdf import PdfReader
        pdf_file.seek(0)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages[:10]:  # first 10 pages only
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"(PDF read failed: {e})"


def plan_document(profile: dict) -> list:
    """Decide which sections to include based on project type."""
    spine = [
        "Abstract",
        "Introduction",
        "Problem Statement",
        "Objectives",
        "Literature Review",
        "Existing System",
        "Proposed System",
        "System Architecture",
        "Methodology",
        "Functional Requirements",
        "Non-Functional Requirements",
        "Implementation",
        "Testing",
        "Conclusion",
        "Future Scope",
        "References",
    ]
    project_type = profile.get("project_type", "").lower()
    if "web" in project_type or "mobile" in project_type or "desktop" in project_type:
        spine.insert(9, "UI Requirements")
    if "ml" in project_type or "ai" in project_type or "vision" in project_type.lower() or "ml" in profile.get("domain", "").lower():
        spine.insert(11, "Model Architecture")
    return spine


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(code_file, abstract: str, paper_file, settings: dict):
    """Orchestrate the full generation pipeline."""
    start_time = time.time()
    
    with st.status("Generating documentation...", expanded=True) as status:
        try:
            # Stage 1: Extract code
            st.write("📂 Extracting source code archive...")
            files = extract_zip(code_file)
            st.write(f"   Found {len(files)} files")
            if not files:
                st.error("No files found in ZIP. Check the archive.")
                return

            # Stage 2: Parse
            st.write("🔍 Parsing code structure...")
            structure = parser.parse_codebase(files)
            st.write(f"   Languages: {', '.join(structure['languages'])}")
            st.write(f"   {len(structure['classes'])} classes, {len(structure['functions'])} functions, {len(structure['imports'])} imports")

            # Stage 3: Read paper
            paper_text = ""
            if paper_file is not None:
                st.write("📑 Reading base paper...")
                paper_text = read_pdf(paper_file)

            # Stage 4: Project understanding
            st.write("🧠 Analyzing project (Gemini)...")
            profile = gemini_client.analyze_project(structure, abstract, paper_text)
            st.session_state.project_profile = profile
            st.write(f"   Title: **{profile.get('title')}**")
            st.write(f"   Domain: {profile.get('domain')}")
            st.write(f"   Type: {profile.get('project_type')}")

            # Stage 5: Plan
            st.write("📋 Planning document structure...")
            section_plan = plan_document(profile)
            st.write(f"   Will generate {len(section_plan)} sections")

            # Stage 6: Sections
            st.write(f"✍️ Generating {len(section_plan)} sections (Gemini)...")
            progress = st.progress(0, text="Starting...")
            sections = []
            for i, section_name in enumerate(section_plan):
                progress.progress(
                    (i + 1) / len(section_plan),
                    text=f"Writing: {section_name} ({i + 1}/{len(section_plan)})"
                )
                try:
                    section = gemini_client.generate_section(section_name, profile, structure)
                    sections.append(section)
                except Exception as e:
                    st.warning(f"   ⚠️ {section_name} failed: {e}")
                    sections.append({"name": section_name, "content": f"_Generation failed: {e}_", "word_count": 0})
            st.session_state.sections = sections

            # Stage 7: Diagrams
            diagrams = []
            if settings.get("include_diagrams", True):
                st.write("📊 Generating UML diagrams...")
                diagrams = diagram_generator.generate_diagrams(structure, profile)
                rendered = sum(1 for d in diagrams if d.get("image_bytes"))
                st.write(f"   {rendered}/{len(diagrams)} diagrams rendered successfully")
                st.session_state.diagrams = diagrams

            # Stage 8: Assemble
            st.write("📄 Assembling DOCX...")
            docx_bytes = docx_builder.assemble_docx(sections, diagrams, profile)
            st.session_state.generated_docx = docx_bytes
            
            # Generate filename from project title
            title_slug = re.sub(r"[^\w\s-]", "", profile.get("title", "documentation")).strip().replace(" ", "_")
            st.session_state.generated_filename = f"{title_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"

            elapsed = time.time() - start_time
            st.session_state.stats = {
                "elapsed_seconds": elapsed,
                "section_count": len(sections),
                "diagram_count": len([d for d in diagrams if d.get("image_bytes")]),
                "word_count": sum(s["word_count"] for s in sections),
            }
            st.session_state.generation_complete = True

            status.update(label=f"✅ Documentation ready in {elapsed:.1f}s", state="complete", expanded=False)

        except Exception as e:
            status.update(label="❌ Generation failed", state="error")
            st.exception(e)


import re  # for filename slug

# ============================================================
# UI: SIDEBAR
# ============================================================
with st.sidebar:
    st.title("📥 Project Inputs")
    
    code_file = st.file_uploader("Source Code (ZIP) *", type=["zip"])
    abstract = st.text_area("Abstract (optional)", height=120, placeholder="Paste abstract if available...")
    paper_file = st.file_uploader("Base Paper (optional)", type=["pdf"])
    
    st.divider()
    st.subheader("⚙️ Settings")
    include_diagrams = st.checkbox("Generate UML diagrams", value=True)
    
    st.divider()
    
    api_key = get_gemini_api_key()
    if api_key:
        st.success("✅ Gemini API key configured")
        gemini_client.configure(api_key)
    else:
        st.error("❌ Gemini API key missing")
        st.caption("Add to .streamlit/secrets.toml or set GEMINI_API_KEY env var")
    
    generate_clicked = st.button(
        "🚀 Generate Documentation",
        type="primary",
        use_container_width=True,
        disabled=(code_file is None or not api_key),
    )

# ============================================================
# UI: MAIN
# ============================================================
st.title("📄 AutoDocs")
st.caption("Generate complete academic documentation from your source code using AI")

if code_file is None and not st.session_state.generation_complete:
    col1, col2, col3 = st.columns(3)
    col1.info("**1. Upload code**\n\nDrop your project ZIP in the sidebar.")
    col2.info("**2. Configure**\n\nOptional: add abstract, base paper.")
    col3.info("**3. Generate**\n\nClick the button. Takes 3-6 minutes.")

elif generate_clicked and code_file is not None and api_key:
    settings = {"include_diagrams": include_diagrams}
    run_pipeline(code_file, abstract, paper_file, settings)
    st.rerun()

# Results
if st.session_state.generation_complete:
    st.success("🎉 Documentation generated successfully!")
    
    stats = st.session_state.stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time", f"{stats['elapsed_seconds']:.1f}s")
    c2.metric("Sections", stats["section_count"])
    c3.metric("Diagrams", stats["diagram_count"])
    c4.metric("Words", f"{stats['word_count']:,}")
    
    st.download_button(
        label="⬇️ Download DOCX Report",
        data=st.session_state.generated_docx,
        file_name=st.session_state.generated_filename,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary",
        use_container_width=True,
    )
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📝 Sections", "📊 Diagrams", "🔍 Profile"])
    with tab1:
        for section in st.session_state.sections:
            with st.expander(f"**{section['name']}** ({section['word_count']} words)"):
                st.markdown(section["content"])
    with tab2:
        if not st.session_state.diagrams:
            st.info("No diagrams generated.")
        else:
            for d in st.session_state.diagrams:
                st.markdown(f"**{d['name']}**")
                if d.get("image_bytes"):
                    st.image(d["image_bytes"])
                with st.expander("PlantUML source"):
                    st.code(d["plantuml"], language="text")
                st.divider()
    with tab3:
        st.json(st.session_state.project_profile)
    
    st.divider()
    if st.button("🔄 Generate New Report", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.divider()
st.caption("AutoDocs · Powered by Gemini 2.5 Flash")
