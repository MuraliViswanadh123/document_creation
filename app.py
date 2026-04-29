"""
AutoDocs — AI Documentation Synthesizer
Hybrid architecture: Ollama (local) for bulk work + Gemini for premium sections.
"""

import streamlit as st
import time
import zipfile
import io
import os
import re
from datetime import datetime

from pipeline import gemini_client, ollama_client, parser, diagram_generator, docx_builder

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


# ============================================================
# HYBRID ROUTING — which sections go to which LLM
# ============================================================

# These "premium" sections benefit from Gemini's stronger prose
GEMINI_SECTIONS = {
    "Abstract",       # Most visible, sets tone
    "Introduction",   # Long, narrative-heavy
    "Conclusion",     # High-impact closing
}

# Everything else → Ollama (saves Gemini quota)


def route_section(section_name: str, profile: dict, structure: dict, use_gemini: bool, use_ollama: bool) -> dict:
    """
    Route a section to the right LLM based on hybrid strategy.
    Falls back gracefully if one is unavailable.
    """
    is_premium = section_name in GEMINI_SECTIONS
    
    # Strategy: premium sections use Gemini if available, others use Ollama
    if is_premium and use_gemini:
        try:
            return gemini_client.generate_section(section_name, profile, structure)
        except Exception as e:
            st.warning(f"   ⚠️ Gemini failed for {section_name}, trying Ollama: {e}")
            if use_ollama:
                return ollama_client.generate_section(section_name, profile, structure)
            return {"name": section_name, "content": f"_Failed: {e}_", "word_count": 0}
    
    if use_ollama:
        try:
            return ollama_client.generate_section(section_name, profile, structure)
        except Exception as e:
            st.warning(f"   ⚠️ Ollama failed for {section_name}, trying Gemini: {e}")
            if use_gemini:
                try:
                    return gemini_client.generate_section(section_name, profile, structure)
                except Exception as e2:
                    return {"name": section_name, "content": f"_Both failed: {e2}_", "word_count": 0}
            return {"name": section_name, "content": f"_Failed: {e}_", "word_count": 0}
    
    # No LLM available
    return {"name": section_name, "content": "_No LLM available_", "word_count": 0}


# ============================================================
# FILE READING HELPERS
# ============================================================
def extract_zip(uploaded_file) -> dict:
    files = {}
    uploaded_file.seek(0)
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            if "__MACOSX" in name or name.startswith("."):
                continue
            try:
                files[name] = z.read(name)
            except Exception:
                continue
    return files


def read_pdf(pdf_file) -> str:
    try:
        from pypdf import PdfReader
        pdf_file.seek(0)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages[:20]:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"(PDF read failed: {e})"


def read_docx(docx_file) -> str:
    try:
        from docx import Document
        docx_file.seek(0)
        doc = Document(docx_file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"(DOCX read failed: {e})"


def read_text_file(txt_file) -> str:
    try:
        txt_file.seek(0)
        return txt_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"(Text file read failed: {e})"


def read_abstract_file(abstract_file) -> str:
    if abstract_file is None:
        return ""
    name = abstract_file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(abstract_file)
    elif name.endswith(".docx"):
        return read_docx(abstract_file)
    elif name.endswith((".txt", ".md")):
        return read_text_file(abstract_file)
    return f"(Unsupported file type: {abstract_file.name})"


def plan_document(profile: dict) -> list:
    spine = [
        "Abstract",
        "Introduction",
        "Problem Statement",
        "Objectives",
        "Literature Review",
        "Feasibility Study",
        "Existing System",
        "Proposed System",
        "System Architecture",
        "Methodology",
        "Functional Requirements",
        "Non-Functional Requirements",
        "Implementation",
        "Testing",
        "Limitations",
        "Conclusion",
        "Future Scope",
        "References",
    ]
    project_type = profile.get("project_type", "").lower()
    domain = profile.get("domain", "").lower()
    if "web" in project_type or "mobile" in project_type or "desktop" in project_type:
        spine.insert(11, "UI Requirements")
    if "ml" in project_type or "ai" in project_type or "vision" in domain or "ml" in domain:
        spine.insert(13, "Model Architecture")
    return spine


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(code_file, abstract_text: str, paper_file, settings: dict, use_gemini: bool, use_ollama: bool):
    start_time = time.time()
    
    with st.status("Generating documentation...", expanded=True) as status:
        try:
            # Show LLM strategy
            strategy = []
            if use_ollama:
                strategy.append("Ollama (local)")
            if use_gemini:
                strategy.append("Gemini (cloud)")
            st.write(f"🤖 LLM strategy: **{' + '.join(strategy)}**")
            if use_ollama and use_gemini:
                st.write(f"   Premium sections (Abstract, Introduction, Conclusion) → Gemini")
                st.write(f"   All other sections → Ollama")

            st.write("📂 Extracting source code archive...")
            files = extract_zip(code_file)
            st.write(f"   Found {len(files)} files")
            if not files:
                st.error("No files found in ZIP. Check the archive.")
                return

            st.write("🔍 Parsing code structure...")
            structure = parser.parse_codebase(files)
            st.write(f"   Languages: {', '.join(structure['languages'])}")
            st.write(f"   {len(structure['classes'])} classes, {len(structure['functions'])} functions, {len(structure['imports'])} imports")

            paper_text = ""
            if paper_file is not None:
                st.write("📑 Reading base paper...")
                paper_text = read_pdf(paper_file)

            if abstract_text:
                st.write(f"📝 Using abstract ({len(abstract_text)} chars)")

            # Project analysis — prefer Ollama to save Gemini quota
            st.write("🧠 Analyzing project...")
            if use_ollama:
                profile = ollama_client.analyze_project(structure, abstract_text, paper_text)
                st.write("   (via Ollama — saves Gemini quota)")
            else:
                profile = gemini_client.analyze_project(structure, abstract_text, paper_text)
                st.write("   (via Gemini)")
            
            st.session_state.project_profile = profile
            st.write(f"   Title: **{profile.get('title')}**")
            st.write(f"   Domain: {profile.get('domain')}")
            st.write(f"   Type: {profile.get('project_type')}")

            st.write("📋 Planning document structure...")
            section_plan = plan_document(profile)
            st.write(f"   Will generate {len(section_plan)} sections")

            # Show projected LLM usage
            if use_ollama and use_gemini:
                gemini_count = sum(1 for s in section_plan if s in GEMINI_SECTIONS)
                ollama_count = len(section_plan) - gemini_count
                st.write(f"   📊 Routing: {ollama_count} sections → Ollama, {gemini_count} sections → Gemini")

            st.write(f"✍️ Generating {len(section_plan)} sections...")
            progress = st.progress(0, text="Starting...")
            sections = []
            for i, section_name in enumerate(section_plan):
                llm_label = "Gemini" if (section_name in GEMINI_SECTIONS and use_gemini) else "Ollama"
                progress.progress(
                    (i + 1) / len(section_plan),
                    text=f"Writing: {section_name} via {llm_label} ({i + 1}/{len(section_plan)})"
                )
                section = route_section(section_name, profile, structure, use_gemini, use_ollama)
                sections.append(section)
                st.write(f"   ✓ {section_name} ({llm_label}): {section['word_count']} words")
            
            st.session_state.sections = sections

            # Diagrams — always use Ollama for PlantUML (structured DSL, saves Gemini quota)
            diagrams = []
            if settings.get("include_diagrams", True):
                st.write("📊 Generating UML diagrams (via Ollama)...")
                # Force Ollama for diagrams in hybrid mode
                if use_ollama:
                    diagrams = diagram_generator.generate_diagrams(structure, profile, llm="ollama")
                elif use_gemini:
                    diagrams = diagram_generator.generate_diagrams(structure, profile, llm="gemini")
                
                rendered = sum(1 for d in diagrams if d.get("image_bytes"))
                for d in diagrams:
                    if d.get("image_bytes"):
                        st.write(f"   ✓ {d['name']}: {d.get('status', 'rendered')}")
                    else:
                        st.write(f"   ✗ {d['name']}: {d.get('status', 'failed')}")
                st.write(f"   **Total: {rendered}/{len(diagrams)} diagrams rendered**")
                st.session_state.diagrams = diagrams

            st.write("📄 Assembling DOCX...")
            docx_bytes = docx_builder.assemble_docx(sections, diagrams, profile)
            st.session_state.generated_docx = docx_bytes
            
            title_slug = re.sub(r"[^\w\s-]", "", profile.get("title", "documentation")).strip().replace(" ", "_")
            st.session_state.generated_filename = f"{title_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"

            elapsed = time.time() - start_time
            total_words = sum(s["word_count"] for s in sections)
            st.session_state.stats = {
                "elapsed_seconds": elapsed,
                "section_count": len(sections),
                "diagram_count": len([d for d in diagrams if d.get("image_bytes")]),
                "word_count": total_words,
                "estimated_pages": max(1, total_words // 280),
            }
            st.session_state.generation_complete = True

            status.update(
                label=f"✅ Documentation ready in {elapsed:.0f}s — ~{st.session_state.stats['estimated_pages']} pages",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            status.update(label="❌ Generation failed", state="error")
            st.exception(e)


# ============================================================
# UI: SIDEBAR
# ============================================================
with st.sidebar:
    st.title("📥 Project Inputs")
    
    code_file = st.file_uploader(
        "Source Code (ZIP) *",
        type=["zip"],
        help="Upload your project as a ZIP archive (required)",
    )
    
    st.divider()
    st.subheader("📝 Abstract")
    st.caption("Provide as text OR upload a file (file takes priority)")
    
    abstract_text_input = st.text_area(
        "Type/paste abstract",
        height=120,
        placeholder="Paste abstract here, or use the file upload below...",
    )
    
    abstract_file = st.file_uploader(
        "Or upload abstract file",
        type=["txt", "md", "docx", "pdf"],
        help="Supported formats: .txt, .md, .docx, .pdf",
    )
    
    final_abstract = ""
    if abstract_file is not None:
        final_abstract = read_abstract_file(abstract_file)
        st.caption(f"📄 Using uploaded file: **{abstract_file.name}**")
    elif abstract_text_input.strip():
        final_abstract = abstract_text_input.strip()
        st.caption("✏️ Using text input")
    
    st.divider()
    st.subheader("📑 Base Paper (optional)")
    paper_file = st.file_uploader(
        "Upload reference paper",
        type=["pdf"],
        help="Used to enrich literature review",
    )
    
    st.divider()
    st.subheader("⚙️ Settings")
    include_diagrams = st.checkbox("Generate UML diagrams", value=True)
    
    st.divider()
    st.subheader("🤖 LLM Status")
    
    # Check both LLMs
    api_key = get_gemini_api_key()
    gemini_available = bool(api_key)
    if gemini_available:
        gemini_client.configure(api_key)
        st.success("✅ Gemini configured")
    else:
        st.error("❌ Gemini API key missing")
    
    ollama_available = ollama_client.is_available()
    if ollama_available:
        st.success("✅ Ollama running locally")
    else:
        st.warning("⚠️ Ollama not detected — make sure it's running with qwen2.5-coder:7b")
    
    # Determine effective strategy
    if ollama_available and gemini_available:
        st.info("🎯 **Hybrid mode** — best of both")
    elif ollama_available:
        st.info("🏠 **Ollama-only mode** — fully local")
    elif gemini_available:
        st.info("☁️ **Gemini-only mode** — quota-limited")
    else:
        st.error("❌ No LLM available!")
    
    st.divider()
    
    can_generate = code_file is not None and (gemini_available or ollama_available)
    generate_clicked = st.button(
        "🚀 Generate Documentation",
        type="primary",
        use_container_width=True,
        disabled=not can_generate,
    )

# ============================================================
# UI: MAIN
# ============================================================
st.title("📄 AutoDocs")
st.caption("Generate complete 60-70 page academic documentation from your source code using AI")

if code_file is None and not st.session_state.generation_complete:
    col1, col2, col3 = st.columns(3)
    col1.info("**1. Upload code**\n\nDrop your project ZIP in the sidebar.")
    col2.info("**2. Add abstract**\n\nType it OR upload as .txt/.docx/.pdf (optional).")
    col3.info("**3. Generate**\n\nClick the button. Targets 60-70 pages output.")
    
    st.divider()
    st.subheader("🔧 Architecture")
    st.markdown("""
    This tool uses a **hybrid LLM architecture**:
    - **Ollama (local)**: Handles code analysis, most sections, and diagrams — fast and free, no quota limits
    - **Gemini (cloud)**: Handles premium sections (Abstract, Introduction, Conclusion) for highest prose quality
    
    Result: ~3-4 Gemini API calls per document (vs 25-30 in cloud-only mode), well under free tier limits.
    """)

elif generate_clicked and code_file is not None:
    settings = {"include_diagrams": include_diagrams}
    run_pipeline(code_file, final_abstract, paper_file, settings, gemini_available, ollama_available)
    st.rerun()

if st.session_state.generation_complete:
    st.success("🎉 Documentation generated successfully!")
    
    stats = st.session_state.stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Time", f"{stats['elapsed_seconds']:.0f}s")
    c2.metric("Pages (est.)", stats.get("estimated_pages", "—"))
    c3.metric("Sections", stats["section_count"])
    c4.metric("Diagrams", stats["diagram_count"])
    c5.metric("Words", f"{stats['word_count']:,}")
    
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
                if d.get("status"):
                    st.caption(f"Status: {d['status']}")
                if d.get("image_bytes"):
                    st.image(d["image_bytes"])
                else:
                    st.warning("Diagram could not be rendered. PlantUML source is below.")
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
st.caption("AutoDocs · Hybrid Architecture · Ollama + Gemini")