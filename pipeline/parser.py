"""
Code parser — extracts universal code structure.

Uses Python's built-in ast module for Python files,
and regex-based extraction for other languages.
This keeps dependencies minimal while supporting common cases.
"""

import ast
import re
from pathlib import Path


# Map file extensions to language names
LANG_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".cpp": "C++",
    ".cc": "C++",
    ".c": "C",
    ".h": "C/C++",
    ".hpp": "C++",
    ".cs": "C#",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".html": "HTML",
    ".css": "CSS",
    ".sql": "SQL",
}

# Files we care about for structure analysis (skip binary/data files)
CODE_EXTENSIONS = set(LANG_MAP.keys())

# Files we read but treat as config
CONFIG_FILES = {
    "requirements.txt", "package.json", "Cargo.toml", "pom.xml",
    "build.gradle", "Gemfile", "composer.json", "go.mod",
    "Dockerfile", "docker-compose.yml", ".env.example", "README.md",
}


def parse_codebase(files: dict) -> dict:
    """
    Parse all source files in a codebase.
    
    Args:
        files: dict mapping filename -> file bytes
    
    Returns:
        Universal code structure dict
    """
    languages = set()
    classes = []
    functions = []
    imports = set()
    file_list = []
    config_contents = {}
    total_lines = 0
    
    for filepath, content_bytes in files.items():
        # Decode bytes to string (skip binary)
        try:
            content = content_bytes.decode("utf-8") if isinstance(content_bytes, bytes) else content_bytes
        except UnicodeDecodeError:
            continue
        
        # Track file
        path = Path(filepath)
        ext = path.suffix.lower()
        filename = path.name
        
        # Capture configs
        if filename in CONFIG_FILES:
            config_contents[filename] = content[:3000]  # truncate large configs
            file_list.append(filepath)
            continue
        
        # Skip non-code files
        if ext not in CODE_EXTENSIONS:
            continue
        
        file_list.append(filepath)
        languages.add(LANG_MAP.get(ext, "Unknown"))
        total_lines += content.count("\n")
        
        # Extract structure
        if ext == ".py":
            _extract_python(content, filepath, classes, functions, imports)
        elif ext in (".js", ".jsx", ".ts", ".tsx"):
            _extract_javascript(content, filepath, classes, functions, imports)
        elif ext == ".java":
            _extract_java(content, filepath, classes, functions, imports)
        else:
            _extract_generic(content, filepath, classes, functions, imports)
    
    return {
        "languages": sorted(languages) or ["Unknown"],
        "file_count": len(file_list),
        "total_lines": total_lines,
        "classes": classes[:50],
        "functions": functions[:80],
        "imports": sorted(imports)[:60],
        "file_list": file_list[:50],
        "config_contents": config_contents,
        "raw_files": files,
    }


def _extract_python(content: str, filepath: str, classes: list, functions: list, imports: set):
    """Extract structure from Python file using ast module."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(f"{node.name} ({Path(filepath).name})")
        elif isinstance(node, ast.FunctionDef):
            functions.append(f"{node.name}() in {Path(filepath).name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])


def _extract_javascript(content: str, filepath: str, classes: list, functions: list, imports: set):
    """Extract structure from JS/TS via regex."""
    # Classes
    for m in re.finditer(r"\bclass\s+(\w+)", content):
        classes.append(f"{m.group(1)} ({Path(filepath).name})")
    # Functions
    for m in re.finditer(r"(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)", content):
        name = m.group(1) or m.group(2)
        if name:
            functions.append(f"{name}() in {Path(filepath).name}")
    # Imports
    for m in re.finditer(r"(?:import.*from\s+['\"]([^'\"]+)['\"]|require\(['\"]([^'\"]+)['\"]\))", content):
        pkg = m.group(1) or m.group(2)
        if pkg and not pkg.startswith("."):
            imports.add(pkg.split("/")[0])


def _extract_java(content: str, filepath: str, classes: list, functions: list, imports: set):
    """Extract structure from Java via regex."""
    for m in re.finditer(r"(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)", content):
        classes.append(f"{m.group(1)} ({Path(filepath).name})")
    for m in re.finditer(r"(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(", content):
        name = m.group(1)
        if name not in ("if", "for", "while", "switch", "return"):
            functions.append(f"{name}() in {Path(filepath).name}")
    for m in re.finditer(r"^import\s+([\w.]+);", content, re.MULTILINE):
        imports.add(m.group(1).split(".")[0])


def _extract_generic(content: str, filepath: str, classes: list, functions: list, imports: set):
    """Generic extractor for languages without a specific handler."""
    # Try common class patterns
    for m in re.finditer(r"\bclass\s+(\w+)", content):
        classes.append(f"{m.group(1)} ({Path(filepath).name})")
    # Try common function patterns
    for m in re.finditer(r"\b(?:def|fn|func|function)\s+(\w+)", content):
        functions.append(f"{m.group(1)}() in {Path(filepath).name}")
