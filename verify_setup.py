"""Verify your environment is correctly configured."""
import sys

print(f"\n🐍 Python version: {sys.version}\n")

# Check Python version
if sys.version_info[:2] != (3, 11):
    print(f"⚠️  WARNING: You're running Python {sys.version_info.major}.{sys.version_info.minor}")
    print("   Recommended: Python 3.11.x")
    print("   This may cause issues on Streamlit Cloud deployment.\n")
else:
    print("✅ Python 3.11 — correct version\n")

packages = [
    ("streamlit", "1.41"),
    ("google.generativeai", None),
    ("tree_sitter", "0.23"),
    ("tree_sitter_languages", "1.10"),
    ("docx", None),
    ("jinja2", "3.1"),
    ("PIL", None),
    ("requests", "2.32"),
    ("pypdf", "5"),
    ("pydantic", "2"),
    ("dotenv", None),
]

print("📦 Package check:\n")
all_ok = True
for pkg_name, expected in packages:
    try:
        module = __import__(pkg_name)
        version = getattr(module, '__version__', 'unknown')
        status = "✅"
        if expected and not str(version).startswith(expected):
            status = "⚠️"
            all_ok = False
        print(f"  {status} {pkg_name:30s} {version}")
    except ImportError:
        print(f"  ❌ {pkg_name:30s} NOT INSTALLED")
        all_ok = False

print()
if all_ok:
    print("🎉 All systems go! Run: streamlit run app.py\n")
else:
    print("⚠️  Some issues. Run: pip install -r requirements.txt\n")
