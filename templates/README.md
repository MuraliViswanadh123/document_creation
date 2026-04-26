# DOCX Templates

Place IEEE-formatted base DOCX templates here.

## Recommended Setup

1. Download an IEEE conference template from https://www.ieee.org/conferences/publishing/templates.html
2. Save as `ieee_template.docx` in this folder
3. Reference it in `pipeline/docx_builder.py`:

```python
from docx import Document
doc = Document("templates/ieee_template.docx")
```

## What the Template Should Define

- Page margins (typically 1 inch)
- Default font (Times New Roman 10-12pt)
- Heading styles (Heading 1, 2, 3)
- Body paragraph style
- Caption style for figures/tables
- Page numbering
- Header/footer styles

The python-docx library will inherit these styles when adding content.
