"""Joins shell.html and sections/*.html into one page that opens straight from disk.
Run: python3 reports/explainer/build.py            -> index.html with every section
     python3 reports/explainer/build.py 3-testing  -> preview-3-testing.html with that section only"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
only = sys.argv[1] if len(sys.argv) > 1 else None
html, scripts = [], []
for f in sorted((HERE / "sections").glob("*.html")):
    if only and f.stem != only:
        continue
    text = f.read_text()
    # A section's <script> moves below the shared helpers, so it can use window.X and window.DATA.
    scripts += re.findall(r"<script>.*?</script>", text, flags=re.S)
    html.append(re.sub(r"<script>.*?</script>", "", text, flags=re.S).strip())
page = (HERE / "shell.html").read_text().replace("<!-- SECTIONS -->", "\n\n".join(html)).replace("<!-- SCRIPTS -->", "\n".join(scripts))
out = HERE / (f"preview-{only}.html" if only else "index.html")
out.write_text(page)
print(out)
