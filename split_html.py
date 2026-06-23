import re
import os

with open("core_packages/static/index.html", "r") as f:
    content = f.read()

style_match = re.search(r'<style>(.*?)</style>', content, re.DOTALL)
if style_match:
    with open("web_interface/src/index.css", "w") as f:
        f.write(style_match.group(1).strip())

script_match = re.search(r'<script>(.*?)</script>', content, re.DOTALL)
if script_match:
    with open("web_interface/src/legacy_script.js", "w") as f:
        f.write(script_match.group(1).strip())

body_match = re.search(r'<body>(.*?)<script>', content, re.DOTALL)
if body_match:
    body_html = body_match.group(1).strip()
    with open("web_interface/src/legacy_body.html", "w") as f:
        f.write(body_html)

print("Split completed.")
