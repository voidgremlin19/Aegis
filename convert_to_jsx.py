import re
with open("web_interface/src/legacy_body.html", "r") as f:
    html = f.read()

# Replace class with className
jsx = html.replace('class=', 'className=')
# Replace inline styles with objects (simplistic)
jsx = re.sub(r'style="([^"]*)"', lambda m: 'style={{' + m.group(1).replace(':', ':"').replace(';', '",') + '}}', jsx)
# Replace self-closing tags
jsx = re.sub(r'<(img|br|hr|input|path|circle)([^>]*)(?<!/)>', r'<\1\2 />', jsx)
# Replace SVG attributes
jsx = jsx.replace('stroke-linecap', 'strokeLinecap')
jsx = jsx.replace('stroke-linejoin', 'strokeLinejoin')
jsx = jsx.replace('stroke-width', 'strokeWidth')
jsx = jsx.replace('stroke-dasharray', 'strokeDasharray')
jsx = jsx.replace('stroke-dashoffset', 'strokeDashoffset')
jsx = jsx.replace('fill-opacity', 'fillOpacity')
jsx = jsx.replace('xml:space', 'xmlSpace')
jsx = jsx.replace('transform-origin', 'transformOrigin')
jsx = jsx.replace('viewBox', 'viewBox') # viewBox is correct
jsx = jsx.replace('clip-path', 'clipPath')

with open("web_interface/src/App.jsx", "w") as f:
    f.write('import React, { useEffect } from "react";\n')
    f.write('import "./index.css";\n\n')
    f.write('export default function App() {\n')
    f.write('  useEffect(() => {\n')
    f.write('    // Legacy JS logic to be added here\n')
    f.write('    const script = document.createElement("script");\n')
    f.write('    script.src = "/legacy_script.js";\n')
    f.write('    document.body.appendChild(script);\n')
    f.write('  }, []);\n\n')
    f.write('  return (\n')
    f.write('    <>\n')
    f.write(jsx)
    f.write('    </>\n')
    f.write('  );\n')
    f.write('}\n')

print("Converted to App.jsx")
