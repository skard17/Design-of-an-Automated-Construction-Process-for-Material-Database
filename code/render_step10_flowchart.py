from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
SVG_PATH = ROOT / "step10_prompt_quality_code_agent_flow.svg"
HTML_PATH = ROOT / "step10_prompt_quality_code_agent_flow.html"


def box(x, y, w, h, title, lines=None, fill="#f8fafc", stroke="#334155"):
    lines = lines or []
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>',
        f'<text x="{x + w / 2}" y="{y + 24}" text-anchor="middle" class="title">{escape(title)}</text>',
    ]
    for idx, line in enumerate(lines):
        parts.append(
            f'<text x="{x + w / 2}" y="{y + 48 + idx * 17}" text-anchor="middle" class="body">{escape(line)}</text>'
        )
    return "\n".join(parts)


def diamond(cx, cy, w, h, title, subtitle=""):
    points = [
        (cx, cy - h / 2),
        (cx + w / 2, cy),
        (cx, cy + h / 2),
        (cx - w / 2, cy),
    ]
    point_text = " ".join(f"{x},{y}" for x, y in points)
    parts = [
        f'<polygon points="{point_text}" fill="#fff7ed" stroke="#c2410c" stroke-width="1.4"/>',
        f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" class="title">{escape(title)}</text>',
    ]
    if subtitle:
        parts.append(f'<text x="{cx}" y="{cy + 16}" text-anchor="middle" class="body">{escape(subtitle)}</text>')
    return "\n".join(parts)


def arrow(x1, y1, x2, y2, label=None):
    parts = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#475569" stroke-width="1.35" marker-end="url(#arrow)"/>'
    ]
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        parts.append(f'<text x="{mid_x}" y="{mid_y - 7}" text-anchor="middle" class="edge">{escape(label)}</text>')
    return "\n".join(parts)


def header():
    return """
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#475569"/>
  </marker>
  <style>
    .heading { font: 800 25px Arial, sans-serif; fill: #0f172a; }
    .sub { font: 14px Arial, sans-serif; fill: #475569; }
    .title { font: 700 14px Arial, sans-serif; fill: #0f172a; }
    .body { font: 12.5px Arial, sans-serif; fill: #334155; }
    .edge { font: 12px Arial, sans-serif; fill: #64748b; }
    .lane { font: 700 13px Arial, sans-serif; fill: #475569; }
  </style>
</defs>
<rect width="100%" height="100%" fill="#f1f5f9"/>
<text x="680" y="34" text-anchor="middle" class="heading">Step 10 Prompt Quality And Code Agent Flow</text>
<text x="680" y="58" text-anchor="middle" class="sub">Local format checks + local quality lint + optional LLM quality review + optional code generation</text>
"""


def build_svg():
    width = 1360
    height = 860
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        header(),
    ]

    # Lanes.
    lanes = [
        (40, 90, 270, 690, "Input"),
        (335, 90, 300, 690, "Local judgement"),
        (665, 90, 300, 690, "LLM quality review"),
        (995, 90, 300, 690, "Code generation"),
    ]
    for x, y, w, h, title in lanes:
        items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#ffffff" stroke="#cbd5e1"/>')
        items.append(f'<text x="{x + 16}" y="{y + 24}" class="lane">{escape(title)}</text>')

    # Input lane.
    items.append(box(75, 135, 200, 86, "Step 9 output", ["skyrmion_prompt_", "generation_output.json"], "#e0f2fe", "#0284c7"))
    items.append(box(75, 285, 200, 86, "Load package", ["module_outputs", "shared_prompt_context"], "#f8fafc", "#334155"))
    items.append(arrow(175, 221, 175, 285))

    # Local judgement lane.
    items.append(box(380, 130, 210, 112, "Format checks", ["status success", "validation empty", "modules present", "104 / 104 fields covered"], "#dcfce7", "#16a34a"))
    items.append(box(380, 300, 210, 132, "Quality lint", ["JSON contract", "evidence grounding", "anti-hallucination", "section ownership", "figure / repair rules"], "#dcfce7", "#16a34a"))
    items.append(box(380, 500, 210, 96, "Base judgement", ["format_checks", "quality_checks", "recommendations"], "#fef9c3", "#ca8a04"))
    items.append(arrow(275, 328, 380, 186, "package"))
    items.append(arrow(275, 328, 380, 366, "package"))
    items.append(arrow(485, 242, 485, 300))
    items.append(arrow(485, 432, 485, 500))

    # LLM review lane.
    items.append(diamond(815, 178, 170, 92, "Run LLM review?", "--run-quality-review"))
    items.append(box(710, 300, 210, 118, "Quality review agent", ["LiteLLM chat API", "scores dimensions", "findings + advice"], "#ede9fe", "#7c3aed"))
    items.append(diamond(815, 505, 170, 92, "API succeeds?"))
    items.append(box(690, 645, 250, 92, "Merged judgement", ["llm_quality_review", "or review error", "prompt_quality_judgement.json"], "#fef9c3", "#ca8a04"))
    items.append(arrow(590, 548, 730, 178, "base judgement"))
    items.append(arrow(815, 224, 815, 300, "yes"))
    items.append(arrow(815, 418, 815, 459))
    items.append(arrow(815, 551, 815, 645, "yes or no"))
    items.append(box(690, 110, 250, 52, "Skip LLM review", ["write local judgement only"], "#f8fafc", "#64748b"))
    items.append(arrow(730, 178, 690, 136, "no"))

    # Code generation lane.
    items.append(diamond(1145, 178, 170, 92, "Judge only?", "--judge-only"))
    items.append(box(1040, 300, 210, 118, "Code generation agent", ["prompt package", "judgement", "quality guidance"], "#ede9fe", "#7c3aed"))
    items.append(box(1040, 500, 210, 86, "Code agent output", ["code_generation_", "agent_output.json"], "#e0f2fe", "#0284c7"))
    items.append(diamond(1145, 675, 180, 90, "Write files?", "--write-generated-files"))
    items.append(box(1015, 780, 260, 54, "generated_code/", ["does not overwrite original files"], "#dcfce7", "#16a34a"))
    items.append(arrow(940, 691, 1060, 178, "judgement"))
    items.append(arrow(1145, 224, 1145, 300, "no"))
    items.append(arrow(1145, 418, 1145, 500))
    items.append(arrow(1145, 586, 1145, 630))
    items.append(arrow(1145, 720, 1145, 780, "yes"))
    items.append(box(1040, 110, 210, 52, "Stop after judgement", ["no code agent call"], "#f8fafc", "#64748b"))
    items.append(arrow(1100, 178, 1040, 136, "yes"))

    # Status note.
    items.append(box(60, 680, 560, 92, "Current API status", ["deepseek-v4 maps to deepseek-v4-pro: server 500", "kimi-k2.6: minimal chat test passed"], "#fee2e2", "#dc2626"))

    items.append("</svg>")
    return "\n".join(items)


def build_html(svg_text):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Step 10 Prompt Quality And Code Agent Flow</title>
  <style>
    body {{ margin: 0; padding: 24px; background: #f8fafc; color: #0f172a; font-family: Arial, sans-serif; }}
    main {{ max-width: 1420px; margin: 0 auto; background: white; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; }}
    svg {{ width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
<main>
{svg_text}
</main>
</body>
</html>
"""


def main():
    svg_text = build_svg()
    SVG_PATH.write_text(svg_text, encoding="utf-8")
    HTML_PATH.write_text(build_html(svg_text), encoding="utf-8")
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
