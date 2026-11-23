"""
Look and feel for ModMyRide AI: "the wrap shop".

A dark showroom banner with a colour-shifting (chameleon wrap) racing stripe, a light primer-grey
workspace, masking-tape step tags, and a live build sheet with a paint swatch that shows the chosen
colour and finish.

Everything here is plain HTML/CSS injected with st.markdown(..., unsafe_allow_html=True).
The base colours also live in .streamlit/config.toml so Streamlit's own widgets match.
"""

from html import escape

INK = "#1A1E29"          # showroom ink: banner, text
PRIMER = "#E3E5E8"       # primer grey: page background
SHEET = "#F7F8FA"        # build sheet / panels
TAPE = "#F3C623"         # masking tape: step tags, labels on photos
BLUE = "#2448E8"         # electric blue: main action
CHAMELEON = "linear-gradient(100deg, #5B2A86 0%, #1F6BFF 34%, #12B5A6 64%, #E9B949 100%)"
# Along the banner stripes: the colour shift falls inside the part of the stripe that is visible
STRIPE = "linear-gradient(180deg, #5B2A86 26%, #1F6BFF 40%, #12B5A6 55%, #E9B949 72%)"

# Swatch colours for the paint options (approximate, for the preview chip only)
PAINT_HEX = {
    "Nardo grey": "#8E9296",
    "Midnight purple": "#3B2A5C",
    "Satin black": "#1E1F22",
    "Pearl white": "#F2F1EC",
    "Racing red": "#C8102E",
    "Electric blue": "#1F6BFF",
    "British racing green": "#0F4D2C",
    "Sunset orange": "#F26B1D",
    "Gold chrome": "#C9A227",
}

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Big+Shoulders+Display:wght@700;800;900&display=swap');

:root {{
  --mmr-ink: {INK};
  --mmr-primer: {PRIMER};
  --mmr-sheet: {SHEET};
  --mmr-tape: {TAPE};
  --mmr-blue: {BLUE};
  --mmr-line: #C9CDD3;
  --mmr-muted: #5A6170;
  --mmr-display: 'Big Shoulders Display', 'Arial Narrow', sans-serif;
  --mmr-body: 'Archivo', system-ui, sans-serif;
}}

/* ---------- page ---------- */
.stApp {{ background: var(--mmr-primer); color: var(--mmr-ink); }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container, [data-testid="stMainBlockContainer"] {{ max-width: 1180px; padding-top: 1.6rem; }}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button {{ font-family: var(--mmr-body); }}
.stApp h1, .stApp h2, .stApp h3 {{ font-family: var(--mmr-display); color: var(--mmr-ink); letter-spacing: 0.01em; }}

/* ---------- showroom banner ---------- */
.mmr-hero {{
  position: relative; overflow: hidden; isolation: isolate;
  background: var(--mmr-ink); color: #F2F3F5;
  border-radius: 20px; padding: 2.6rem 2.4rem 2.3rem; margin-bottom: 1.4rem;
}}
.mmr-hero h1 {{
  font-family: var(--mmr-display); font-weight: 900; color: #F2F3F5;
  font-size: clamp(2.6rem, 8.5vw, 5.6rem); line-height: 0.92; margin: 0 0 0.9rem; padding: 0;
  word-break: normal; overflow-wrap: normal; hyphens: none;
}}
.mmr-hero h1 .mmr-word {{ display: inline-block; white-space: nowrap; }}
[class^="mmr-"] [data-testid="stHeaderActionElements"], .mmr-hero h1 a {{ display: none; }}
.mmr-hero p {{ color: #C3C8D2; font-size: 1.08rem; line-height: 1.5; max-width: 38ch; margin: 0; }}
.mmr-stripes {{
  position: absolute; z-index: -1; top: -60%; right: 6%; height: 220%;
  display: flex; gap: 22px; transform: rotate(22deg);
}}
.mmr-stripes span {{
  display: block; height: 100%; background: {STRIPE}; background-size: 100% 125%;
  animation: mmr-shift 2.4s ease-out 1 both;
}}
.mmr-stripes span:nth-child(1) {{ width: 96px; }}
.mmr-stripes span:nth-child(2) {{ width: 34px; opacity: 0.85; }}
@keyframes mmr-shift {{ from {{ background-position: 0 100%; }} to {{ background-position: 0 0; }} }}
@media (max-width: 700px) {{
  .mmr-hero {{ padding: 2rem 1.4rem 1.8rem; }}
  .mmr-stripes {{ right: -18%; opacity: 0.55; }}
}}
@media (prefers-reduced-motion: reduce) {{ .mmr-stripes span {{ animation: none; }} }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {{
  gap: 0.4rem; height: auto; overflow: visible; padding: 0.3rem 0; margin-bottom: 0.6rem;
}}
.stTabs [data-baseweb="tab"], .stTabs button[role="tab"] {{
  height: auto; min-height: 2.6rem; padding: 0.45rem 1.15rem; border-radius: 999px;
  background: transparent; border: 1.5px solid transparent;
}}
.stTabs button[role="tab"] p, .stTabs [data-baseweb="tab"] p {{
  font-family: var(--mmr-display); font-weight: 800; font-size: 1.15rem; line-height: 1.1;
  margin: 0; color: var(--mmr-muted);
}}
.stTabs button[role="tab"]:hover {{ border-color: var(--mmr-ink); }}
.stTabs button[role="tab"][aria-selected="true"], .stTabs [data-baseweb="tab"][aria-selected="true"] {{
  background: var(--mmr-ink); border-color: var(--mmr-ink);
}}
.stTabs button[role="tab"][aria-selected="true"] p, .stTabs [data-baseweb="tab"][aria-selected="true"] p {{ color: #F2F3F5; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ---------- masking-tape step tags ---------- */
.mmr-step {{ display: flex; align-items: center; gap: 0.75rem; margin: 0.4rem 0 0.25rem; }}
.mmr-step-no {{
  font-family: var(--mmr-display); font-weight: 900; font-size: 1.35rem; line-height: 1;
  background: var(--mmr-tape); color: var(--mmr-ink); padding: 0.3rem 0.7rem 0.25rem;
  transform: rotate(-4deg); border-radius: 2px;
  box-shadow: 0 1px 0 rgba(26, 30, 41, 0.18);
}}
.mmr-step h3 {{ font-family: var(--mmr-display); font-weight: 800; font-size: 2rem; margin: 0; padding: 0; line-height: 1; }}
.mmr-step-hint {{ color: var(--mmr-muted); font-size: 0.95rem; margin: 0 0 0.7rem; }}

/* ---------- widgets ---------- */
[data-testid="stWidgetLabel"] p {{ font-weight: 600; color: var(--mmr-ink); }}
[data-baseweb="select"] > div, .stTextArea textarea, .stTextInput input {{
  background: #FFFFFF; border-radius: 10px; border-color: var(--mmr-line);
}}
[data-testid="stFileUploaderDropzone"] {{
  background: #FFFFFF; border: 2px dashed #9AA1AD; border-radius: 14px; padding: 1.4rem 1rem;
}}
[data-testid="stExpander"] details {{ background: var(--mmr-sheet); border: 1px solid var(--mmr-line); border-radius: 12px; }}
[data-testid="stImage"] img {{ border-radius: 14px; }}

/* buttons */
.stButton > button, .stDownloadButton > button {{
  border-radius: 999px; border: 1.5px solid var(--mmr-ink); background: transparent;
  color: var(--mmr-ink); font-weight: 600;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ background: var(--mmr-ink); color: #F2F3F5; border-color: var(--mmr-ink); }}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  width: 100%; border: none; border-radius: 14px; background: var(--mmr-blue); color: #FFFFFF;
  padding: 0.85rem 1rem; margin-top: 0.3rem;
}}
.stButton > button[kind="primary"] p, [data-testid="stBaseButton-primary"] p {{
  font-family: var(--mmr-display); font-weight: 900; font-size: 1.6rem; letter-spacing: 0.02em; color: #FFFFFF;
}}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{ background: #1A37C4; color: #FFFFFF; }}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{ outline: 3px solid var(--mmr-tape); outline-offset: 2px; }}

/* ---------- live build sheet ---------- */
.mmr-sheet {{
  display: grid; grid-template-columns: 112px 1fr; gap: 1.1rem; align-items: start;
  background: var(--mmr-sheet); border: 1px solid var(--mmr-line); border-radius: 16px;
  padding: 1rem 1.1rem; margin: 0.6rem 0 0.4rem;
}}
.mmr-swatch {{
  width: 112px; height: 112px; border-radius: 50%;
  box-shadow: inset 0 0 0 1px rgba(26, 30, 41, 0.18), 0 6px 14px -8px rgba(26, 30, 41, 0.6);
}}
.mmr-swatch-name {{ text-align: center; font-size: 0.85rem; color: var(--mmr-muted); margin-top: 0.45rem; line-height: 1.3; }}
.mmr-sheet h4 {{ font-family: var(--mmr-display); font-weight: 800; font-size: 1.45rem; margin: 0 0 0.35rem; padding: 0; color: var(--mmr-ink); }}
.mmr-sheet dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 0.9rem; margin: 0; font-size: 0.95rem; }}
.mmr-sheet dt {{ color: var(--mmr-muted); }}
.mmr-sheet dd {{ margin: 0; color: var(--mmr-ink); font-weight: 600; }}
.mmr-sheet dd.mmr-none {{ color: #8A909C; font-weight: 400; }}
@media (max-width: 520px) {{
  .mmr-sheet {{ grid-template-columns: 1fr; }}
  .mmr-swatch {{ margin: 0 auto; }}
}}

/* ---------- results ---------- */
.mmr-result-title {{ font-family: var(--mmr-display); font-weight: 900; font-size: clamp(2rem, 4vw, 2.8rem); margin: 0.4rem 0 0.8rem; line-height: 1; }}
.mmr-tag {{
  display: inline-block; font-family: var(--mmr-display); font-weight: 800; font-size: 1.1rem;
  background: var(--mmr-tape); color: var(--mmr-ink); padding: 0.2rem 0.65rem; margin-bottom: 0.45rem;
  transform: rotate(-2deg); border-radius: 2px;
}}
.mmr-tag.mmr-after {{ background: var(--mmr-blue); color: #FFFFFF; }}
.mmr-section-title {{ font-family: var(--mmr-display); font-weight: 800; font-size: 1.8rem; margin: 1.6rem 0 0.2rem; }}

/* ---------- advisor ---------- */
[data-testid="stChatMessage"] {{ background: var(--mmr-sheet); border: 1px solid var(--mmr-line); border-radius: 14px; }}

/* ---------- footer ---------- */
.mmr-footer {{ color: var(--mmr-muted); font-size: 0.85rem; border-top: 1px solid var(--mmr-line); padding-top: 0.9rem; margin-top: 2rem; max-width: 80ch; }}
"""


def style_tag() -> str:
    return f"<style>{CSS}</style>"


def hero_html() -> str:
    return (
        '<section class="mmr-hero">'
        '<div class="mmr-stripes" aria-hidden="true"><span></span><span></span></div>'
        '<h1><span class="mmr-word">ModMyRide</span> <span class="mmr-word">AI</span></h1>'
        "<p>Try a new wrap, rims and bodykit on a photo of your own car "
        "before you spend a ringgit at the workshop.</p>"
        "</section>"
    )


def step_html(number: int, title: str, hint: str = "") -> str:
    hint_html = f'<p class="mmr-step-hint">{escape(hint)}</p>' if hint else ""
    return (
        f'<div class="mmr-step"><span class="mmr-step-no">{number}</span>'
        f"<h3>{escape(title)}</h3></div>{hint_html}"
    )


def swatch_background(colour: str, finish: str) -> str:
    """CSS background that previews the paint colour with the chosen finish."""
    if colour == "Colour-shifting (chameleon)":
        base = CHAMELEON
    elif colour in PAINT_HEX:
        base = PAINT_HEX[colour]
    else:  # keep original: neutral hatch so it reads as "no change"
        base = "repeating-linear-gradient(135deg, #D5D8DD 0 8px, #EEF0F3 8px 16px)"

    finish_layer = {
        "Gloss": "radial-gradient(circle at 32% 28%, rgba(255,255,255,.75) 0 9%, rgba(255,255,255,0) 32%)",
        "Satin": "radial-gradient(circle at 35% 30%, rgba(255,255,255,.32) 0 14%, rgba(255,255,255,0) 50%)",
        "Matte": "linear-gradient(rgba(0,0,0,.06), rgba(0,0,0,.06))",
        "Chrome": ("linear-gradient(160deg, rgba(255,255,255,.85) 0%, rgba(255,255,255,0) 30%, "
                   "rgba(0,0,0,.35) 52%, rgba(255,255,255,.6) 70%, rgba(0,0,0,.25) 100%)"),
        "Metallic flake": ("radial-gradient(circle at 32% 28%, rgba(255,255,255,.55) 0 7%, rgba(255,255,255,0) 30%), "
                           "radial-gradient(rgba(255,255,255,.55) 1px, transparent 1.6px) 0 0 / 7px 7px"),
    }.get(finish, "")
    if colour == "Colour-shifting (chameleon)" or colour not in PAINT_HEX:
        layers = [finish_layer, base] if finish_layer else [base]
    else:
        layers = [finish_layer, f"linear-gradient({base}, {base})"] if finish_layer else [base]
    return ", ".join(layers)


def _value(text: str, none_label: str = "No change") -> str:
    if not text or text in ("Keep original", "None"):
        return f'<dd class="mmr-none">{escape(none_label)}</dd>'
    return f"<dd>{escape(text)}</dd>"


def build_sheet_html(vehicle: str, choices: dict) -> str:
    """Live summary of what the AI will be asked to do, with a paint swatch."""
    colour, finish = choices.get("colour", ""), choices.get("finish", "")
    if colour and colour != "Keep original":
        paint_name = colour if not finish or finish == "Keep original" else f"{colour}, {finish.lower()}"
    else:
        paint_name = "Original paint" if not finish or finish == "Keep original" else f"Original colour, {finish.lower()}"
    extras = ", ".join(choices.get("extras") or [])
    idea = (choices.get("free_text") or "").strip()
    idea_row = f"<dt>Your idea</dt><dd>{escape(idea[:120])}{'...' if len(idea) > 120 else ''}</dd>" if idea else ""
    car = vehicle if vehicle and not vehicle.startswith("Other") else "Not chosen yet"

    return (
        '<div class="mmr-sheet">'
        f'<div><div class="mmr-swatch" style="background: {swatch_background(colour, finish)};"></div>'
        f'<div class="mmr-swatch-name">{escape(paint_name)}</div></div>'
        "<div><h4>Build sheet</h4><dl>"
        f"<dt>Car</dt><dd>{escape(car)}</dd>"
        f"<dt>Style</dt>{_value(choices.get('preset', ''), 'None')}"
        f"<dt>Rims</dt>{_value(choices.get('rims', ''))}"
        f"<dt>Bodykit</dt>{_value(choices.get('bodykit', ''))}"
        f"<dt>Extras</dt>{_value(extras, 'None')}"
        f"{idea_row}"
        "</dl></div></div>"
    )


def tag_html(text: str, after: bool = False) -> str:
    css_class = "mmr-tag mmr-after" if after else "mmr-tag"
    return f'<span class="{css_class}">{escape(text)}</span>'


def footer_html(text: str) -> str:
    return f'<p class="mmr-footer">{escape(text)}</p>'
