"""
ModMyRide AI - preview custom car exteriors with AI (Streamlit version).

Flow:  upload photo -> user picks body type, car model and style
       -> Gemini writes a detailed edit prompt -> image model (FLUX.2 on Cloudflare) redesigns the photo
       -> before/after view, design history and download.
Plus a Style Advisor chat powered by Gemini.

Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub, then deploy on Streamlit Community Cloud. See README.md.
"""

import hashlib
import io
import os
from datetime import datetime

import streamlit as st
from PIL import Image

import ai

SECRET_NAMES = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "TEXT_MODEL", "IMAGE_MODEL", "IMAGE_BACKEND",
                "HF_TOKEN", "HF_IMAGE_MODEL", "CF_ACCOUNT_ID", "CF_API_TOKEN", "CF_IMAGE_MODEL"]

# ---------------------------------------------------------------------------
# Options shown in the app
# ---------------------------------------------------------------------------
CAR_MODELS = [
    "Perodua Myvi",
    "Perodua Axia",
    "Perodua Bezza",
    "Perodua Ativa",
    "Proton Saga",
    "Proton Persona",
    "Proton X50",
    "Proton X70",
    "Honda City",
    "Honda Civic",
    "Toyota Vios",
    "Other / not sure",
]

BODY_TYPES = ["Hatchback", "Sedan", "SUV", "MPV", "Not sure"]

PRESETS = ["None", "JDM street racer", "Clean luxury (OEM+)", "Track-day racing livery",
           "Off-road adventure", "Stealth all-black", "Retro classic"]

COLOURS = ["Keep original", "Nardo grey", "Midnight purple", "Satin black", "Pearl white", "Racing red",
           "Electric blue", "British racing green", "Sunset orange", "Gold chrome",
           "Colour-shifting (chameleon)"]

FINISHES = ["Keep original", "Gloss", "Matte", "Satin", "Chrome", "Metallic flake"]

RIMS = ["Keep original", "Black multi-spoke", "Gold mesh (BBS-style)", "Bronze six-spoke",
        "Silver deep-dish", "Gunmetal five-spoke", "White racing wheels"]

BODYKITS = ["Keep original", "Subtle front lip and side skirts", "Full sporty bodykit with rear diffuser",
            "Wide-body kit", "Rally style with mud flaps"]

EXTRAS = ["Racing stripes", "Ducktail spoiler", "GT wing", "Tinted windows", "Lowered suspension",
          "Black roof", "Side decals"]

# Short styling ideas per car model (ideas to try, not facts about the car)
SUGGESTIONS = {
    "Perodua Myvi": "satin or Nardo grey wrap, black multi-spoke rims, subtle lip kit, black roof.",
    "Perodua Axia": "bright colour wrap (orange, electric blue), white rims, racing stripes for a fun look.",
    "Perodua Bezza": "clean OEM+ look with pearl white, gunmetal rims and tinted windows.",
    "Perodua Ativa": "off-road style, matte finish, bronze rims and a black roof.",
    "Proton Saga": "retro classic look, deep-dish silver rims, or a sleeper all-black style.",
    "Proton Persona": "clean luxury with metallic finish and five-spoke gunmetal rims.",
    "Proton X50": "stealth all-black, bronze rims, or a two-tone black roof.",
    "Proton X70": "luxury satin wrap, large black rims and chrome delete.",
    "Honda City": "JDM look with front lip, ducktail spoiler and gold mesh rims.",
    "Honda Civic": "Type-R inspired racing red, GT wing, black multi-spoke rims.",
    "Toyota Vios": "track-day livery, side decals and white racing wheels.",
}
BODY_SUGGESTIONS = {
    "Hatchback": "hot-hatch look: Nardo grey or satin wrap, black multi-spoke rims, front lip, black roof.",
    "Sedan": "JDM street racer: front lip, side skirts, ducktail spoiler and gold mesh rims, or clean luxury OEM+.",
    "SUV": "off-road adventure: matte finish, bronze rims, black roof and rally mud flaps, or stealth all-black.",
    "MPV": "clean luxury: pearl white, gunmetal five-spoke rims, tinted windows and a subtle lip kit.",
}
DEFAULT_SUGGESTION = "try a preset style first, then fine-tune colour, rims and bodykit."

ADVISOR_EXAMPLES = [
    "What colour and rims suit my car for a sporty look?",
    "Wrap or respray: which is better for a student budget?",
    "Apa gaya JDM yang sesuai untuk Myvi?",
    "Suggest three looks that are subtle but stylish.",
]

PHOTO_TIPS = """**Photo tips for best results**
- Daylight, with the whole car in the frame
- Side view or front three-quarter view works best
- Avoid heavy shadows and cropped bumpers"""

DISCLAIMER = (
    "Designs are AI previews for inspiration only. Results may not match real parts exactly. "
    "Check the latest JPJ and insurance requirements before modifying your car."
)


# ---------------------------------------------------------------------------
# Helpers (no Streamlit calls, easy to test)
# ---------------------------------------------------------------------------
def suggestion_for(car_model: str, body_type: str = "Not sure") -> str:
    """Car-model ideas if a specific model is picked, otherwise ideas for the body type."""
    if car_model in SUGGESTIONS:
        idea = SUGGESTIONS[car_model]
    else:
        idea = BODY_SUGGESTIONS.get(body_type, DEFAULT_SUGGESTION)
    return f"**Style ideas:** {idea}"


def vehicle_description(car_model: str, body_type: str) -> str:
    """Text sent to the AI about the car, e.g. 'Toyota Vios (sedan)' or 'sedan car'."""
    known_model = car_model and not car_model.startswith("Other")
    known_body = body_type and body_type in BODY_TYPES and body_type != "Not sure"
    if known_model and known_body:
        return f"{car_model} ({body_type.lower()})"
    if known_model:
        return car_model
    if known_body:
        return f"{body_type} car" if body_type in ("SUV", "MPV") else f"{body_type.lower()} car"
    return "Other / not sure"


def make_caption(number: int, preset: str, colour: str) -> str:
    if preset and preset != "None":
        name = preset
    elif colour and colour != "Keep original":
        name = colour
    else:
        name = "Custom design"
    return f"#{number} {name}"


def create_design(original, choices: dict, number: int) -> dict:
    """Run the AI pipeline. Raises ai.AIError with a friendly message if something fails."""
    vehicle = vehicle_description(choices["car_model"], choices.get("body_type", "Not sure"))
    prompt, used_llm = ai.build_edit_prompt(
        vehicle, choices["preset"], choices["colour"], choices["finish"],
        choices["rims"], choices["bodykit"], choices["extras"], choices["free_text"],
    )
    result = ai.edit_image(original, prompt, choices.get("strength", 0.5))
    return {
        "original": original,
        "image": result,
        "prompt": prompt,
        "used_llm": used_llm,
        "caption": make_caption(number, choices["preset"], choices["colour"]),
    }


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------------------------
def load_secrets_into_env():
    """Copy Streamlit secrets into environment variables, where ai.py reads them."""
    try:
        for name in SECRET_NAMES:
            if name in st.secrets and not os.getenv(name):
                os.environ[name] = str(st.secrets[name])
    except Exception:  # noqa: BLE001 - no secrets file when running locally with a .env file
        pass
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def init_state():
    defaults = {
        "photo_id": None,
        "original": None,
        "designs": [],
        "chat": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def handle_new_photo(data: bytes, photo_id: str):
    st.session_state.photo_id = photo_id
    st.session_state.original = ai.prepare_photo(Image.open(io.BytesIO(data)))


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------
def render_upload_column():
    st.subheader("1. Upload your car")
    uploaded = st.file_uploader("Photo of your car", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        st.info("Upload a photo to begin.")
        st.markdown(PHOTO_TIPS)
        return

    data = uploaded.getvalue()
    photo_id = hashlib.md5(data).hexdigest()
    if st.session_state.photo_id != photo_id:
        handle_new_photo(data, photo_id)

    st.image(st.session_state.original, caption="Your car")
    st.caption("Now choose the body type, your car model and a style on the right.")
    with st.expander("Photo tips"):
        st.markdown(PHOTO_TIPS)


def render_style_column() -> dict:
    st.subheader("2. Choose your style")
    if st.session_state.get("body_type") not in BODY_TYPES:
        st.session_state.body_type = "Not sure"
    if st.session_state.get("car_model") not in CAR_MODELS:
        st.session_state.car_model = "Other / not sure"
    type_col, model_col = st.columns(2)
    with type_col:
        body_type = st.selectbox("Body type", BODY_TYPES, key="body_type")
    with model_col:
        car_model = st.selectbox("Car model (optional)", CAR_MODELS, key="car_model")
    st.caption(suggestion_for(car_model, body_type))

    preset = st.selectbox("Preset style", PRESETS)
    col_a, col_b = st.columns(2)
    with col_a:
        colour = st.selectbox("Wrap / paint colour", COLOURS)
        rims = st.selectbox("Rims", RIMS)
    with col_b:
        finish = st.selectbox("Finish", FINISHES)
        bodykit = st.selectbox("Bodykit", BODYKITS)
    extras = st.multiselect("Extras", EXTRAS)
    free_text = st.text_area(
        "Or describe your idea in your own words (English or BM)",
        placeholder="e.g. Nak nampak macam kereta drift Jepun, warna biru gelap dengan rim emas",
        height=80,
    )
    strength = 0.5
    if ai.cf_uses_strength():
        strength = st.slider(
            "How much should the AI change your photo?", min_value=0.3, max_value=0.8, value=0.5, step=0.05,
            help="Lower keeps your car's shape and background closer to the original. "
                 "Higher gives stronger style changes but may distort the car.",
        )
    return {"car_model": car_model, "body_type": body_type, "preset": preset, "colour": colour, "finish": finish, "rims": rims,
            "bodykit": bodykit, "extras": extras, "free_text": free_text, "strength": strength}


def run_generation(choices: dict):
    original = st.session_state.original
    if original is None:
        st.error("Please upload a photo of your car first.")
        return
    if not ai.has_any_change(choices["preset"], choices["colour"], choices["finish"], choices["rims"],
                             choices["bodykit"], choices["extras"], choices["free_text"]):
        st.warning("Pick at least one change (a preset, colour, rims, bodykit or extra) or describe your idea.")
        return
    try:
        with st.spinner("Gemini is writing the design brief, then the image model is redesigning your car. "
                        "This can take a little while..."):
            design = create_design(original, choices, len(st.session_state.designs) + 1)
    except ai.AIError as exc:
        st.error(str(exc))
        return
    st.session_state.designs.append(design)


def render_results():
    designs = st.session_state.designs
    if not designs:
        return

    latest = designs[-1]
    st.divider()
    st.subheader(f"Your design: {latest['caption']}")
    before_col, after_col = st.columns(2)
    with before_col:
        st.image(latest["original"], caption="Before")
    with after_col:
        st.image(latest["image"], caption="After")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("Download this design", data=png_bytes(latest["image"]),
                       file_name=f"modmyride_design_{stamp}.png", mime="image/png")

    source = "Gemini (LLM)" if latest["used_llm"] else "the built-in template (LLM unavailable)"
    with st.expander("See the prompt that was sent to the image model"):
        st.markdown(f"Written by {source}:\n\n> {latest['prompt']}")

    st.subheader("Design history")
    st.caption("Each new design starts from your original photo, so try as many styles as you like.")
    columns = st.columns(4)
    for index, design in enumerate(designs):
        with columns[index % 4]:
            st.image(design["image"], caption=design["caption"])
    if st.button("Clear history"):
        st.session_state.designs = []
        st.rerun()


def render_studio():
    left, right = st.columns(2, gap="large")
    with left:
        render_upload_column()
    with right:
        choices = render_style_column()
        if st.button("3. Generate my design", type="primary"):
            run_generation(choices)
    render_results()


def render_advisor():
    st.write("Ask the AI style advisor for ideas. It knows the body type and car model from the Design Studio.")

    clicked = None
    example_cols = st.columns(2)
    for index, question in enumerate(ADVISOR_EXAMPLES):
        with example_cols[index % 2]:
            if st.button(question, key=f"example_{index}"):
                clicked = question

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed = st.chat_input("Ask about colours, rims, bodykits...")
    question = typed or clicked
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                vehicle = vehicle_description(st.session_state.get("car_model", "Other / not sure"),
                                              st.session_state.get("body_type", "Not sure"))
                reply = ai.advisor_reply(question, st.session_state.chat, vehicle)
            except ai.AIError as exc:
                reply = f"Sorry, I can't answer right now. {exc}"
        st.markdown(reply)
    st.session_state.chat.append({"role": "user", "content": question})
    st.session_state.chat.append({"role": "assistant", "content": reply})


def main():
    st.set_page_config(page_title="ModMyRide AI", page_icon="🚗", layout="wide")
    load_secrets_into_env()
    init_state()

    st.title("ModMyRide AI")
    st.write("Upload your car, choose a style, and see it redesigned by AI before you spend a single ringgit.")

    studio_tab, advisor_tab = st.tabs(["Design Studio", "Style Advisor"])
    with studio_tab:
        render_studio()
    with advisor_tab:
        render_advisor()

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
