"""
AI services for ModMyRide AI.

Three jobs live here:
  1. build_edit_prompt()  - Gemini (LLM) turns the user's choices into a detailed editing prompt.
  2. edit_image()         - an image-editing model applies that prompt to the user's car photo.
  3. advisor_reply()      - Gemini answers styling questions in the "Style advisor" chat.

Model names and keys come from environment variables / secrets,
so you can switch models without touching the code.
On Streamlit Community Cloud, add them under App settings > Secrets.
"""

import io
import os

from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Settings (read each time, so secrets set by the hosting platform are always picked up)
# ---------------------------------------------------------------------------
def text_model() -> str:
    return os.getenv("TEXT_MODEL", "gemini-3.5-flash")


def image_model() -> str:
    return os.getenv("IMAGE_MODEL", "gemini-3.1-flash-image")


def image_backend() -> str:
    """'gemini' (default) or 'huggingface' (plan B if Gemini image editing is not available to you)."""
    return os.getenv("IMAGE_BACKEND", "gemini").strip().lower()


def hf_image_model() -> str:
    return os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-Kontext-dev")


MAX_PHOTO_SIDE = 1024  # smaller photos = faster and cheaper edits


class AIError(Exception):
    """An error with a message that is safe and helpful to show to the user."""


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise AIError(
            "No Gemini API key found. Get a free key at aistudio.google.com, then add it as a "
            "secret named GEMINI_API_KEY (Streamlit Community Cloud: App settings > Secrets)."
        )
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - only when requirements are missing
        raise AIError("The google-genai package is not installed. Run: pip install google-genai") from exc

    _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _friendly_error(exc: Exception, what: str) -> AIError:
    """Turn raw API errors into messages a user (or lecturer) can understand."""
    text = str(exc)
    upper = text.upper()
    if "429" in text or "RESOURCE_EXHAUSTED" in upper or "QUOTA" in upper:
        return AIError(
            f"The {what} has hit its usage limit for now. Wait a minute and try again, "
            "or check your free-tier quota in Google AI Studio."
        )
    if "API_KEY" in upper or "API KEY" in upper or "401" in text or "403" in text or "PERMISSION" in upper:
        return AIError(
            f"The {what} rejected the API key. Check that GEMINI_API_KEY is correct and that "
            f"your key can use the model '{image_model() if 'image' in what else text_model()}'."
        )
    if "404" in text or "NOT_FOUND" in upper:
        return AIError(
            f"The {what} model was not found. It may have been renamed. Set the "
            "TEXT_MODEL / IMAGE_MODEL secret to a current model name from ai.google.dev."
        )
    if "SAFETY" in upper or "BLOCKED" in upper:
        return AIError("The AI refused this request for safety reasons. Try a different photo or style.")
    short = text if len(text) < 200 else text[:200] + "..."
    return AIError(f"Something went wrong with the {what}: {short}")


# ---------------------------------------------------------------------------
# Photo helpers
# ---------------------------------------------------------------------------
def prepare_photo(photo: Image.Image) -> Image.Image:
    """Fix phone-camera rotation, convert to RGB and shrink large photos."""
    photo = ImageOps.exif_transpose(photo)
    photo = photo.convert("RGB")
    photo.thumbnail((MAX_PHOTO_SIDE, MAX_PHOTO_SIDE), Image.Resampling.LANCZOS)
    return photo


# ---------------------------------------------------------------------------
# 1. Prompt builder (LLM)
# ---------------------------------------------------------------------------
PROMPT_BUILDER_INSTRUCTIONS = """You write editing instructions for an AI photo editor that customises car exteriors.

You receive the car model and the owner's chosen modifications. Write ONE clear editing instruction that:
- keeps the SAME car: same body shape, proportions, camera angle, background, lighting and reflections direction
- keeps the number plate unchanged and does not add any brand logos or text unless asked
- changes ONLY the exterior parts the owner asked for (paint/wrap, finish, rims, bodykit, decals)
- describes each change concretely (colour, finish, rim style and colour, bodykit parts such as front lip, side skirts, rear diffuser, spoiler)
- asks for a photorealistic result that looks like a real photo of a modified car

If the owner wrote a free-text request, possibly in Bahasa Melayu or mixed language, translate it into concrete changes.
If an option says "Keep original", do not change that part.
Reply with the instruction only, in English, under 120 words, no markdown, no quotes."""


def describe_choices(car_model, preset, colour, finish, rims, bodykit, extras, free_text) -> str:
    """Plain-text summary of the user's choices (sent to the LLM, also shown in the app)."""
    lines = [f"Car model: {car_model or 'unknown car'}"]
    if preset and preset != "None":
        lines.append(f"Overall style: {preset}")
    if colour and colour != "Keep original":
        lines.append(f"Paint / wrap colour: {colour}")
    if finish and finish != "Keep original":
        lines.append(f"Finish: {finish}")
    if rims and rims != "Keep original":
        lines.append(f"Rims: {rims}")
    if bodykit and bodykit != "Keep original":
        lines.append(f"Bodykit: {bodykit}")
    if extras:
        lines.append(f"Extras: {', '.join(extras)}")
    if free_text and free_text.strip():
        lines.append(f"Owner's own words: {free_text.strip()}")
    return "\n".join(lines)


def has_any_change(preset, colour, finish, rims, bodykit, extras, free_text) -> bool:
    picked = [
        preset and preset != "None",
        colour and colour != "Keep original",
        finish and finish != "Keep original",
        rims and rims != "Keep original",
        bodykit and bodykit != "Keep original",
        bool(extras),
        bool(free_text and free_text.strip()),
    ]
    return any(picked)


def fallback_prompt(choices_text: str) -> str:
    """Template prompt used if the LLM is unavailable, so the demo can still run."""
    return (
        "Edit this photo of a car. Keep exactly the same car body shape, proportions, camera angle, "
        "background, lighting and number plate. Only change the exterior as follows:\n"
        f"{choices_text}\n"
        "Make it photorealistic, like a real photo of a professionally modified car."
    )


def build_edit_prompt(car_model, preset, colour, finish, rims, bodykit, extras, free_text):
    """Return (prompt, used_llm). Falls back to a template if Gemini text is unavailable."""
    choices_text = describe_choices(car_model, preset, colour, finish, rims, bodykit, extras, free_text)
    try:
        from google.genai import types

        client = _get_gemini_client()
        response = client.models.generate_content(
            model=text_model(),
            contents=choices_text,
            config=types.GenerateContentConfig(system_instruction=PROMPT_BUILDER_INSTRUCTIONS),
        )
        prompt = (response.text or "").strip()
        if prompt:
            return prompt, True
    except AIError:
        raise  # missing key: the image step would fail too, so tell the user now
    except Exception as exc:  # noqa: BLE001 - any LLM failure falls back to the template
        print(f"[prompt builder] falling back to template: {exc}")
    return fallback_prompt(choices_text), False


# ---------------------------------------------------------------------------
# 2. Image editor
# ---------------------------------------------------------------------------
def edit_image(photo: Image.Image, prompt: str) -> Image.Image:
    if image_backend() in ("huggingface", "hf"):
        return _edit_with_huggingface(photo, prompt)
    return _edit_with_gemini(photo, prompt)


def _edit_with_gemini(photo: Image.Image, prompt: str) -> Image.Image:
    from google.genai import types

    client = _get_gemini_client()
    try:
        response = client.models.generate_content(
            model=image_model(),
            contents=[prompt, photo],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
    except Exception as exc:  # noqa: BLE001
        raise _friendly_error(exc, "image model") from exc

    image = _first_image_from_response(response)
    if image is not None:
        return image

    # No image came back: show whatever the model said instead
    note = ""
    try:
        note = (response.text or "").strip()
    except Exception:  # noqa: BLE001
        pass
    raise AIError(
        "The image model did not return a picture. "
        + (f"It said: {note[:200]}" if note else "Try a clearer photo or a simpler style.")
    )


def _first_image_from_response(response):
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return Image.open(io.BytesIO(inline.data)).convert("RGB")
    return None


def _edit_with_huggingface(photo: Image.Image, prompt: str) -> Image.Image:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise AIError("IMAGE_BACKEND is 'huggingface' but no HF_TOKEN secret was found.")
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:  # pragma: no cover
        raise AIError("The huggingface_hub package is not installed.") from exc

    buffer = io.BytesIO()
    photo.save(buffer, format="PNG")
    try:
        client = InferenceClient(api_key=token)
        result = client.image_to_image(buffer.getvalue(), prompt=prompt, model=hf_image_model())
    except Exception as exc:  # noqa: BLE001
        raise _friendly_error(exc, "Hugging Face image model") from exc
    return result.convert("RGB")


# ---------------------------------------------------------------------------
# 3. Style advisor chat (LLM)
# ---------------------------------------------------------------------------
ADVISOR_INSTRUCTIONS = """You are the ModMyRide AI style advisor, a friendly car-modification consultant in Malaysia.
Help car owners choose exterior styles: wrap colours, finishes, rims, bodykits, stripes and decals.
- Give specific, practical suggestions and explain briefly why they suit the car.
- Reply in the same language the user writes in (English or Bahasa Melayu).
- Keep answers short: at most about 120 words, use a short list when giving options.
- You can mention that wraps are usually easier to reverse than a full respray.
- Never claim a modification is legal or approved. If asked about legality, say rules can change and the
  owner should check the latest JPJ (Road Transport Department) and insurance requirements.
- Do not invent exact prices; if asked, say prices vary by workshop and suggest getting quotes."""


def _message_text(content) -> str:
    """Gradio chat messages can hold a string or a list of content blocks; return plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, (list, tuple)):
        return " ".join(_message_text(item) for item in content if item is not None).strip()
    return "" if content is None else str(content)


def advisor_reply(message: str, history: list, car_model: str | None = None) -> str:
    from google.genai import types

    client = _get_gemini_client()

    contents = []
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else None
        text = _message_text(turn.get("content")) if isinstance(turn, dict) else ""
        if role in ("user", "assistant") and text:
            contents.append(
                types.Content(role="user" if role == "user" else "model", parts=[types.Part(text=text)])
            )

    context = f"(The user's car: {car_model}.) " if car_model and not car_model.startswith("Other") else ""
    contents.append(types.Content(role="user", parts=[types.Part(text=context + message)]))

    try:
        response = client.models.generate_content(
            model=text_model(),
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=ADVISOR_INSTRUCTIONS),
        )
    except Exception as exc:  # noqa: BLE001
        raise _friendly_error(exc, "style advisor") from exc
    return (response.text or "Sorry, I couldn't think of a reply. Try asking another way.").strip()
