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

import base64
import io
import json
import os
import urllib.error
import urllib.request

from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Settings (read each time, so secrets set by the hosting platform are always picked up)
# ---------------------------------------------------------------------------
def text_model() -> str:
    return os.getenv("TEXT_MODEL", "gemini-3.5-flash")


def image_model() -> str:
    return os.getenv("IMAGE_MODEL", "gemini-3.1-flash-image")


def image_backend() -> str:
    """'gemini' (default), 'cloudflare' (free daily allowance) or 'huggingface' (small free credits)."""
    return os.getenv("IMAGE_BACKEND", "gemini").strip().lower()


def hf_image_model() -> str:
    return os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-Kontext-dev")


def cf_image_model() -> str:
    return os.getenv("CF_IMAGE_MODEL", "@cf/runwayml/stable-diffusion-v1-5-img2img")


def uses_cloudflare() -> bool:
    return image_backend() in ("cloudflare", "cf")


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
    compact = upper.replace(" ", "")
    if "LIMIT:0" in compact or "LIMIT\":0" in compact:
        return AIError(
            f"Your API key has no free quota for the {what}. Switch to the free Cloudflare backend "
            "(see README: 'Free image backend: Cloudflare'), or enable billing for this model."
        )
    if "429" in text or "RESOURCE_EXHAUSTED" in upper or "QUOTA" in upper or "CAPACITY" in upper:
        return AIError(
            f"The {what} has hit its usage limit for now. Wait a minute and try again, "
            "or check your free quota for today."
        )
    if "API_KEY" in upper or "API KEY" in upper or "401" in text or "403" in text or "PERMISSION" in upper \
            or "AUTHENTICATION" in upper:
        if "Cloudflare" in what:
            return AIError(
                "Cloudflare rejected the request. Check CF_ACCOUNT_ID and CF_API_TOKEN in your secrets, "
                "and that the token was made with the 'Workers AI' template."
            )
        return AIError(
            f"The {what} rejected the API key. Check that GEMINI_API_KEY is correct and that "
            f"your key can use the model '{image_model() if 'image' in what else text_model()}'."
        )
    if "404" in text or "NOT_FOUND" in upper or "NO ROUTE" in upper:
        return AIError(
            f"The {what} was not found. It may have been renamed. Set the TEXT_MODEL / IMAGE_MODEL / "
            "CF_IMAGE_MODEL secret to a current model name."
        )
    if "SAFETY" in upper or "BLOCKED" in upper or "NSFW" in upper:
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


KEYWORD_PROMPT_INSTRUCTIONS = """You write prompts for a Stable Diffusion img2img model that restyles a photo of a car.

You receive the car model and the owner's chosen modifications. Write ONE comma-separated prompt under 60 words
that describes the FINISHED photo (a description, not instructions):
- start with "photo of a <car model>"
- then the paint/wrap colour and finish, rims, bodykit parts, extras
- end with "photorealistic, sharp focus, professional automotive photography"
Translate any Bahasa Melayu into English. Skip parts marked "Keep original".
Reply with the prompt only, no quotes, no markdown."""

CF_NEGATIVE_PROMPT = (
    "deformed car, distorted body, extra wheels, missing wheels, melted, blurry, low quality, cartoon, "
    "drawing, text, watermark, logo, people"
)


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


def keyword_fallback(car_model, preset, colour, finish, rims, bodykit, extras, free_text) -> str:
    """Short keyword prompt for Stable Diffusion, used if the LLM is unavailable."""
    parts = [f"photo of a {car_model}" if car_model and not car_model.startswith("Other") else "photo of a car"]
    if preset and preset != "None":
        parts.append(f"{preset} style")
    paint = " ".join(x for x in (colour, finish) if x and x != "Keep original")
    if paint:
        parts.append(f"{paint} paint")
    if rims and rims != "Keep original":
        parts.append(f"{rims} rims")
    if bodykit and bodykit != "Keep original":
        parts.append(bodykit.lower())
    parts.extend(e.lower() for e in (extras or []))
    if free_text and free_text.strip():
        parts.append(free_text.strip())
    parts.append("photorealistic, sharp focus, professional automotive photography")
    return ", ".join(parts)


def build_edit_prompt(car_model, preset, colour, finish, rims, bodykit, extras, free_text):
    """Return (prompt, used_llm). Falls back to a template if Gemini text is unavailable.

    Gemini image models get a full editing instruction; the Cloudflare Stable Diffusion model
    gets a short keyword-style description, which is what that model understands best.
    """
    choices_text = describe_choices(car_model, preset, colour, finish, rims, bodykit, extras, free_text)
    keyword_style = uses_cloudflare()
    instructions = KEYWORD_PROMPT_INSTRUCTIONS if keyword_style else PROMPT_BUILDER_INSTRUCTIONS
    try:
        from google.genai import types

        client = _get_gemini_client()
        response = client.models.generate_content(
            model=text_model(),
            contents=choices_text,
            config=types.GenerateContentConfig(system_instruction=instructions),
        )
        prompt = (response.text or "").strip()
        if prompt:
            return prompt, True
    except AIError:
        raise  # missing key: the image step would fail too, so tell the user now
    except Exception as exc:  # noqa: BLE001 - any LLM failure falls back to the template
        print(f"[prompt builder] falling back to template: {exc}")
    if keyword_style:
        return keyword_fallback(car_model, preset, colour, finish, rims, bodykit, extras, free_text), False
    return fallback_prompt(choices_text), False


# ---------------------------------------------------------------------------
# 2. Image editor
# ---------------------------------------------------------------------------
def edit_image(photo: Image.Image, prompt: str, strength: float = 0.5) -> Image.Image:
    """strength (0-1) is only used by the Cloudflare model: lower keeps the result closer to the photo."""
    if uses_cloudflare():
        return _edit_with_cloudflare(photo, prompt, strength)
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


def _sd_size(photo: Image.Image, longest: int = 768):
    """Stable Diffusion 1.5 works best around 512-768 px, with sides in multiples of 64."""
    scale = longest / max(photo.size)
    width = max(256, int(photo.width * scale) // 64 * 64)
    height = max(256, int(photo.height * scale) // 64 * 64)
    return width, height


def _image_from_cloudflare(content_type: str, body: bytes) -> Image.Image:
    if content_type.startswith("image/") or body[:8] == b"\x89PNG\r\n\x1a\n" or body[:3] == b"\xff\xd8\xff":
        return Image.open(io.BytesIO(body)).convert("RGB")
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AIError("Cloudflare returned something that is not an image.") from exc
    if not data.get("success", True):
        raise _friendly_error(RuntimeError(json.dumps(data.get("errors"))), "Cloudflare image model")
    result = data.get("result")
    encoded = result.get("image") if isinstance(result, dict) else None
    if encoded:
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    raise AIError("Cloudflare did not return an image. Try again or try a different photo.")


def _edit_with_cloudflare(photo: Image.Image, prompt: str, strength: float = 0.5) -> Image.Image:
    account_id = os.getenv("CF_ACCOUNT_ID")
    token = os.getenv("CF_API_TOKEN")
    if not account_id or not token:
        raise AIError(
            "IMAGE_BACKEND is 'cloudflare' but CF_ACCOUNT_ID or CF_API_TOKEN is missing from your secrets."
        )

    width, height = _sd_size(photo)
    resized = photo.resize((width, height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")

    payload = {
        "prompt": prompt,
        "negative_prompt": CF_NEGATIVE_PROMPT,
        "image": list(buffer.getvalue()),
        "strength": float(min(max(strength, 0.1), 0.95)),
        "guidance": 7.5,
        "num_steps": 20,
        "width": width,
        "height": height,
    }
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{cf_image_model()}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise _friendly_error(RuntimeError(f"{exc.code} {detail}"), "Cloudflare image model") from exc
    except urllib.error.URLError as exc:
        raise AIError(f"Could not reach Cloudflare: {exc.reason}") from exc

    result = _image_from_cloudflare(content_type, body)
    return result.resize(photo.size, Image.Resampling.LANCZOS)


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
