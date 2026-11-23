# ModMyRide AI

Upload a photo of your car, pick a style, and see it redesigned by AI before you spend money at a workshop.

SDT 2143 Mini AI Hackathon, Theme 2: AI for Creativity. Built with Streamlit and hosted free on Streamlit Community Cloud.

## How it works

1. **Upload photo:** the user uploads a photo of their own car.
2. **Describe the car:** the user picks the body type (hatchback, sedan, SUV, MPV) and, optionally, the exact car model. The app shows matching style ideas.
3. **Choose a style:** preset, wrap colour, finish, rims, bodykit, extras, or a free-text idea in English or BM.
4. **Build prompt (Gemini LLM):** Gemini turns the choices into a detailed editing instruction that tells the image model to keep the same car, angle and background. It also translates BM requests.
5. **Edit the photo (FLUX.2 [klein] on Cloudflare Workers AI):** the image model applies the new exterior.
6. **Before / after:** the original and new design are shown side by side.
7. **Save or retry:** download the design, or try another style. Every design is kept in the session's history.

The **Style Advisor** tab is a Gemini chatbot that suggests colours, rims and bodykits for the user's car.

## Files

| File | What it does |
| --- | --- |
| `app.py` | The website (Streamlit): layout, buttons and how the steps connect |
| `ai.py` | Gemini prompt builder and style advisor, plus the image-editing backends (Cloudflare, Gemini, Hugging Face) |
| `requirements.txt` | Python packages the app needs |
| `.streamlit/secrets.toml.example` | Template for your keys when running on your own computer |
| `.gitignore` | Stops your secret key file from being uploaded to GitHub |

---

## Step 1: Get a Gemini API key (free)

1. Go to **aistudio.google.com** and sign in with a Google account.
2. Click **Get API key**, then **Create API key**.
3. Copy the key and keep it private. Never paste it into the code, GitHub, or screenshots.

The free tier covers the text model used for prompts and the Style Advisor. Gemini's image models need billing, so this app uses Cloudflare for images instead (Step 2).

## Step 2: Get Cloudflare Workers AI details (free)

The app uses Cloudflare's **FLUX.2 [klein] 4B** model (`@cf/black-forest-labs/flux-2-klein-4b`) to edit the photo. Each design uses only a small part of Cloudflare's free daily allowance (10,000 neurons per day, reset at 00:00 UTC, which is 8am Malaysia time).

1. Create a free account at **dash.cloudflare.com/sign-up**. No credit card needed.
2. In the dashboard, search for **Workers AI** and open it. Choose **Use REST API**.
3. **Account ID:** copy it from that page, or from your browser's address bar (`dash.cloudflare.com/<account-id>/...`). It is 32 characters long.
4. **API token:** click **Create a Workers AI API Token**, then **Create API Token**. Copy it with Ctrl+C. Do not retype it by hand, because letters like O and 0 are easy to mix up.

## Step 3: Put the code on GitHub

1. Create a free account at **github.com**.
2. Click **+** (top right), then **New repository**. Visibility: **Public**. Click **Create repository**.
3. Click **uploading an existing file** (or **Add file > Upload files**).
4. Drag in `app.py`, `ai.py`, `requirements.txt` and `README.md`. Do **not** upload any file that contains your keys.
5. Click **Commit changes**.

## Step 4: Deploy on Streamlit Community Cloud (free)

1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Choose your repository, branch `main`, main file path `app.py`.
4. Open **Advanced settings > Secrets** and paste (with your real values):
   ```toml
   GEMINI_API_KEY = "your-gemini-key"
   IMAGE_BACKEND = "cloudflare"
   CF_ACCOUNT_ID = "your-account-id"
   CF_API_TOKEN = "your-api-token"
   ```
5. Click **Deploy**. Your link looks like `https://your-app-name.streamlit.app`.

**Before your demo:** free apps go to sleep after about 12 hours without visitors. Open the link a few minutes before presenting and click the wake-up button if you see one.

**To change secrets later:** on share.streamlit.io, click your app's **⋮ > Settings > Secrets**, edit, save, then **⋮ > Reboot app**.

## Run it on your own computer (optional)

```bash
cd modmyride-ai
python -m venv venv
# Windows: venv\Scripts\activate     Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, paste your keys, then run `streamlit run app.py`. It opens at http://localhost:8501.

## Settings (secrets)

| Name | Needed? | Default | Purpose |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes | none | Gemini key for prompts and the Style Advisor |
| `IMAGE_BACKEND` | Yes, set to `cloudflare` | `gemini` | Which service edits the photo: `cloudflare`, `gemini` (needs billing) or `huggingface` |
| `CF_ACCOUNT_ID` | For Cloudflare | none | Your Cloudflare account ID |
| `CF_API_TOKEN` | For Cloudflare | none | Cloudflare Workers AI API token |
| `CF_IMAGE_MODEL` | No | `@cf/black-forest-labs/flux-2-klein-4b` | Cloudflare image-editing model |
| `TEXT_MODEL` | No | `gemini-3.5-flash` | Gemini model for prompts and the Style Advisor |
| `IMAGE_MODEL` | No | `gemini-3.1-flash-image` | Gemini image model (only if `IMAGE_BACKEND = "gemini"` and billing is on) |
| `HF_TOKEN` / `HF_IMAGE_MODEL` | Only for Hugging Face | none / `black-forest-labs/FLUX.1-Kontext-dev` | Backup image option with small free credits |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "No Gemini API key found" | Add `GEMINI_API_KEY` in Secrets (with quotes), then reboot the app |
| "Cloudflare rejected the request" | Check `CF_ACCOUNT_ID` and `CF_API_TOKEN`. The message ends with "Cloudflare said: ..." giving the exact reason |
| "not allowed to use the model" | Remove `CF_IMAGE_MODEL` from Secrets to use the default FLUX.2 [klein] model |
| "usage limit" | The free daily allowance is used up. It resets at 8am Malaysia time |
| The car changes too much | Use a clear side-view photo, ask for one or two changes at a time, or generate again |
| App shows an error page | On share.streamlit.io, open **Manage app** to see the logs |

## Development notes (useful for the reflection report)

- **Hosting:** first built with Gradio for Hugging Face Spaces, then moved to Streamlit Community Cloud when Gradio Spaces became paid.
- **Image model:** Gemini image editing has no free tier (billing needed, minimum prepayment), so image editing moved to Cloudflare Workers AI. Cloudflare's older Stable Diffusion img2img model was restricted for new accounts, so the app uses FLUX.2 [klein].
- **Teachable Machine:** a car model / body type classifier was tried, but similar-looking cars gave low confidence, so users now choose the body type and model themselves.

## Screenshots to take for the reflection report

- Google AI Studio API key page and Cloudflare Workers AI page (keys hidden)
- The code for each step: `build_edit_prompt`, `edit_image` / `_cloudflare_flux_edit`, `advisor_reply`
- GitHub repository, Streamlit deploy screen and secrets page (values hidden), and the live app
- Before/after results for 3 or more styles, and a Style Advisor conversation
