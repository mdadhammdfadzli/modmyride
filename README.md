# ModMyRide AI

Upload a photo of your car, pick a style, and see it redesigned by AI before you spend money at a workshop.

SDT 2143 Mini AI Hackathon, Theme 2: AI for Creativity. Built with Streamlit and hosted free on Streamlit Community Cloud.

## How it works

1. **Upload photo:** the user uploads a photo of their own car.
2. **Detect model (Teachable Machine):** a trained image classifier recognises the car model, for example Perodua Myvi.
3. **Choose a style:** preset, wrap colour, finish, rims, bodykit, extras, or a free-text idea in English or BM.
4. **Build prompt (Gemini LLM):** Gemini turns the choices into a detailed editing prompt that tells the image model to keep the same car, angle and background.
5. **Edit the photo (Gemini image model):** the image model applies the new exterior.
6. **Before / after:** the original and new design are shown side by side.
7. **Save or retry:** download the design, or try another style. Every design is kept in the session's history.

The **Style Advisor** tab is a Gemini chatbot that suggests colours, rims and bodykits for the user's car.

## Files

| File | What it does |
| --- | --- |
| `app.py` | The website (Streamlit): layout, buttons and how the steps connect |
| `ai.py` | Gemini prompt builder, image editing and style advisor |
| `classifier.py` | Loads and runs the Teachable Machine car classifier |
| `model/` | Put `keras_model.h5` and `labels.txt` from Teachable Machine here |
| `requirements.txt` | Python packages the app needs |
| `.streamlit/secrets.toml.example` | Template for your API key when running on your own computer |
| `.gitignore` | Stops your secret key file from being uploaded to GitHub |

---

## Step 1: Get a Gemini API key

1. Go to **aistudio.google.com** and sign in with a Google account.
2. Click **Get API key**, then **Create API key**.
3. Copy the key and keep it private. Never paste it into the code, GitHub, or screenshots.
4. On the AI Studio rate-limits page, check which models your free tier can use. If `gemini-3.1-flash-image` is not available on your key, try `gemini-3.1-flash-lite-image` (set it as `IMAGE_MODEL`, see Settings below).

## Step 2: Train the car classifier in Teachable Machine

1. Go to **teachablemachine.withgoogle.com**, then **Get Started > Image Project > Standard image model**.
2. Make one class per car model, for example `Myvi`, `Axia`, `Saga`, `City`, `Civic`. Add one more class called `Other` with random non-matching cars.
3. Upload 50 to 100 photos per class: different colours, angles and backgrounds.
4. Click **Train Model**, then test it with new photos in the Preview panel. Screenshot the accuracy (**Advanced > Under the hood**) for your report.
5. Click **Export Model > Tensorflow > Keras > Download my model**.
6. Unzip it and put `keras_model.h5` and `labels.txt` in the `model/` folder.

Class names are matched to the app's car list automatically, so `Myvi` becomes `Perodua Myvi`. The app still works without this model; users just choose their car from the list. You can add the model later.

## Step 3: Put the code on GitHub

1. Create a free account at **github.com**.
2. Click **+** (top right), then **New repository**. Name: `modmyride-ai`. Visibility: **Public**. Click **Create repository**.
3. On the new repository page, click **uploading an existing file**.
4. Drag in `app.py`, `ai.py`, `classifier.py`, `requirements.txt`, `README.md` and the `model` folder.
5. Do **not** upload any file that contains your API key.
6. Click **Commit changes**.

## Step 4: Deploy on Streamlit Community Cloud (free)

1. Go to **share.streamlit.io** and sign in with your GitHub account. Allow Streamlit to access your repositories.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Choose your repository `modmyride-ai`, branch `main`, main file path `app.py`.
4. Optional: choose a custom app URL, for example `modmyride-ai`.
5. Open **Advanced settings**:
   - **Python version:** choose **3.12** (or 3.11). Car detection needs one of these.
   - **Secrets:** paste this line with your real key:
     ```toml
     GEMINI_API_KEY = "paste-your-key-here"
     ```
6. Click **Deploy**. The first build takes a few minutes because TensorFlow is large.
7. Test on a laptop and a phone. Your link looks like `https://modmyride-ai.streamlit.app`.

**Before your demo:** free apps go to sleep after about 12 hours without visitors. Open the link a few minutes before presenting and click the wake-up button if you see one.

**To change the key later:** in your app, open the menu (bottom right "Manage app" or the three dots), go to **Settings > Secrets**, edit, and save.

## Free image backend: Cloudflare Workers AI

If your Gemini key has no free quota for image editing ("no free quota" or "usage limit" errors), switch the image step to Cloudflare. Gemini still writes the prompts and runs the Style Advisor.

The app uses Cloudflare's **FLUX.2 [klein] 4B** model (`@cf/black-forest-labs/flux-2-klein-4b`), which edits a photo from a text instruction. Each design uses only a small part of Cloudflare's free daily allowance (10,000 neurons per day, reset at 00:00 UTC, which is 8am Malaysia time).

1. Create a free account at **dash.cloudflare.com/sign-up**. No credit card needed.
2. In the dashboard, search for **Workers AI** and open it. Choose **Use REST API**.
3. **Account ID:** copy it from that page, or from your browser's address bar (`dash.cloudflare.com/<account-id>/...`). It is 32 characters long.
4. **API token:** click **Create a Workers AI API Token**, then **Create API Token**. Copy it with Ctrl+C. Do not retype it by hand, because letters like O and 0 are easy to mix up.
5. In Streamlit, open your app's **Settings > Secrets** and add these lines under your Gemini key:
   ```toml
   IMAGE_BACKEND = "cloudflare"
   CF_ACCOUNT_ID = "your-account-id"
   CF_API_TOKEN = "your-api-token"
   ```
6. Save, then **Reboot app** from the Streamlit dashboard.

**Tips for good results:**
- Clear side or front three-quarter photos work best.
- Ask for one or two changes at a time (for example colour plus rims) rather than everything at once.
- If a result changes the car too much, generate again; each run gives a slightly different result.

**Note for your report:** Cloudflare restricted its older Stable Diffusion img2img model for new accounts during development, so the app switched to FLUX.2 [klein]. This is a good example of adapting when AI services change.

## Run it on your own computer (optional)

You need Python 3.11 or 3.12.

```bash
cd modmyride-ai
python -m venv venv
# Windows: venv\Scripts\activate     Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, paste your key, then run:

```bash
streamlit run app.py
```

It opens in your browser at http://localhost:8501.

## Settings (secrets)

Add these in Streamlit Cloud under **Settings > Secrets**, one per line in the form `NAME = "value"`.

| Name | Needed? | Default | Purpose |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes | none | Your Gemini key |
| `TEXT_MODEL` | No | `gemini-3.5-flash` | Model for the prompt builder and style advisor |
| `IMAGE_MODEL` | No | `gemini-3.1-flash-image` | Model that edits the car photo |
| `IMAGE_BACKEND` | No | `gemini` | `gemini`, `cloudflare` (free, see above) or `huggingface` (plan B below) |
| `CF_ACCOUNT_ID` | For Cloudflare | none | Your Cloudflare account ID |
| `CF_API_TOKEN` | For Cloudflare | none | Cloudflare Workers AI API token |
| `CF_IMAGE_MODEL` | No | `@cf/black-forest-labs/flux-2-klein-4b` | Cloudflare image-editing model |
| `HF_TOKEN` | Only for plan B | none | Hugging Face access token |
| `HF_IMAGE_MODEL` | No | `black-forest-labs/FLUX.1-Kontext-dev` | Hugging Face image-editing model for plan B |

**Plan B:** if Gemini image editing is not available to you, create a Hugging Face access token (huggingface.co > Settings > Access Tokens) and add `IMAGE_BACKEND = "huggingface"` and `HF_TOKEN = "<token>"`. Hugging Face's free inference credits are small, so use this for testing and the demo only.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "No Gemini API key found" | Add `GEMINI_API_KEY` in Settings > Secrets (with quotes around the key), then reboot the app |
| "model was not found" | Google renamed or retired the model. Check current names at ai.google.dev/gemini-api/docs/models and set `TEXT_MODEL` / `IMAGE_MODEL` |
| "usage limit" | Free-tier quota reached. Wait, or use a teammate's key for the demo |
| The car's shape changes | Use clearer side-view photos, pick fewer changes at once, or try the other image model |
| "Car detection is not set up yet" | Check both files are in `model/` on GitHub and named exactly `keras_model.h5` and `labels.txt`, and that the Python version is 3.11 or 3.12 |
| Build fails / app shows an error | Click **Manage app** (bottom right) to see the logs. Send the red error lines for help |

## Screenshots to take for the reflection report

- Teachable Machine classes, training and accuracy screens
- Google AI Studio API key page (key hidden)
- The code for each step: `classifier.py` predict, `build_edit_prompt`, `edit_image`
- GitHub repository, Streamlit deploy screen and secrets page (values hidden), and the live app
- Before/after results for 3 or more styles, and a style advisor conversation
