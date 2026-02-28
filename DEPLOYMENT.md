# Deployment Guide — Physical AI & Humanoid Robotics Book Platform

## Architecture Overview

```
GitHub Repo (this repo)
  |
  |-- Dockerfile  (repo root)
  |       |
  |       v
  |   Hugging Face Spaces  (Docker SDK, port 7860)
  |   URL: https://nafay-physical-ai-book-backend.hf.space
  |   - FastAPI + Uvicorn
  |   - Qdrant Cloud (external, already indexed)
  |   - SQLite at /data/app.db (HF persistent storage)
  |
  |-- project/frontend/
          |
          v
      Vercel  (Docusaurus static site)
      URL: https://<your-project>.vercel.app
```

---

## Part 1: Deploy Backend to Hugging Face Spaces

### Step 1 — Create the Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - Owner: `nafay` (your HF username)
   - Space name: `physical-ai-book-backend`
   - License: MIT (or your choice)
   - SDK: **Docker**
   - Visibility: Public (or Private — Private requires a paid plan for API access)
3. Click "Create Space".

### Step 2 — Link your GitHub repository

Option A — Push directly to HF Space git remote:

```bash
# From repo root D:/Physical-AI-and-Humanoid-Robtoics/
git remote add hf https://huggingface.co/spaces/nafay/physical-ai-book-backend
git push hf main
```

Option B — GitHub integration (recommended):
In the Space Settings page, under "Repository" connect your GitHub repo and select the `main` branch.

HF Spaces will detect the `Dockerfile` at the repo root and build it automatically.

### Step 3 — Set Repository Secrets (Environment Variables)

In your Space page, go to **Settings -> Repository secrets** and add each of the following.
Never paste secrets into code or the Dockerfile.

| Secret Name | Value / Description |
|---|---|
| `SECRET_KEY` | A long random string for JWT signing (generate with: `python -c "import secrets; print(secrets.token_hex(32))"`) |
| `DATABASE_URL` | `sqlite:////data/app.db` (four slashes — absolute path on HF persistent storage) |
| `OPENAI_API_KEY` | Your OpenAI API key |
| `GEMINI_API_KEY` | Your Google Gemini API key |
| `OPENROUTER_API_KEY` | Your OpenRouter API key |
| `QDRANT_URL` | `https://0044f4cb-e610-47e4-a014-49d836da6c90.eu-west-2-0.aws.cloud.qdrant.io` |
| `QDRANT_API_KEY` | Your Qdrant Cloud API key (JWT token) |
| `BACKEND_CORS_ORIGINS` | `["https://your-project.vercel.app"]` — update after Vercel deployment |

Optional (have defaults set in code):

| Secret Name | Default Value |
|---|---|
| `OPENROUTER_API_BASE` | `https://openrouter.ai/api/v1` |
| `OPENROUTER_APP_NAME` | `Physical-AI-Book-Platform` |
| `OPENROUTER_MODEL` | `openai/gpt-4o` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60480` |

### Step 4 — Verify the Backend

After the build completes (usually 3-5 minutes), visit:

```
https://nafay-physical-ai-book-backend.hf.space/health
```

Expected response:
```json
{"status": "healthy", "service": "Educational Book Platform API"}
```

Interactive API docs are available at:
```
https://nafay-physical-ai-book-backend.hf.space/api/v1/openapi.json
```

---

## Part 2: Deploy Frontend to Vercel

### Step 1 — Import the GitHub repository

1. Go to https://vercel.com/new
2. Click "Import Git Repository" and select this GitHub repo.
3. On the "Configure Project" screen set:
   - **Framework Preset**: Docusaurus (Vercel auto-detects from `vercel.json`)
   - **Root Directory**: `project/frontend`
   - **Build Command**: `npm run build` (pre-filled from `vercel.json`)
   - **Output Directory**: `build` (pre-filled from `vercel.json`)
   - **Node.js Version**: 18.x or higher

### Step 2 — Deploy

Click "Deploy". Vercel will build and publish the Docusaurus static site.
Your production URL will be something like:
```
https://physical-ai-book.vercel.app
```

### Step 3 — Update CORS on the backend

Now that you have the Vercel URL, go back to your HF Space secrets and update:

```
BACKEND_CORS_ORIGINS = ["https://physical-ai-book.vercel.app"]
```

Then restart the HF Space (Settings -> Restart Space) so the new env var takes effect.

---

## Part 3: Update Frontend API URL (if your HF Space username differs)

If your Hugging Face username is not `nafay`, or you chose a different Space name,
update the `API_BASE_URL` constant in these two files before pushing:

```
project/frontend/src/components/ChatbotWidget.jsx   — line 5
project/frontend/src/components/UrduTranslation.jsx — line 5
```

Change:
```js
const API_BASE_URL = 'https://nafay-physical-ai-book-backend.hf.space';
```

To:
```js
const API_BASE_URL = 'https://<your-hf-username>-<your-space-name>.hf.space';
```

The HF Space URL format is always:
```
https://<username>-<space-name>.hf.space
```
(hyphens replace underscores in usernames)

Then commit and push — Vercel will auto-redeploy.

---

## Part 4: Rebuilding the Qdrant Index (if needed)

The Qdrant Cloud collection already has 207 vectors indexed. If you need to re-index:

```bash
cd D:/Physical-AI-and-Humanoid-Robtoics
# Make sure your local .env has the correct QDRANT_URL and QDRANT_API_KEY
python project/scripts/index_content.py
```

This script does NOT need to run on HF Spaces — it runs locally and pushes vectors
directly to Qdrant Cloud.

---

## Troubleshooting

### Backend build fails on HF Spaces
- Check the Space "Logs" tab for the Docker build output.
- Most common cause: a package in `requirements.txt` fails to compile.
  The `build-essential` and `libpq-dev` layers in the Dockerfile cover most C extension builds.
- If `asyncpg` fails (it is not used with SQLite), you can safely remove it from
  `requirements.txt` since the app uses the SQLAlchemy sync engine with SQLite.

### 500 errors on /health after deploy
- The backend failed to start. Check the Space runtime logs.
- Most likely cause: a required secret (env var) is missing.
  Cross-check every row in the secrets table above.

### CORS errors in browser console
- The `BACKEND_CORS_ORIGINS` secret on HF Spaces does not include your Vercel URL.
- Update the secret value and restart the Space.

### SQLite database not persisting between Space restarts
- Confirm `DATABASE_URL` is set to `sqlite:////data/app.db` (four slashes).
- Three slashes (`sqlite:///`) makes a relative path; four slashes makes an absolute path to `/data/app.db`.
- HF Spaces persistent storage is mounted at `/data` — the Dockerfile creates this directory.

### Frontend shows "couldn't connect to AI assistant"
- The `API_BASE_URL` in the JSX components does not match the actual HF Space URL.
- Verify the URL by visiting it directly in a browser and checking the `/health` endpoint.

---

## File Reference

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the backend Docker image for HF Spaces |
| `project/backend/requirements.txt` | Python dependencies |
| `project/backend/app/core/config.py` | Pydantic settings — reads env vars / secrets |
| `project/backend/main.py` | FastAPI entry point (adds `/app` to sys.path for `import ai`) |
| `project/frontend/vercel.json` | Vercel build configuration for Docusaurus |
| `project/frontend/src/components/ChatbotWidget.jsx` | Chatbot UI — contains `API_BASE_URL` |
| `project/frontend/src/components/UrduTranslation.jsx` | Translation UI — contains `API_BASE_URL` |
