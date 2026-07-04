# S.A.I.N.T. — Supplier AI & Intelligence Network Tracker

A Streamlit app that generates a Weighted Risk Index (WRI) and CPO-style intelligence
report for a given supplier/vendor, using a two-model pipeline (Mistral draft →
DeepSeek audit) plus live web context.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # then fill in real values
export MISTRAL_API_KEY=...          # or use python-dotenv / direnv to load .env
export DEEPSEEK_API_KEY=...

streamlit run saint_app.py
```

The app will refuse to start (with a clear on-screen message) if either API key
env var is missing — it never falls back to a hardcoded key.

## Deploying for a demo (Streamlit Community Cloud)

1. Push this repo to GitHub (public or private both work).
2. Go to https://share.streamlit.io and sign in with **Continue with GitHub**.
3. Click **Create app** → **Yup, I have an app** → pick this repo, the branch
   (usually `main`), and set the main file path to `saint_app.py`.
4. Open **Advanced settings** before deploying and paste into the **Secrets** box:

   ```toml
   MISTRAL_API_KEY = "your-real-mistral-key"
   DEEPSEEK_API_KEY = "your-real-deepseek-key"
   ```

   Root-level secrets like these are exposed both as `st.secrets[...]` and as
   regular OS environment variables, so the existing `os.getenv(...)` calls in
   `saint_app.py` work unchanged — no code edits needed.
5. Click **Deploy**. The app builds in a couple of minutes and you get a
   permanent `*.streamlit.app` URL you can reuse for future demos.
6. To rotate keys or update anything later: app dropdown (⋮) → **Settings** →
   **Secrets**, edit, save — the app restarts automatically with the new values.

## Data retention

Analyses are stored in a local SQLite file (`~/saint_data.db` on whichever
machine/container runs the app) and auto-purge after 12 months. On Community
Cloud this file lives inside the app's container, so it resets on a full
redeploy/restart — for a persistent multi-session history you'd want an
external database instead of local SQLite.

## Repo hygiene

- `.gitignore` already excludes `*.db` files and any `.env` — never commit
  real secrets or the local database.
- `.env.example` shows the two required variable names without real values.
