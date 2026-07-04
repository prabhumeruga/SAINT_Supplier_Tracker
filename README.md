# S.A.I.N.T. — Supplier AI & Intelligence Network Tracker

A Streamlit app that generates a Weighted Risk Index (WRI) and CPO-style intelligence
report for a given supplier/vendor, using a two-model pipeline (Mistral draft →
DeepSeek audit) grounded in real data sources rather than scraped search results.

## Data sources

| Source | What it provides | Cost | Coverage |
|---|---|---|---|
| [SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | Verified revenue, net income, assets, liabilities straight from 10-K XBRL filings | Free, no key | US public (SEC-registered) companies only |
| [Financial Modeling Prep](https://site.financialmodelingprep.com/pricing-plans) | Company profile: sector, industry, description, market cap, employees, CEO | Free tier: 250 calls/day | Public companies |
| [Tavily](https://docs.tavily.com/documentation/api-credits) | Recent news / market context, replacing raw Google-scraping | Free tier: 1,000 credits/mo | Anything indexed by web search |

Each source is independent and optional — if a company isn't in SEC/FMP's
coverage (private or non-US suppliers) or a key isn't set, that source is
skipped with an on-screen note rather than breaking the analysis. When SEC
data is found, its revenue/net-income figures are shown as a separate
**Verified Financials** chart, clearly labeled as real filings data rather
than the LLM's own estimate.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # then fill in real values
export MISTRAL_API_KEY=...          # or use python-dotenv / direnv to load .env
export DEEPSEEK_API_KEY=...
export FMP_API_KEY=...              # optional -- company profile data
export TAVILY_API_KEY=...           # optional -- live news search
export SEC_USER_AGENT="YourOrg your-email@example.com"   # optional but recommended

streamlit run saint_app.py
```

The app will refuse to start (with a clear on-screen message) if `MISTRAL_API_KEY`
or `DEEPSEEK_API_KEY` is missing — it never falls back to a hardcoded key.
`FMP_API_KEY` and `TAVILY_API_KEY` are optional enhancements: without them the
app still runs, just with less-grounded reports (fewer real numbers, no live
news), and it tells you on-screen which sources it skipped.

### Getting the optional API keys (both free)

- **Financial Modeling Prep**: sign up at [financialmodelingprep.com](https://site.financialmodelingprep.com/), grab your API key from the dashboard. Free tier = 250 calls/day, no card required.
- **Tavily**: sign up at [tavily.com](https://www.tavily.com/), copy your API key. Free tier = 1,000 search credits/month, no card required.
- **SEC EDGAR** needs no signup or key at all — just set `SEC_USER_AGENT` to identify yourself/your org, per [SEC's fair-access policy](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

## Deploying for a demo (Streamlit Community Cloud)

1. Push this repo to GitHub (public or private both work).
2. Go to https://share.streamlit.io and sign in with **Continue with GitHub**.
3. Click **Create app** → **Yup, I have an app** → pick this repo, the branch
   (usually `main`), and set the main file path to `saint_app.py`.
4. Open **Advanced settings** before deploying and paste into the **Secrets** box:

   ```toml
   MISTRAL_API_KEY = "your-real-mistral-key"
   DEEPSEEK_API_KEY = "your-real-deepseek-key"
   FMP_API_KEY = "your-real-fmp-key"
   TAVILY_API_KEY = "your-real-tavily-key"
   SEC_USER_AGENT = "YourOrg your-email@example.com"
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
