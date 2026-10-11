# Hosting a public demo

The backend runs on [Railway](https://railway.app) from `backend/Dockerfile`; the frontend is a
static build on [Vercel](https://vercel.com). Both have free or trial tiers and deploy from GitHub.
Allow about 20 minutes.

These steps follow the repository's configuration (`backend/railway.json`, `frontend/vercel.json`).
The Docker image is built and smoke-tested in CI on every push, but the hosted setup itself has not
been run from this guide yet, so treat the first deploy as a test.

## 1. Backend on Railway

1. **New Project → Deploy from GitHub repo →** choose `STATMIND-AI`.
2. Open the service → **Settings → Source → Root Directory** → `backend`.
   Railway then builds `backend/Dockerfile` using `backend/railway.json`.
3. **New → Database → PostgreSQL** in the same project.
4. On the API service, **Variables**:

   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` |
   | `STATMIND_ENV` | `production` |
   | `JWT_SECRET` | 32 or more random characters (`python -c "import secrets; print(secrets.token_urlsafe(48))"`) |
   | `ALLOWED_ORIGIN_REGEX` | filled in at step 3 below; leave the default for the first deploy |
   | `ANTHROPIC_API_KEY` | optional: enables the Claude-backed analyst |

5. **Settings → Volumes → Add volume**, mount path `/data`. Uploaded datasets and trained models
   live there; without a volume they are lost on every redeploy.
6. **Settings → Networking → Generate Domain.** Open `https://<your-domain>/api/health`; it should
   return `"status": "ok"`.

`STATMIND_ENV=production` turns login on, so visitors create an account and only see their own
data. Leave MLflow and lakeFS unset for the demo: the app tracks experiments in its own database.

## 2. Frontend on Vercel

1. **Add New → Project →** import `STATMIND-AI`.
2. **Root Directory** → `frontend`. Vercel detects Vite from `vercel.json`.
3. **Environment Variables** → `VITE_API_URL` = `https://<your-railway-domain>` (no trailing slash).
   It is read at build time, so changing it later needs a redeploy.
4. **Deploy**, then open the Vercel URL.

## 3. Lock the API to your frontend

Back on Railway, set `ALLOWED_ORIGIN_REGEX` to your Vercel address, for example
`https://statmind-ai.*\\.vercel\\.app`, so other sites cannot call your API from a browser.

## 4. Check it

Register an account, upload `tests/data/breast_cancer.csv`, choose `target` in ML Studio and run
AutoML. Then add the URL to the repository's **About → Website** field and to the top of the README.

## If something fails

| Symptom | Likely cause |
|---|---|
| Build fails at `COPY requirements.txt` | Root Directory is not set to `backend` |
| Service crashes on start with a `JWT_SECRET` error | `STATMIND_ENV=production` without `JWT_SECRET` |
| App loads but every request fails | `VITE_API_URL` wrong or missing at build time, or `ALLOWED_ORIGIN_REGEX` does not match the Vercel URL |
| AutoML job is killed part-way | The plan's memory limit; use a smaller dataset or a larger plan |
| Data disappears after a redeploy | No volume mounted at `/data` |
