# Math2LaTeX 

A single FastAPI app that serves the website AND runs your trained model.
Draw on the left, get typeset LaTeX on the right, organized into chats.

## Files

```
app/
├── app/
│   ├── main.py          # API: login, chats, /api/predict, optional Ask-Claude
│   ├── model_io.py      # loads math2latex_model.pt
│   └── static/index.html
├── math2latex_model.pt  # the trained model 
├── requirements.txt
└── Dockerfile
```

## Run locally (Windows, using your existing venv)

```powershell
cd path\to\app
..\venv\Scripts\python.exe -m pip install fastapi "uvicorn[standard]" httpx python-multipart
$env:APP_PASSWORD = "choose-a-password"
..\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 — you can also reach it from your phone on the same
Wi-Fi at http://YOUR-PC-IP:8000. Your venv's CUDA torch means local predictions
run on the 4060.

## Deploy to Railway / Render / Fly.io

The included Dockerfile is all three platforms need:

1. Push this folder to a GitHub repo.
2. Railway: New Project -> Deploy from GitHub repo. Render: New Web Service -> pick the
   repo. Fly: `fly launch`.
3. Set environment variables in the platform dashboard:
   - `APP_PASSWORD` (required)
   - `ANTHROPIC_API_KEY` (optional — turns on the "Ask Claude" button; set a spending
     limit in the Anthropic console as a backstop)
4. Deploy. Inference runs on CPU there; this model is small, so a
   conversion takes well under a second.

**Persistence caveat:** chat history lives in a SQLite file (`data.db`). Railway/Render/Fly
have ephemeral filesystems by default, so history resets on each redeploy unless you attach
a volume and set `DB_PATH` to a path on it, e.g. `/data/data.db`.
Predictions themselves need no storage.

**Alternative for zero cost:** run the server on your PC and access it from anywhere
privately with Tailscale — no public exposure, no hosting bill.

## Environment variables

| Var                 | Required | Purpose                                        |
|---------------------|----------|------------------------------------------------|
| `APP_PASSWORD`      | yes      | Login password (server refuses to start unset) |
| `ANTHROPIC_API_KEY` | no       | Enables Ask Claude (key never leaves backend)  |
| `MODEL_PATH`        | no       | Default: math2latex_model.pt beside the app    |
| `DB_PATH`           | no       | Default: data.db in the working directory      |

## Notes

- Editing an entry's LaTeX (click the code, edit, press Enter) fixes model mistakes —
  and quietly builds a corrected dataset for future fine-tuning.
- The model was trained on MathWriting (CC BY-NC-SA): personal/non-commercial use only.
- Everything sensitive stays server-side: the browser only ever sees your password at
  login and the LaTeX results. The Anthropic key is never sent to the frontend.
