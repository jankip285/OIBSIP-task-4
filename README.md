# SpamShield AI

## Setup

Open PowerShell in this folder:

```powershell
cd "C:\Users\ASUS\Downloads\task 4 oibsip"
.\setup_env.ps1
```

## Run

After setup, launch the app with:

```powershell
.\run_app.ps1
```

## Alternate manual run

If you prefer:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

## Notes

- `best_spam_detector.keras` and `tokenizer.pkl` must remain in the same folder as `app.py`.
- Open `http://localhost:7860` in your browser after the app starts.
