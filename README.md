
# Cross-Source Record Linker — Flexible Schema

**Demo Video:** https://drive.google.com/file/d/1KTwz7S5vd_sPuPqyClPF_-h1TUm3KNsx/view?usp=drive_link

This is the **full** build (no placeholders). It includes:
- Auto-detection + manual mapping UI
- Stable session state (Preview → Run works)
- Arrow-safe rendering
- **Flattened suspects** (Top-K candidates as columns) + CSV export

## Run
```bash
conda create -n xlinker python=3.11 -y && conda activate xlinker
pip install -r requirements.txt
streamlit run app.py
```

