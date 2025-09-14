
# Cross-Source Record Linker — Flexible Schema

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

