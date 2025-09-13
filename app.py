
import io, json
import streamlit as st
import pandas as pd
from matching import link, guess_column, CANDIDATE_KEYS

st.set_page_config(page_title="Cross-Source Record Linker — Flexible v5", layout="wide")
st.title("🔗 Cross-Source Record Linker — Flexible Schema (v5)")

st.markdown("Upload any two CSVs. Auto-guess or map columns, then run. Suspects show Top-K candidates in columns.")

# -------- Session State --------
if "A_df" not in st.session_state: st.session_state.A_df = None
if "B_df" not in st.session_state: st.session_state.B_df = None
if "mapping" not in st.session_state: st.session_state.mapping = {"A": {}, "B": {}}
if "show_mapping" not in st.session_state: st.session_state.show_mapping = False
if "autoA" not in st.session_state: st.session_state.autoA = {}
if "autoB" not in st.session_state: st.session_state.autoB = {}

# -------- Helpers --------
def _safe_display_df(df):
    import pandas as _pd, json as _json
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].apply(lambda v: isinstance(v, (list, dict))).any():
            out[col] = out[col].apply(lambda v: _json.dumps(v, ensure_ascii=False))
    return out

def _flatten_suspects(suspects_df, A_df, B_df, mapping, top_k: int = 3):
    import json as _json
    if suspects_df is None or len(suspects_df) == 0:
        return suspects_df

    out_rows = []
    a_id_col = mapping.get("A", {}).get("id")
    b_id_col = mapping.get("B", {}).get("id")

    for _, row in suspects_df.iterrows():
        base = {"a_index": row.get("a_index"), "reason": row.get("reason"), "a_id": ""}
        try:
            if a_id_col and a_id_col in A_df.columns and base["a_index"] is not None:
                base["a_id"] = str(A_df.loc[base["a_index"], a_id_col])
        except Exception:
            pass

        cands = row.get("candidates", []) or []
        if isinstance(cands, str):
            try:
                cands = _json.loads(cands)
            except Exception:
                cands = []

        for j in range(top_k):
            prefix = f"cand{j+1}_"
            if j < len(cands):
                c = cands[j]
                b_idx = c.get("b_index")
                base[prefix + "b_index"] = b_idx
                # derive b_id if mapping available
                try:
                    if b_id_col and b_id_col in B_df.columns and b_idx is not None:
                        base[prefix + "b_id"] = str(B_df.loc[b_idx, b_id_col])
                    else:
                        base[prefix + "b_id"] = ""
                except Exception:
                    base[prefix + "b_id"] = ""
                base[prefix + "tier"] = c.get("matched_tier")
                base[prefix + "score"] = c.get("raw_score")
                sup = c.get("support", {}) or {}
                base[prefix + "date_diff_days"] = sup.get("date_diff_days")
                base[prefix + "amount_diff"] = sup.get("amount_diff")
                base[prefix + "total_diff"] = sup.get("total_diff")
                base[prefix + "email_equal"] = str(bool(sup.get("email_equal"))) if sup.get("email_equal") is not None else ""
                base[prefix + "po_equal"] = str(bool(sup.get("po_equal"))) if sup.get("po_equal") is not None else ""
                r = c.get("rationale", [])
                base[prefix + "rationale"] = " | ".join(r) if isinstance(r, list) else str(r)
                base[prefix + "tb_path"] = " > ".join(c.get("tie_breaker_path", [])) if isinstance(c.get("tie_breaker_path", []), list) else ""
            else:
                base[prefix + "b_index"] = None
                base[prefix + "b_id"] = ""
                base[prefix + "tier"] = ""
                base[prefix + "score"] = None
                base[prefix + "date_diff_days"] = None
                base[prefix + "amount_diff"] = None
                base[prefix + "total_diff"] = None
                base[prefix + "email_equal"] = ""
                base[prefix + "po_equal"] = ""
                base[prefix + "rationale"] = ""
                base[prefix + "tb_path"] = ""

        out_rows.append(base)

    import pandas as _pd
    df = _pd.DataFrame(out_rows)

    key_cols = ["a_index", "a_id", "reason"]
    cand_cols = []
    for j in range(top_k):
        prefix = f"cand{j+1}_"
        cand_cols += [
            prefix + "b_index", prefix + "b_id", prefix + "tier", prefix + "score",
            prefix + "date_diff_days", prefix + "amount_diff", prefix + "total_diff",
            prefix + "email_equal", prefix + "po_equal", prefix + "rationale", prefix + "tb_path",
        ]
    ordered = [c for c in key_cols + cand_cols if c in df.columns]
    return df[ordered]

# -------- Configuration --------
with st.expander("Configuration", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        exact_id = st.checkbox("Tier 1: Exact ID", value=True)
        digits_only = st.checkbox("Tier 2: Digits-only ID", value=True)
        prefixes = st.text_input("Strip Prefixes (comma-separated)", value="EXT-,INV-,REF-")
        suffixes = st.text_input("Strip Suffixes (comma-separated)", value="-A,-B")
        contains_token = st.checkbox("Tier 3: Token containment", value=True)
    with c2:
        fuzzy_name = st.checkbox("Tier 4: Fuzzy Name", value=True)
        fuzzy_threshold = st.slider("Fuzzy Name Threshold", 60, 100, 85)
        days_tolerance = st.number_input("Date tolerance (days)", min_value=0, value=5, step=1)
    with c3:
        amount_tolerance = st.number_input("Amount-like abs tol", min_value=0.0, value=2.0, step=0.1, format="%.2f")
        total_tolerance  = st.number_input("Total-like abs tol", min_value=0.0, value=2.0, step=0.1, format="%.2f")
        amount_pct_tol   = st.number_input("Amount-like % tol (e.g., 0.002 = 0.2%)", min_value=0.0, value=0.0, step=0.001, format="%.3f")
        total_pct_tol    = st.number_input("Total-like % tol", min_value=0.0, value=0.0, step=0.001, format="%.3f")

with st.expander("Tie-breakers", expanded=True):
    order = st.multiselect("Order", ["date","amount","total","email","po","score"], default=["date","amount","total","email","po"])

# -------- Upload --------
st.markdown("### 1) Upload CSVs")
col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Source A CSV", type=["csv"], key="file_a")
with col2:
    file_b = st.file_uploader("Source B CSV", type=["csv"], key="file_b")

def auto_guess_map(df):
    return {
        "id":    guess_column(df, "id"),
        "date":  guess_column(df, "date"),
        "amount":guess_column(df, "amount"),
        "total": guess_column(df, "total"),
        "name":  guess_column(df, "name"),
        "email": guess_column(df, "email"),
        "po":    guess_column(df, "po"),
    }

# -------- Buttons Row --------
btn_cols = st.columns([1,1,6])
with btn_cols[0]:
    if st.button("🔍 Preview & Map Columns"):
        if not file_a or not file_b:
            st.error("Please upload both CSVs first.")
        else:
            st.session_state.A_df = pd.read_csv(file_a, sep=None, engine="python")
            st.session_state.B_df = pd.read_csv(file_b, sep=None, engine="python")
            st.session_state.autoA = auto_guess_map(st.session_state.A_df)
            st.session_state.autoB = auto_guess_map(st.session_state.B_df)
            # init mapping
            st.session_state.mapping["A"] = st.session_state.autoA.copy()
            st.session_state.mapping["B"] = st.session_state.autoB.copy()
            st.session_state.show_mapping = True

with btn_cols[1]:
    run_clicked = st.button("▶️ Run Linking")

# -------- Mapping UI --------
if st.session_state.show_mapping and st.session_state.A_df is not None and st.session_state.B_df is not None:
    st.subheader("2) Column Mapping")
    st.caption("We auto-guessed the best columns. Adjust any dropdown if needed.")

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown("**Source A**")
        for key in ["id","date","amount","total","name","email","po"]:
            options = [None] + list(st.session_state.A_df.columns)
            default = st.session_state.mapping["A"].get(key)
            idx = options.index(default) if default in options else 0
            st.session_state.mapping["A"][key] = st.selectbox(f"A → {key}", options=options, index=idx, key=f"A_{key}")
        st.dataframe(st.session_state.A_df.head(), use_container_width=True)
    with mcol2:
        st.markdown("**Source B**")
        for key in ["id","date","amount","total","name","email","po"]:
            options = [None] + list(st.session_state.B_df.columns)
            default = st.session_state.mapping["B"].get(key)
            idx = options.index(default) if default in options else 0
            st.session_state.mapping["B"][key] = st.selectbox(f"B → {key}", options=options, index=idx, key=f"B_{key}")
        st.dataframe(st.session_state.B_df.head(), use_container_width=True)

# -------- Run Linking --------
if run_clicked:
    # Ensure dataframes exist; if not, read & auto-map
    if st.session_state.A_df is None or st.session_state.B_df is None:
        if not file_a or not file_b:
            st.error("Please upload both CSVs, then click Preview & Map Columns before running.")
            st.stop()
        st.session_state.A_df = pd.read_csv(file_a, sep=None, engine="python")
        st.session_state.B_df = pd.read_csv(file_b, sep=None, engine="python")
        if not st.session_state.mapping["A"]:
            st.session_state.mapping["A"] = auto_guess_map(st.session_state.A_df)
        if not st.session_state.mapping["B"]:
            st.session_state.mapping["B"] = auto_guess_map(st.session_state.B_df)

    rule_cfg = {
        "exact_id": exact_id,
        "digits_only": digits_only,
        "strip_prefixes": [x.strip() for x in prefixes.split(",") if x.strip()],
        "strip_suffixes": [x.strip() for x in suffixes.split(",") if x.strip()],
        "contains_token": contains_token,
        "fuzzy_name": fuzzy_name,
        "fuzzy_threshold": int(fuzzy_threshold),
        "amount_tolerance": float(amount_tolerance),
        "total_tolerance": float(total_tolerance),
        "days_tolerance": int(days_tolerance),
        "amount_pct_tol": float(amount_pct_tol),
        "total_pct_tol": float(total_pct_tol),
    }
    tb_cfg = {"order": [x for x in order if x != "score"]}

    result = link(st.session_state.A_df, st.session_state.B_df, rule_cfg, tb_cfg, st.session_state.mapping)

    st.markdown("### 3) Results")
    tabs = st.tabs(["✅ Matched", "🕵️ Suspects", "❌ Unmatched", "ℹ️ Meta & Mapping", "Advanced"])

    with tabs[0]:
        st.write(f"Total matched: **{len(result['matched'])}**")
        st.dataframe(_safe_display_df(result["matched"]), use_container_width=True)
        st.download_button("⬇️ Download Matched CSV", result["matched"].to_csv(index=False).encode("utf-8"),
                           "matched.csv", "text/csv")

    with tabs[1]:
        st.write(f"Total suspects: **{len(result['suspects'])}**")
        top_k = st.number_input("Show top K candidates per A-record", min_value=1, max_value=5, value=3, step=1)
        flat = _flatten_suspects(result["suspects"], st.session_state.A_df, st.session_state.B_df, st.session_state.mapping, int(top_k))
        st.dataframe(_safe_display_df(flat), use_container_width=True)
        st.download_button("⬇️ Download Suspects (flattened) CSV", flat.to_csv(index=False).encode("utf-8"),
                           "suspects_flattened.csv", "text/csv")

    with tabs[2]:
        st.write(f"Total unmatched: **{len(result['unmatched'])}**")
        st.dataframe(_safe_display_df(result["unmatched"]), use_container_width=True)

    with tabs[3]:
        st.json(result["meta"])
        st.download_button("⬇️ Download Full JSON", data=io.BytesIO(json.dumps({
            "matched": result["matched"].to_dict(orient="records"),
            "suspects": result["suspects"].to_dict(orient="records"),
            "unmatched": result["unmatched"].to_dict(orient="records"),
            "meta": result["meta"],
        }, indent=2).encode("utf-8")), file_name="link_results.json", mime="application/json")

    with tabs[4]:
        st.caption("Raw suspects with nested objects (for debugging):")
        st.dataframe(_safe_display_df(result["suspects"]), use_container_width=True)
