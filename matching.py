
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime
from rapidfuzz import fuzz

# Heuristic tokens for column auto-detection
CANDIDATE_KEYS = {
    "id":        ["invoice", "inv", "ref", "code", "id", "reference", "doc", "number"],
    "date":      ["date", "doc_date", "invoice_date", "txn", "created"],
    "amount":    ["amount", "net", "subtotal", "base"],
    "total":     ["total", "grand", "gross"],
    "name":      ["name", "client", "customer", "account"],
    "email":     ["email", "mail"],
    "po":        ["po", "purchase", "order"],
    "currency":  ["currency", "ccy", "curr"],
}

def guess_column(df: pd.DataFrame, kind: str) -> Optional[str]:
    """Pick a column by keyword containment; fallback to fuzzy scoring."""
    if df is None or df.empty:
        return None
    candidates = CANDIDATE_KEYS.get(kind, [])
    cols = list(df.columns)

    # direct contains (case-insensitive)
    for token in candidates:
        for c in cols:
            if token.lower() in str(c).lower():
                return c

    # fallback: rapidfuzz best token_set_ratio
    best_col, best_score = None, -1
    for c in cols:
        for token in candidates:
            score = fuzz.token_set_ratio(str(c).lower(), token.lower())
            if score > best_score:
                best_score, best_col = score, c
    return best_col

def to_date(x):
    if pd.isna(x):
        return None
    fmts = ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%y", "%Y-%m-%d %H:%M:%S")
    for fmt in fmts:
        try:
            return datetime.strptime(str(x), fmt).date()
        except Exception:
            continue
    try:
        return pd.to_datetime(x, errors="coerce").date()
    except Exception:
        return None

def to_float(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x).replace(",", "").strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def only_digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")

@dataclass
class RuleConfig:
    # tiers
    exact_id: bool = True
    digits_only: bool = True
    strip_prefixes: List[str] = field(default_factory=lambda: ["EXT-", "INV-", "REF-"])
    strip_suffixes: List[str] = field(default_factory=lambda: ["-A", "-B"])
    contains_token: bool = True
    fuzzy_name: bool = True
    fuzzy_threshold: int = 85
    # tolerances
    amount_tolerance: float = 2.0
    total_tolerance: float = 2.0
    days_tolerance: int = 5
    amount_pct_tol: float = 0.0
    total_pct_tol: float = 0.0

@dataclass
class TieBreakers:
    order: List[str] = field(default_factory=lambda: ["date", "amount", "total", "email", "po"])

def normalize_string(x: Any) -> str:
    return str(x).strip().lower() if x is not None and not pd.isna(x) else ""

def strip_patterns(s: str, prefixes: List[str], suffixes: List[str]) -> str:
    out = s or ""
    for p in prefixes:
        if out.startswith(p):
            out = out[len(p):]
    for suf in suffixes:
        if out.endswith(suf):
            out = out[: -len(suf)]
    return out

def within_numeric(a_val: Optional[float], b_val: Optional[float], abs_tol: float, pct_tol: float) -> Tuple[bool, Optional[float]]:
    if a_val is None or b_val is None:
        return False, None
    diff = abs(a_val - b_val)
    if diff <= abs_tol:
        return True, diff
    if pct_tol > 0:
        base = max(abs(a_val), 1e-9)
        if diff <= base * pct_tol:
            return True, diff
    return False, diff

def date_close(a_date, b_date, days: int) -> Tuple[bool, Optional[int]]:
    if a_date is None or b_date is None:
        return False, None
    dd = abs((a_date - b_date).days)
    return dd <= days, dd

def prepare_frames(a: pd.DataFrame, b: pd.DataFrame, mapping: Dict[str, Dict[str, Optional[str]]]) -> Dict[str, Any]:
    """Prepare frames using provided mapping; fallback to auto-guessing."""
    def choose(df, key, side):
        col = mapping.get(side, {}).get(key, None)
        if col is None:
            col = guess_column(df, key)
        return col

    A = a.copy()
    B = b.copy()

    ida = choose(A, "id", "A") or A.columns[0]
    idb = choose(B, "id", "B") or B.columns[0]
    da  = choose(A, "date", "A")
    db  = choose(B, "date", "B")
    aa  = choose(A, "amount", "A")
    ab  = choose(B, "amount", "B")
    ta  = choose(A, "total", "A")
    tb  = choose(B, "total", "B")
    na  = choose(A, "name", "A")
    nb  = choose(B, "name", "B")
    ea  = choose(A, "email", "A")
    eb  = choose(B, "email", "B")
    poa = choose(A, "po", "A")
    pob = choose(B, "po", "B")

    A["_id_norm"] = A[ida].astype(str)
    B["_id_norm"] = B[idb].astype(str)
    A["_id_digits"] = A["_id_norm"].apply(only_digits)
    B["_id_digits"] = B["_id_norm"].apply(only_digits)

    A["_date"] = A[da].apply(to_date) if (da and da in A.columns) else None
    B["_date"] = B[db].apply(to_date) if (db and db in B.columns) else None

    A["_amount_like"] = A[aa].apply(to_float) if (aa and aa in A.columns) else None
    B["_amount_like"] = B[ab].apply(to_float) if (ab and ab in B.columns) else None
    A["_total_like"]  = A[ta].apply(to_float) if (ta and ta in A.columns) else None
    B["_total_like"]  = B[tb].apply(to_float) if (tb and tb in B.columns) else None

    A["_name"]  = A[na].apply(normalize_string) if (na and na in A.columns) else ""
    B["_name"]  = B[nb].apply(normalize_string) if (nb and nb in B.columns) else ""
    A["_email"] = A[ea].apply(normalize_string) if (ea and ea in A.columns) else ""
    B["_email"] = B[eb].apply(normalize_string) if (eb and eb in B.columns) else ""
    A["_po"]    = A[poa].apply(normalize_string) if (poa and poa in A.columns) else ""
    B["_po"]    = B[pob].apply(normalize_string) if (pob and pob in B.columns) else ""

    return dict(a=A, b=B, ida=ida, idb=idb, da=da, db=db, aa=aa, ab=ab, ta=ta, tb=tb,
                na=na, nb=nb, ea=ea, eb=eb, poa=poa, pob=pob)

def apply_rules_for(a_row: pd.Series, b_df: pd.DataFrame, cfg: RuleConfig) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    a_id = str(a_row.get("_id_norm", ""))
    a_digits = str(a_row.get("_id_digits", ""))
    a_stripped = strip_patterns(a_id, cfg.strip_prefixes, cfg.strip_suffixes)
    a_name = a_row.get("_name", "")
    a_email = a_row.get("_email", "")
    a_po = a_row.get("_po", "")

    for idx, b_row in b_df.iterrows():
        rationale = []
        score = 0
        matched_tier = None

        if cfg.exact_id and a_id == b_row.get("_id_norm", ""):
            matched_tier = "Tier 1 - exact_id"
            score += 100
            rationale.append("Exact ID match")

        if matched_tier is None and cfg.digits_only and a_digits and a_digits == b_row.get("_id_digits", ""):
            matched_tier = "Tier 2 - digits_only"
            score += 85
            rationale.append("Digits-only ID match")

        b_stripped = strip_patterns(b_row.get("_id_norm", ""), cfg.strip_prefixes, cfg.strip_suffixes)
        if matched_tier is None and (a_stripped and b_stripped):
            if a_stripped == b_stripped:
                matched_tier = "Tier 3 - stripped_equal"
                score += 80
                rationale.append("IDs equal after stripping patterns")
            elif cfg.contains_token and (a_stripped in b_stripped or b_stripped in a_stripped):
                matched_tier = "Tier 3 - token_contains"
                score += 70
                rationale.append("ID token containment after stripping")

        ok_amt, amt_diff = within_numeric(a_row.get("_amount_like"), b_row.get("_amount_like"),
                                          cfg.amount_tolerance, cfg.amount_pct_tol)
        if ok_amt:
            score += 10
            rationale.append(f"Amount-like within tol (diff={amt_diff:.2f})")

        ok_tot, tot_diff = within_numeric(a_row.get("_total_like"), b_row.get("_total_like"),
                                          cfg.total_tolerance, cfg.total_pct_tol)
        if ok_tot:
            score += 10
            rationale.append(f"Total-like within tol (diff={tot_diff:.2f})")

        ok_date, date_diff = date_close(a_row.get("_date"), b_row.get("_date"), cfg.days_tolerance)
        if ok_date:
            score += 8
            rationale.append(f"Date within tol (diff_days={date_diff})")

        if a_email and a_email == b_row.get("_email", ""):
            score += 6
            rationale.append("Email exact match")
        if a_po and a_po == b_row.get("_po", ""):
            score += 4
            rationale.append("PO exact match")

        if matched_tier is None and cfg.fuzzy_name and a_name and b_row.get("_name", ""):
            sim = fuzz.token_set_ratio(a_name, b_row.get("_name", ""))
            if sim >= cfg.fuzzy_threshold:
                matched_tier = "Tier 4 - fuzzy_name"
                score += 50 + (sim / 10.0)
                rationale.append(f"Fuzzy name match (score={sim})")

        if matched_tier is not None:
            cands.append({
                "b_index": idx,
                "matched_tier": matched_tier,
                "raw_score": score,
                "rationale": rationale,
                "support": {
                    "amount_diff": amt_diff if ok_amt else None,
                    "total_diff":  tot_diff if ok_tot else None,
                    "date_diff_days": date_diff if ok_date else None,
                    "email_equal": a_email and a_email == b_row.get("_email", ""),
                    "po_equal": a_po and a_po == b_row.get("_po", ""),
                }
            })

    cands.sort(key=lambda x: x["raw_score"], reverse=True)
    return cands

def break_ties(cands: List[Dict[str, Any]], order: List[str]) -> Dict[str, Any]:
    if not cands:
        return {}
    if len(cands) == 1:
        cands[0]["tie_breaker_path"] = ["single_candidate"]
        return cands[0]
    remaining = cands.copy()
    path = []
    def sort_by(key):
        if key == "date":
            remaining.sort(key=lambda x: (x["support"].get("date_diff_days") is None, x["support"].get("date_diff_days", 9999)))
            path.append("date_diff_days")
        elif key == "amount":
            remaining.sort(key=lambda x: (x["support"].get("amount_diff") is None, x["support"].get("amount_diff", 1e9)))
            path.append("amount_diff")
        elif key == "total":
            remaining.sort(key=lambda x: (x["support"].get("total_diff") is None, x["support"].get("total_diff", 1e9)))
            path.append("total_diff")
        elif key == "email":
            remaining.sort(key=lambda x: (not bool(x["support"].get("email_equal")), 0))
            path.append("email_equal")
        elif key == "po":
            remaining.sort(key=lambda x: (not bool(x["support"].get("po_equal")), 0))
            path.append("po_equal")
        elif key == "score":
            remaining.sort(key=lambda x: x["raw_score"], reverse=True)
            path.append("raw_score")
    for key in order + ["score"]:
        sort_by(key)
        best = remaining[0]
        if key == "date":
            v = best["support"].get("date_diff_days", None)
            remaining = [c for c in remaining if c["support"].get("date_diff_days", None) == v]
        elif key == "amount":
            v = best["support"].get("amount_diff", None)
            remaining = [c for c in remaining if c["support"].get("amount_diff", None) == v]
        elif key == "total":
            v = best["support"].get("total_diff", None)
            remaining = [c for c in remaining if c["support"].get("total_diff", None) == v]
        elif key == "email":
            v = best["support"].get("email_equal", False)
            remaining = [c for c in remaining if c["support"].get("email_equal", False) == v]
        elif key == "po":
            v = best["support"].get("po_equal", False)
            remaining = [c for c in remaining if c["support"].get("po_equal", False) == v]
        elif key == "score":
            v = best["raw_score"]
            remaining = [c for c in remaining if c["raw_score"] == v]
        if len(remaining) == 1:
            remaining[0]["tie_breaker_path"] = path
            return remaining[0]
    remaining[0]["tie_breaker_path"] = path + ["fallback_first"]
    return remaining[0]

def link(a: pd.DataFrame, b: pd.DataFrame, rule_cfg: Dict[str, Any], tb_cfg: Dict[str, Any], mapping: Dict[str, Dict[str, Optional[str]]]) -> Dict[str, Any]:
    cfg = RuleConfig(**rule_cfg)
    meta = prepare_frames(a, b, mapping)
    A, B = meta["a"], meta["b"]
    order = tb_cfg.get("order", ["date","amount","total","email","po"])

    matches, suspects, unmatched = [], [], []
    used_b = set()

    for i, a_row in A.iterrows():
        cands = apply_rules_for(a_row, B, cfg)
        if not cands:
            unmatched.append({"a_index": i, "reason": "no_candidates"})
            continue
        chosen = break_ties(cands, order)
        if not chosen:
            suspects.append({"a_index": i, "candidates": cands, "reason": "tie_unresolved"})
            continue
        b_idx = chosen["b_index"]
        if b_idx in used_b:
            suspects.append({"a_index": i, "candidates": cands, "reason": "b_conflict"})
            continue
        used_b.add(b_idx)
        matches.append({
            "a_index": i,
            "b_index": b_idx,
            "matched_tier": chosen["matched_tier"],
            "raw_score": chosen["raw_score"],
            "rationale": chosen["rationale"],
            "tie_breaker_path": chosen.get("tie_breaker_path", []),
        })

    def row(df, idx):
        return df.iloc[idx] if (idx is not None and 0 <= idx < len(df)) else None

    matched_rows = []
    for rec in matches:
        a_row = row(A, rec["a_index"])
        b_row = row(B, rec["b_index"])
        out = {
            "a_index": rec["a_index"],
            "b_index": rec["b_index"],
            "a_id": a_row.get("_id_norm", ""),
            "b_id": b_row.get("_id_norm", ""),
            "match_status": "matched",
            "matched_by": rec["matched_tier"],
            "raw_score": rec["raw_score"],
            "rationale": " | ".join(rec["rationale"]),
            "tie_breaker_path": " > ".join(rec["tie_breaker_path"]),
        }
        try:
            out["date_diff_days"] = abs((a_row.get("_date") - b_row.get("_date")).days) if (a_row.get("_date") and b_row.get("_date")) else None
        except Exception:
            out["date_diff_days"] = None
        try:
            out["amount_like_diff"] = abs(a_row.get("_amount_like") - b_row.get("_amount_like")) if (a_row.get("_amount_like") is not None and b_row.get("_amount_like") is not None) else None
        except Exception:
            out["amount_like_diff"] = None
        try:
            out["total_like_diff"] = abs(a_row.get("_total_like") - b_row.get("_total_like")) if (a_row.get("_total_like") is not None and b_row.get("_total_like") is not None) else None
        except Exception:
            out["total_like_diff"] = None
        matched_rows.append(out)

    return {
        "matched": pd.DataFrame(matched_rows),
        "suspects": pd.DataFrame(suspects),
        "unmatched": pd.DataFrame(unmatched),
        "meta": {
            "mapping_used": mapping,
            "config": rule_cfg,
            "tiebreakers": tb_cfg,
        }
    }
