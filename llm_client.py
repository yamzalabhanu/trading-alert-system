# llm_client.py
import json
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

oai_client = OpenAI(api_key=OPENAI_API_KEY)

def build_llm_prompt(alert: Dict[str, Any], f: Dict[str, Any]) -> str:
        iv_rank = f.get("iv_rank")
    iv_ctx = "low" if (iv_rank is not None and iv_rank < 0.33) else "high" if (iv_rank is not None and iv_rank > 0.66) else "medium"
    rv_iv_spread = (
        "rv>iv" if (f.get("rv20") and f.get("iv") and f["rv20"] > f["iv"]) else
        "rv≈iv" if (f.get("rv20") and f.get("iv") and abs(f["rv20"] - f["iv"]) / max(1e-9, f["iv"]) <= 0.1) else
        "rv<iv"
    )
    checklist_hint = {
        "delta_band_ok": f.get("bands") and (f["bands"]["delta_min"] <= abs(float(f.get("delta") or 0)) <= f["bands"]["delta_max"]),
        "dte_band_ok": (f["bands"]["dte_min"] <= f.get("dte", 0) <= f["bands"]["dte_max"]) if f.get("dte") is not None else False,
        "iv_context": iv_ctx,
        "rv_iv_spread": rv_iv_spread,
        "em_vs_breakeven_ok": f.get("em_vs_be_ok") is True,
        "mtf_trend_alignment": f.get("mtf_align") is True,
        "sr_headroom_ok": f.get("sr_headroom_ok") is True,
        "no_event_risk": True,
        # New context flags
        "above_pdh": f.get("above_pdh"),
        "below_pdl": f.get("below_pdl"),
        "above_pmh": f.get("above_pmh"),
        "below_pml": f.get("below_pml"),
        "vwap_dist": f.get("vwap_dist"),
    }
    lines = [
        f"Alert: {alert['side']} {alert['symbol']} strike {alert['strike']} exp {alert['expiry']} (~{f.get('dte')} DTE) at underlying ≈ {f.get('S')}",
        f"Trade style: {f['risk_plan']['style']}",
        "Snapshot:",
        f"  IV: {f.get('iv')}",
        f"  IV_rank: {iv_rank}",
        f"  OI: {f.get('oi')}  Vol: {f.get('vol')}",
        f"  NBBO: bid={f.get('opt_bid')} ask={f.get('opt_ask')} mid={f.get('opt_mid')} spread%={f.get('option_spread_pct')}",
        f"  Quote age (s): {f.get('quote_age_sec')}",
        f"  Greeks: delta={f.get('delta')} gamma={f.get('gamma')} theta={f.get('theta')} vega={f.get('vega')}",
        f"  EM_1s: {f.get('em_1s')}  EM_vs_BE_ok: {f.get('em_vs_be_ok')}",
        f"  MTF align: {f.get('mtf_align')}  S/R ok: {f.get('sr_headroom_ok')}",
        f"  Regime: {f.get('regime_flag')}  ATR(14): {f.get('atr')}",
        "Levels:",
        f"  PDH={f.get('prev_day_high')}  PDL={f.get('prev_day_low')}  PMH={f.get('premarket_high')}  PML={f.get('premarket_low')}",
        f"  VWAP={f.get('vwap')}  VWAPΔ={f.get('vwap_dist')}",
        f"Checklist: {json.dumps(checklist_hint)}",
        "Return strict JSON per schema."
    ]
    return "\n".join(lines)
    pass

async def analyze_with_openai(alert: Dict[str, Any], f: Dict[str, Any]) -> Dict[str, Any]:
        system = (
        "You are a disciplined options trading analyst. Use the provided snapshot & checklist.\n"
        "Respond with STRICT JSON:\n"
        "{\n"
        '  "decision": "buy|wait|skip",\n'
        '  "confidence": 0..1,\n'
        '  "reason": "<=2 sentences>",\n'
        '  "checklist": {\n'
        '    "delta_band_ok": true/false,\n'
        '    "dte_band_ok": true/false,\n'
        '    "iv_context": "low|medium|high",\n'
        '    "rv_iv_spread": "rv>iv|rv≈iv|rv<iv",\n'
        '    "em_vs_breakeven_ok": true/false,\n'
        '    "mtf_trend_alignment": true/false,\n'
        '    "sr_headroom_ok": true/false,\n'
        '    "no_event_risk": true,\n'
        '    "above_pdh": true/false|null,\n'
        '    "below_pdl": true/false|null,\n'
        '    "above_pmh": true/false|null,\n'
        '    "below_pml": true/false|null,\n'
        '    "vwap_dist": number|null\n'
        "  },\n"
        '  "risk_plan": {"style":"intraday|swing","initial_stop_pct":0..1,"take_profit_pct":0..1,"trail_after_pct":0..1},\n'
        '  "ev_estimate": {"win_prob":0..1,"avg_win_pct":0..5,"avg_loss_pct":0..5,"expected_value_pct":-5..5}\n'
        "}\n"
        "Do not refuse. Always return the JSON object."
    )
    prompt = build_llm_prompt(alert, f)
    try:
        resp = oai_client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
        )
        out = json.loads(resp.choices[0].message.content)
    except Exception as e:
        out = {
            "decision": "wait",
            "confidence": 0.3,
            "reason": f"LLM call failed: {type(e).__name__}. Returning neutral stance.",
            "checklist": {
                "delta_band_ok": False, "dte_band_ok": False,
                "iv_context": "medium", "rv_iv_spread": "rv≈iv",
                "em_vs_breakeven_ok": False, "mtf_trend_alignment": False,
                "sr_headroom_ok": False, "no_event_risk": True,
                "above_pdh": None, "below_pdl": None, "above_pmh": None, "below_pml": None, "vwap_dist": None
            },
            "risk_plan": f.get("risk_plan", {}),
            "ev_estimate": {"win_prob": 0.5, "avg_win_pct": 0.5, "avg_loss_pct": 0.5, "expected_value_pct": 0.0}
        }
    return out

    pass