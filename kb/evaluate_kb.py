import json
import os
import sys


"""
Valutazione minima della Knowledge Base Prolog.

Lo script esegue casi rappresentativi per verificare che le regole simboliche
distinguano correttamente scenari diversi: dominio storico confermato, tradizione
con probabilità incerta, nuova potenza e sfida a bassa evidenza.
"""


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(CURRENT_DIR, "rules.pl")
REPORT_DIR = os.path.join(CURRENT_DIR, "report")


TEST_CASES = [
    {"probability": 0.84, "noc": "ITA", "sport": "Fencing", "expected_keyword": "SUCCESSO ATTESO"},
    {"probability": 0.52, "noc": "USA", "sport": "Swimming", "expected_keyword": "TRADIZIONE COMPETITIVA"},
    {"probability": 0.78, "noc": "CAN", "sport": "Rowing", "expected_keyword": "NUOVA POTENZA"},
    {"probability": 0.22, "noc": "ITA", "sport": "Basketball", "expected_keyword": "SFIDA ESTREMA"},
]


def prolog_atom(value):
    """
    Converte una stringa Python in un atomo Prolog sicuro.
    """
    escaped = str(value).replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


def decode_value(value):
    """
    Normalizza i valori restituiti da PySwip.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    return str(value)


def run_case(prolog, case):
    """
    Esegue un caso di test sulla KB e restituisce esito e motivazioni.
    """
    probability = f"{case['probability']:.6f}"
    noc = prolog_atom(case["noc"])
    sport = prolog_atom(case["sport"])

    advice_query = f"olympic_advice({probability}, {noc}, {sport}, Advice)"
    reasons_query = f"explain_verdict({probability}, {noc}, {sport}, Reasons)"

    advice_result = list(prolog.query(advice_query))
    reasons_result = list(prolog.query(reasons_query))

    advice = decode_value(advice_result[0]["Advice"]) if advice_result else ""
    reasons = decode_value(reasons_result[0]["Reasons"]) if reasons_result else []
    passed = case["expected_keyword"] in advice

    return {
        **case,
        "advice": advice,
        "reasons": reasons,
        "passed": passed,
    }


def main():
    """
    Produce un report JSON con i casi di valutazione della KB.
    """
    try:
        from pyswip import Prolog
    except Exception as exc:
        print(f"[!] PySwip/SWI-Prolog non disponibile: {exc}")
        sys.exit(1)

    os.makedirs(REPORT_DIR, exist_ok=True)
    prolog = Prolog()
    prolog.consult(RULES_PATH)

    results = [run_case(prolog, case) for case in TEST_CASES]
    summary = {
        "total_cases": len(results),
        "passed_cases": sum(1 for result in results if result["passed"]),
        "failed_cases": sum(1 for result in results if not result["passed"]),
        "cases": results,
    }

    output_path = os.path.join(REPORT_DIR, "valutazione_kb.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    print(f"[OK] Valutazione KB salvata in {output_path}")
    print(f"[OK] Casi superati: {summary['passed_cases']}/{summary['total_cases']}")


if __name__ == "__main__":
    main()
