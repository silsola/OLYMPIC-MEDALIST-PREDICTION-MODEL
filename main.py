import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
from pyswip import Prolog
from thefuzz import process 

warnings.filterwarnings("ignore", category=UserWarning)

def get_input_validato(messaggio, tipo, range_val=None, opzioni=None):
    while True:
        try:
            valore = input(messaggio).strip()
            if tipo == "sex":
                valore = valore.upper()
                if valore in ['M', 'F']: return valore
                print("[!] Inserisci 'M' o 'F'.\n")
            elif tipo == "numeric":
                num = float(valore)
                if range_val[0] <= num <= range_val[1]: return num
                print(f"[!] Valore fuori range ({range_val[0]}-{range_val[1]}).\n")
            elif tipo == "noc":
                valore = valore.upper()
                if valore in opzioni: return valore
                print(f"[!] Codice NOC '{valore}' non riconosciuto dal dataset.\n")
            elif tipo == "sport":
                scelta, score = process.extractOne(valore, opzioni)
                if score == 100: return scelta
                if score > 70:
                    conferma = input(f"[?] Intendevi '{scelta}'? (S/N): ").strip().upper()
                    if conferma == 'S': return scelta
                print(f"[!] Sport non trovato. Riprova.\n")
        except ValueError:
            print("[!] Errore: Inserisci un valore numerico valido.\n")


def load_resources():
    base_path = os.path.dirname(os.path.abspath(__file__))
    path_supervisionato = os.path.join(base_path, 'apprendimento_supervisionato', 'modelli')
    path_probabilistico = os.path.join(base_path, 'apprendimento_probabilistico', 'modelli')
    try:
        return {
            "rf_model": joblib.load(os.path.join(path_supervisionato, 'random_forest.pkl')),
            "scaler": joblib.load(os.path.join(path_supervisionato, 'scaler.pkl')),
            "sport_map": joblib.load(os.path.join(path_supervisionato, 'sport_mapping.pkl')),
            "noc_map": joblib.load(os.path.join(path_supervisionato, 'noc_mapping.pkl')),
            "nb_model": joblib.load(os.path.join(path_probabilistico, 'naive_bayes.pkl')),
            "sport_cols_nb": joblib.load(os.path.join(path_probabilistico, 'sport_columns_nb.pkl'))
        }
    except FileNotFoundError as e:
        print(f"[!] Errore: Uno o più file non trovati.\nSpecifiche: {e}")
        sys.exit()


def get_olympic_advice_detailed(prob_ml, noc, sport):
    """
    Interroga la KB Prolog per ottenere sia il verdetto che le spiegazioni (XAI).
    """
    prolog = Prolog()
    kb_path = "kb/rules.pl"
    if not os.path.exists(kb_path): 
        return "Conoscenza Prolog non trovata.", []
    
    try:
        prolog.consult(kb_path)
        p_val = round(float(prob_ml), 2)
        
        query_advice = f"olympic_advice({p_val}, '{noc}', '{sport}', Advice)"
        res_advice = list(prolog.query(query_advice))
        
        query_explain = f"explain_verdict({p_val}, '{noc}', '{sport}', Reasons)"
        res_explain = list(prolog.query(query_explain))
        
        advice = "Nessun consiglio disponibile."
        reasons = []

        if res_advice:
            raw_advice = res_advice[0]['Advice']
            advice = raw_advice.decode('utf-8') if isinstance(raw_advice, bytes) else raw_advice
        
        if res_explain:
            raw_reasons = res_explain[0]['Reasons']
            reasons = [r.decode('utf-8') if isinstance(r, bytes) else r for r in raw_reasons]
            
        return advice, reasons

    except Exception as e:
        return f"Errore Prolog: {e}", []


def find_sector_leader(resources, user_sport_str):
    model = resources['nb_model']
    cols = resources['sport_cols_nb']
    if user_sport_str not in cols: return "N/D", 0.0
    input_vec = np.zeros((1, len(cols)))
    input_vec[0, cols.index(user_sport_str)] = 1
    probs = model.predict_proba(input_vec)[0]
    best_idx = np.argmax(probs)
    return model.classes_[best_idx], probs[best_idx]


def main():
    res = load_resources()
    print("\n" + "═"*60)
    print("            SISTEMA DI PREDIZIONE OLIMPICA       ")
    print("═"*60)

    lista_noc = list(res['noc_map'].values())
    lista_sport = list(res['sport_map'].values())

    u_sex_raw = get_input_validato(" Sesso (M/F): ", "sex")
    u_noc_str = get_input_validato(" Codice Nazione (es. ITA, USA): ", "noc", opzioni=lista_noc)
    u_sport_str = get_input_validato(" Sport (es. Basketball): ", "sport", opzioni=lista_sport)

    sport_to_id = {v: k for k, v in res['sport_map'].items()}
    noc_to_id = {v: k for k, v in res['noc_map'].items()}
    
    u_sex = 0 if u_sex_raw == 'M' else 1
    u_sport_id = sport_to_id[u_sport_str]
    u_noc_id = noc_to_id[u_noc_str]

    atleta_df = pd.DataFrame([[u_sex, u_noc_id, u_sport_id]], columns=['Sex', 'NOC', 'Sport'])
    atleta_scaled = res['scaler'].transform(atleta_df)
    prob_atleta = res['rf_model'].predict_proba(atleta_scaled)[0][1]

    leader_noc, dominanza = find_sector_leader(res, u_sport_str)
    advice, reasons = get_olympic_advice_detailed(prob_atleta, u_noc_str, u_sport_str)

    print("\n" + "─"*60)
    print(f" RISULTATI PREVISIONE:  {prob_atleta:.2%} di probabilità podio.")
    print(f"\n ANALISI DI SETTORE ({u_sport_str.upper()}):")
    print(f" Leader storico: {leader_noc} (Dominanza: {dominanza:.2%})")
    
    print("\n" + "─"*60)
    print(" CONSIGLIO DELL'ESPERTO:")
    print(f" {advice}")
    
    if reasons:
        print("\n MOTIVAZIONI LOGICHE:")
        for r in reasons:
            print(f"  • {r}")
    
    print("═"*60 + "\n")

if __name__ == "__main__":
    main()