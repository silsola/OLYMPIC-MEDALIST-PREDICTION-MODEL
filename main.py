import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
from thefuzz import process 

"""
Modulo principale del Sistema di Predizione Olimpica.
Questo script funge da orchestratore tra i modelli di apprendimento supervisionato
(Random Forest e Gradient Boosting), probabilistico (Naive Bayes) e la Base di Conoscenza 
(KB) in Prolog. Gestisce l'interazione con l'utente tramite interfaccia testuale.
"""

warnings.filterwarnings("ignore", category=UserWarning)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def _normalizza_mapping(mapping):
    """
    Converte mapping caricati da joblib in dizionari Python standard.
    """
    return {int(k): str(v) for k, v in mapping.items()}


def _decode_prolog_value(value):
    """
    Normalizza i valori restituiti da PySWIP in stringhe/liste Python leggibili.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [_decode_prolog_value(item) for item in value]
    return value


def _prolog_atom(value):
    """
    Crea un atomo Prolog quotato in modo sicuro.
    """
    escaped = str(value).replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


def get_input_validato(messaggio, tipo, opzioni=None):
    """
    Gestisce l'input dell'utente garantendo la correttezza formale e semantica dei dati.
    """
    opzioni = [str(opzione) for opzione in opzioni] if opzioni else []

    while True:
        valore = input(messaggio).strip()

        if tipo == "sex":
            valore = valore.upper()
            if valore in ["M", "F"]:
                return valore
            print("[!] Inserisci 'M' o 'F'.\n")

        elif tipo == "noc":
            valore = valore.upper()
            if valore in opzioni:
                return valore
            print(f"[!] Codice NOC '{valore}' non riconosciuto dal dataset.\n")

        elif tipo == "sport":
            if not valore:
                print("[!] Inserisci il nome di uno sport.\n")
                continue

            match = process.extractOne(valore, opzioni)
            if not match:
                print("[!] Nessuno sport disponibile per la validazione.\n")
                continue

            scelta, score = match
            if score == 100:
                return scelta
            if score > 70:
                conferma = input(f"[?] Intendevi '{scelta}'? (S/N): ").strip().upper()
                if conferma == "S":
                    return scelta
            print("[!] Sport non trovato. Riprova.\n")


def load_resources():
    """
    Carica i modelli serializzati (incluso il Gradient Boosting) e i file di supporto.
    """
    path_supervisionato = os.path.join(BASE_PATH, 'apprendimento_supervisionato', 'modelli')
    path_probabilistico = os.path.join(BASE_PATH, 'apprendimento_probabilistico', 'modelli')
    kb_path = os.path.join(BASE_PATH, 'kb', 'rules.pl')
    try:
        return {
            "rf_model": joblib.load(os.path.join(path_supervisionato, 'random_forest.pkl')),
            "gb_model": joblib.load(os.path.join(path_supervisionato, 'gradient_boosting.pkl')), # <--- Caricamento Gradient Boosting
            "scaler": joblib.load(os.path.join(path_supervisionato, 'scaler.pkl')),
            "sport_map": _normalizza_mapping(joblib.load(os.path.join(path_supervisionato, 'sport_mapping.pkl'))),
            "noc_map": _normalizza_mapping(joblib.load(os.path.join(path_supervisionato, 'noc_mapping.pkl'))),
            "history_profile": joblib.load(os.path.join(path_supervisionato, 'history_profile.pkl')),
            "nb_model": joblib.load(os.path.join(path_probabilistico, 'naive_bayes.pkl')),
            "sport_cols_nb": [str(col) for col in joblib.load(os.path.join(path_probabilistico, 'sport_columns_nb.pkl'))],
            "kb_path": kb_path
        }
    except FileNotFoundError as e:
        print(f"[!] Errore: Uno o più file non trovati.\nSpecifiche: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[!] Errore durante il caricamento delle risorse: {e}")
        sys.exit(1)


def get_olympic_advice_detailed(prob_ml, noc, sport, kb_path):
    """
    Interroga la Knowledge Base (KB) in Prolog per generare spiegazioni simboliche.
    """
    if not os.path.exists(kb_path): 
        return "Conoscenza Prolog non trovata.", []
    
    try:
        from pyswip import Prolog

        prolog = Prolog()
        prolog.consult(kb_path)
        p_val = float(prob_ml)
        p_literal = f"{p_val:.6f}"
        noc_atom = _prolog_atom(noc)
        sport_atom = _prolog_atom(sport)
        
        query_advice = f"olympic_advice({p_literal}, {noc_atom}, {sport_atom}, Advice)"
        res_advice = list(prolog.query(query_advice))
        
        query_explain = f"explain_verdict({p_literal}, {noc_atom}, {sport_atom}, Reasons)"
        res_explain = list(prolog.query(query_explain))
        
        advice = "Nessun consiglio disponibile."
        reasons = []

        if res_advice:
            advice = _decode_prolog_value(res_advice[0]['Advice'])
        
        if res_explain:
            reasons = _decode_prolog_value(res_explain[0]['Reasons'])
            
        return advice, reasons

    except Exception as e:
        return f"Errore Prolog: {e}", []


def find_sector_leader(resources, user_sport_str):
    """
    Utilizza il modello Naive Bayes per identificare la nazione dominante in uno sport.
    """
    model = resources['nb_model']
    cols = resources['sport_cols_nb']
    if user_sport_str not in cols:
        return "N/D", 0.0

    input_vec = np.zeros((1, len(cols)))
    input_vec[0, cols.index(user_sport_str)] = 1
    probs = model.predict_proba(input_vec)[0]
    best_idx = np.argmax(probs)
    return str(model.classes_[best_idx]), float(probs[best_idx])


def predict_medal_probability(model, scaled_data):
    """
    Restituisce la probabilità associata alla classe positiva 'ha vinto medaglia'.
    """
    probabilities = model.predict_proba(scaled_data)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Il modello non contiene la classe positiva 1 (Won_Medal).")

    medal_idx = classes.index(1)
    return float(probabilities[medal_idx])


def get_historical_context(resources, noc_id, sport_id):
    """
    Recupera le feature storiche più adatte per la coppia NOC-Sport.
    """
    profile = resources["history_profile"]
    key = (int(noc_id), int(sport_id))

    if key in profile["by_noc_sport"]:
        return profile["by_noc_sport"][key], "profilo storico NOC+Sport"
    if int(noc_id) in profile["by_noc"]:
        return profile["by_noc"][int(noc_id)], "profilo storico NOC"
    if int(sport_id) in profile["by_sport"]:
        return profile["by_sport"][int(sport_id)], "profilo storico Sport"
    return profile["defaults"], "profilo storico medio"


def build_athlete_dataframe(resources, sex_value, noc_id, sport_id):
    """
    Costruisce il vettore di input con feature categoriche e storiche.
    """
    history_features, source = get_historical_context(resources, noc_id, sport_id)
    row = {
        "Sex": int(sex_value),
        "NOC": int(noc_id),
        "Sport": int(sport_id),
        **history_features,
    }
    feature_columns = resources["history_profile"]["feature_columns"]
    return pd.DataFrame([[row[column] for column in feature_columns]], columns=feature_columns), source


def print_results(prob_rf, prob_gb, sport, leader_noc, dominanza, history_source, advice, reasons):
    """
    Stampa l'analisi finale includendo i risultati comparativi di entrambi i modelli avanzati.
    """
    print("\n" + "─"*60)
    print(" STIME DI PROBABLITÀ PODIO (MODELLI DI MACHINE LEARNING):")
    print(f"  • Random Forest Classifier:   {prob_rf:.2%}")
    print(f"  • Gradient Boosting Classifier: {prob_gb:.2%}")
    
    print(f"\n ANALISI DI SETTORE ({sport.upper()}):")
    print(f"  Leader storico/probabilistico: {leader_noc} (Dominanza: {dominanza:.2%})")
    print(f"  Contesto usato dal modello:    {history_source}")
    
    print("\n" + "─"*60)
    print(" CONSIGLIO DELL'ESPERTO (INFERENZA LOGICA PROLOG):")
    print(f"  {advice}")
    
    if reasons:
        print("\n MOTIVAZIONI LOGICHE:")
        for reason in reasons:
            print(f"  • {reason}")
    
    print("═"*60 + "\n")


def main():
    """
    Workflow principale del sistema con interazione ciclica.
    """
    res = load_resources()
    
    lista_noc = sorted(res['noc_map'].values())
    lista_sport = sorted(res['sport_map'].values())

    sport_to_id = {v: k for k, v in res['sport_map'].items()}
    noc_to_id = {v: k for k, v in res['noc_map'].items()}

    while True:
        print("\n" + "═"*60)
        print("            SISTEMA DI PREDIZIONE OLIMPICA       ")
        print("═"*60)

        u_sex_raw = get_input_validato(" Sesso (M/F): ", "sex")
        u_noc_str = get_input_validato(" Codice Nazione (es. ITA, USA): ", "noc", opzioni=lista_noc)
        u_sport_str = get_input_validato(" Sport (es. Basketball): ", "sport", opzioni=lista_sport)

        u_sex = 0 if u_sex_raw == 'M' else 1
        u_sport_id = sport_to_id[u_sport_str]
        u_noc_id = noc_to_id[u_noc_str]

        # Costruzione e scaling delle feature
        atleta_df, history_source = build_athlete_dataframe(res, u_sex, u_noc_id, u_sport_id)
        atleta_scaled = res['scaler'].transform(atleta_df)
        
        # Generazione delle predizioni probabilistiche dai due modelli avanzati
        prob_rf = predict_medal_probability(res['rf_model'], atleta_scaled)
        prob_gb = predict_medal_probability(res['gb_model'], atleta_scaled)

        # Calcolo di una probabilità combinata stabile (media) da passare al motore Prolog
        prob_combinata = (prob_rf + prob_gb) / 2.0

        # Analisi probabilistica settoriale del Naive Bayes
        leader_noc, dominanza = find_sector_leader(res, u_sport_str)
        
        # Interrogazione del motore di inferenza logica Prolog
        advice, reasons = get_olympic_advice_detailed(prob_combinata, u_noc_str, u_sport_str, res['kb_path'])
        
        # Output dei dati unificati
        print_results(prob_rf, prob_gb, u_sport_str, leader_noc, dominanza, history_source, advice, reasons)

        continua = input("[?] Vuoi analizzare un altro atleta? (S/N): ").strip().upper()
        if continua != "S":
            print("\n[*] Chiusura del sistema!")
            break

if __name__ == "__main__":
    main()