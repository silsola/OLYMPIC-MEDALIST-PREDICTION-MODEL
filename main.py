import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
from pyswip import Prolog
from thefuzz import process 

"""
Modulo principale del Sistema di Predizione Olimpica.
Questo script funge da orchestratore tra i modelli di apprendimento supervisionato,
probabilistico e la Base di Conoscenza (KB) in Prolog. Gestisce l'interazione
con l'utente tramite interfaccia testuale e fornisce analisi predittive complete.
"""

warnings.filterwarnings("ignore", category=UserWarning)

def get_input_validato(messaggio, tipo, range_val=None, opzioni=None):
    """
    Gestisce l'input dell'utente garantendo la correttezza formale e semantica dei dati.
    
    Supporta la validazione di:

    * **Sesso**: (M/F)
    * **Codici nazione (NOC)**: presenti nel dataset
    * **Nomi degli sport**: con correzione automatica tramite Fuzzy Matching

    :param messaggio: Testo da mostrare all'utente.
    :param tipo: Tipo di dato atteso ('sex', 'noc', 'sport').
    :param range_val: Tuple (min, max) per i valori numerici.
    :param opzioni: Lista di valori validi (per NOC e Sport).
    :return: Il valore validato e normalizzato.
    """
    while True:
        try:
            valore = input(messaggio).strip()
            if tipo == "sex":
                valore = valore.upper()
                if valore in ['M', 'F']: return valore
                print("[!] Inserisci 'M' o 'F'.\n")
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
    """
    Carica i modelli serializzati e i file di supporto necessari alle previsioni.
    
    Vengono caricati i seguenti componenti:

    * **Modello Random Forest**: e Scaler (Supervisionato).
    * **Mapping categorici**: per Sport e Nazioni.
    * **Modello Naive Bayes**: (Probabilistico).

    :return: Dizionario contenente tutti gli oggetti joblib caricati.
    """
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
    Interroga la Knowledge Base (KB) in Prolog per generare spiegazioni simboliche.
    
    Combina la probabilità numerica del Machine Learning con le regole 
    logiche definite in Prolog per fornire:

    1. Un verdetto testuale basato sulla soglia di probabilità.
    2. Una lista di motivazioni logiche legate alla nazione e allo sport.

    :param prob_ml: Probabilità di vittoria (0-1) calcolata dal Random Forest.
    :param noc: Codice della nazione dell'atleta.
    :param sport: Disciplina sportiva analizzata.
    :return: Una tupla (Verdetto testuale, Lista di motivazioni logiche).
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
    """
    Utilizza il modello Naive Bayes per identificare la nazione dominante in uno sport.
    
    Analizza la distribuzione di probabilità per lo sport indicato e restituisce 
    la nazione con la probabilità più alta di vittoria storica.

    :param resources: Dizionario delle risorse caricate.
    :param user_sport_str: Nome dello sport da analizzare.
    :return: Una tupla (Nome Nazione Leader, Valore di Dominanza).
    """
    model = resources['nb_model']
    cols = resources['sport_cols_nb']
    if user_sport_str not in cols: return "N/D", 0.0
    input_vec = np.zeros((1, len(cols)))
    input_vec[0, cols.index(user_sport_str)] = 1
    probs = model.predict_proba(input_vec)[0]
    best_idx = np.argmax(probs)
    return model.classes_[best_idx], probs[best_idx]


def main():
    """
    Workflow principale del sistema:
    1. Caricamento modelli.
    2. Input utente validato.
    3. Predizione numerica tramite Random Forest.
    4. Analisi probabilistica del leader di settore.
    5. Generazione di spiegazioni logiche tramite Prolog.
    6. Output formattato dei risultati.
    """
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