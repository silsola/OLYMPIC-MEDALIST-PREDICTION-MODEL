import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

"""
Modulo di utilità per la gestione e il preprocessing del dataset olimpico.
Contiene funzioni per il caricamento, la pulizia, la standardizzazione dei dati
e l'estrazione di subset specifici per i modelli di machine learning.
"""

def load_and_prepare_data(file_path):
    """
    Carica il dataset olimpico, esegue la pulizia dei nomi e prepara i dati.
    
    Operazioni:
    
    * Normalizza i nomi delle colonne.
    * Crea la colonna target binaria 'Won_Medal'.
    * Codifica il genere (Sex) in formato numerico.
    * Genera e salva i mapping per 'Sport' e 'NOC'.

    :param file_path: Percorso del file CSV del dataset.
    :return: DataFrame pre-processato pronto per l'analisi.
    """
    df = pd.read_csv(file_path)

    df.columns = [col.strip().replace(' ', '_')for col in df.columns]

    df['Won_Medal'] = df['Medal'].apply(lambda x: 1 if str(x) in ['Gold', 'Silver', 'Bronze'] else 0)

    df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})

    df['Sport'] = df['Sport'].astype('category')
    sport_mapping = dict(enumerate(df['Sport'].cat.categories))
    joblib.dump(sport_mapping, 'modelli/sport_mapping.pkl')
    os.makedirs('modelli', exist_ok=True)
    df['Sport_cat'] = df['Sport'].cat.codes

    df['NOC'] = df['NOC'].astype('category')
    noc_mapping = dict(enumerate(df['NOC'].cat.categories))
    joblib.dump(noc_mapping, 'modelli/noc_mapping.pkl')
    df['NOC_cat'] = df['NOC'].cat.codes

    return df


def standardize_features(X_train, X_test, scaler_path="modelli/scaler.pkl"):
    """
    Applica la standardizzazione (Z-score normalization) alle feature del modello.
    
    Lo scaler viene addestrato solo sul set di training per evitare data leakage
    e viene successivamente salvato su disco.

    -param X_train: Feature del set di addestramento.
    -param X_test: Feature del set di test.
    -param scaler_path: Percorso dove salvare l'oggetto StandardScaler serializzato.
    -return: Una tupla contenente i DataFrame (X_train_scaled, X_test_scaled).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path) 

    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)


def get_vincitori_for_nb(path_dataset):
    """
    Estrae dal dataset originale esclusivamente le righe relative ad atleti che hanno vinto una medaglia.
    
    Questa funzione è specifica per il training del modello Naive Bayes, che lavora 
    sulla distribuzione probabilistica delle vittorie per nazione e sport.

    -param path_dataset: Percorso del file CSV originale.
    -return: DataFrame contenente solo i record dei medagliati.
    -rtype: pandas.DataFrame
    """
    if not os.path.exists(path_dataset):
        print(f"[!] Errore: Il file {path_dataset} non esiste.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path_dataset)
        # Filtro medaglie (Oro, Argento, Bronzo)
        medaglie_valide = ['Gold', 'Silver', 'Bronze']
        vincitori = df[df['Medal'].isin(medaglie_valide)].copy()
        return vincitori
    except Exception as e:
        print(f"[!] Errore nel caricamento: {e}")
        return pd.DataFrame()