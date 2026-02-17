import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(file_path):
    """
    Carica il dataset olimpico, prepara i dati per l'analisi supervisionata 
    e salva la mappatura delle nazioni e degli sport.
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
    Standardizza le feature e salva lo scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path) 

    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)


def get_vincitori_for_nb(path_dataset):
    """
    Estrae solo i medagliati per il training del Naive Bayes.
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