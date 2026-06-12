import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""
Modulo di utilità per la gestione e il preprocessing del dataset olimpico.

Questo modulo concentra le operazioni riutilizzabili:
caricamento del dataset, creazione del target, codifica delle variabili categoriche,
salvataggio dei mapping e standardizzazione delle feature.

Fornisce il medesimo set di feature scalate a tutti i modelli supervisionati:
* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
"""


MEDAL_VALUES = {"Gold", "Silver", "Bronze"}
BASE_FEATURE_COLUMNS = ["Sex", "NOC", "Sport"]
HISTORICAL_FEATURE_COLUMNS = [
    "NOC_Total_Medals_Before",
    "NOC_Medal_Rate_Before",
    "NOC_Sport_Medals_Before",
    "NOC_Sport_Medal_Rate_Before",
    "Sport_Medal_Rate_Before",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + HISTORICAL_FEATURE_COLUMNS
TARGET_COLUMN = "Won_Medal"


def _normalize_columns(df):
    """
    Normalizza i nomi colonna mantenendo la semantica del dataset olimpico.
    """
    df = df.copy()
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    return df


def _validate_columns(df, required_columns):
    """
    Verifica che il dataset contenga tutte le colonne richieste.
    """
    missing_columns = set(required_columns).difference(df.columns)
    if missing_columns:
        raise ValueError(f"Colonne mancanti nel dataset: {', '.join(sorted(missing_columns))}")


def _save_mapping(mapping, output_dir, filename):
    """
    Salva un mapping categorico nella directory indicata.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(mapping, os.path.join(output_dir, filename))


def _merge_group_history(df, group_columns, prefix):
    """
    Aggiunge statistiche cumulative precedenti all'anno corrente per un gruppo.

    Il calcolo avviene aggregando per anno e sottraendo il valore dell'anno
    corrente dal cumulativo, in modo da non usare informazioni future o della
    stessa edizione olimpica per predire quella riga.
    """
    yearly = (
        df.groupby(group_columns + ["Year"], observed=False)[TARGET_COLUMN]
        .agg(Appearances_Year="size", Medals_Year="sum")
        .reset_index()
        .sort_values(group_columns + ["Year"])
    )

    group_obj = yearly.groupby(group_columns, observed=False)
    yearly[f"{prefix}_Appearances_Before"] = (
        group_obj["Appearances_Year"].cumsum() - yearly["Appearances_Year"]
    )
    yearly[f"{prefix}_Medals_Before"] = group_obj["Medals_Year"].cumsum() - yearly["Medals_Year"]
    yearly[f"{prefix}_Medal_Rate_Before"] = np.where(
        yearly[f"{prefix}_Appearances_Before"] > 0,
        yearly[f"{prefix}_Medals_Before"] / yearly[f"{prefix}_Appearances_Before"],
        0.0,
    )

    keep_columns = group_columns + [
        "Year",
        f"{prefix}_Medals_Before",
        f"{prefix}_Medal_Rate_Before",
    ]
    return df.merge(yearly[keep_columns], on=group_columns + ["Year"], how="left")


def add_historical_features(df):
    """
    Crea feature storiche per rappresentare la conoscenza pregressa.

    Le feature descrivono il rendimento storico della nazione e della coppia
    nazione-sport prima dell'anno considerato.
    """
    df = _merge_group_history(df, ["NOC"], "NOC_Total")
    df = df.rename(columns={"NOC_Total_Medal_Rate_Before": "NOC_Medal_Rate_Before"})
    df = _merge_group_history(df, ["NOC", "Sport"], "NOC_Sport")
    df = _merge_group_history(df, ["Sport"], "Sport")

    df["Sport_Medal_Rate_Before"] = df["Sport_Medal_Rate_Before"].fillna(0.0)
    for column in HISTORICAL_FEATURE_COLUMNS:
        df[column] = df[column].fillna(0.0)

    return df


def save_history_profile(df, output_dir):
    """
    Salva un profilo storico aggiornato per usare le stesse feature nel main.
    """
    latest_rows = df.sort_values("Year").copy()

    by_noc_sport = {}
    for _, row in latest_rows.groupby(["NOC", "Sport"], observed=False).tail(1).iterrows():
        by_noc_sport[(int(row["NOC"]), int(row["Sport"]))] = {
            column: float(row[column]) for column in HISTORICAL_FEATURE_COLUMNS
        }

    by_noc = {}
    for _, row in latest_rows.groupby("NOC", observed=False).tail(1).iterrows():
        by_noc[int(row["NOC"])] = {column: float(row[column]) for column in HISTORICAL_FEATURE_COLUMNS}

    by_sport = {}
    for _, row in latest_rows.groupby("Sport", observed=False).tail(1).iterrows():
        by_sport[int(row["Sport"])] = {column: float(row[column]) for column in HISTORICAL_FEATURE_COLUMNS}

    defaults = {column: float(latest_rows[column].median()) for column in HISTORICAL_FEATURE_COLUMNS}

    profile = {
        "feature_columns": FEATURE_COLUMNS,
        "historical_feature_columns": HISTORICAL_FEATURE_COLUMNS,
        "by_noc_sport": by_noc_sport,
        "by_noc": by_noc,
        "by_sport": by_sport,
        "defaults": defaults,
    }
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(profile, os.path.join(output_dir, "history_profile.pkl"))


def load_and_prepare_data(file_path, mappings_dir="modelli"):
    """
    Carica il dataset olimpico, esegue la pulizia dei nomi e prepara i dati.
    
    Operazioni:
    
    * Normalizza i nomi delle colonne.
    * Crea la colonna target binaria 'Won_Medal'.
    * Codifica il genere (Sex) in formato numerico.
    * Codifica la nazione (NOC) in formato numerico.
    * Genera e salva i mapping per 'Sport' e 'NOC'.

    :param file_path: Percorso del file CSV del dataset olimpico.
    :param mappings_dir: Directory in cui salvare sport_mapping.pkl e noc_mapping.pkl.
    :return: DataFrame pre-processato pronto per l'apprendimento supervisionato.
    """
    df = pd.read_csv(file_path)
    df = _normalize_columns(df)
    _validate_columns(df, ["Year", "Sex", "NOC", "Sport", "Medal"])

    df[TARGET_COLUMN] = df["Medal"].isin(MEDAL_VALUES).astype(int)

    df = df[["Year"] + BASE_FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()
    df["Year"] = df["Year"].astype(int)
    df["NOC"] = df["NOC"].astype(str).str.upper().str.strip()
    df["Sport"] = df["Sport"].astype(str).str.strip()
    df = add_historical_features(df)

    df["Sex"] = df["Sex"].map({"M": 0, "F": 1})
    df = df.dropna(subset=["Sex"]).copy()
    df["Sex"] = df["Sex"].astype(int)

    df["Sport"] = df["Sport"].astype(str).str.strip().astype("category")
    sport_mapping = dict(enumerate(df["Sport"].cat.categories))
    _save_mapping(sport_mapping, mappings_dir, "sport_mapping.pkl")
    df["Sport"] = df["Sport"].cat.codes

    df["NOC"] = df["NOC"].astype(str).str.upper().str.strip().astype("category")
    noc_mapping = dict(enumerate(df["NOC"].cat.categories))
    _save_mapping(noc_mapping, mappings_dir, "noc_mapping.pkl")
    df["NOC"] = df["NOC"].cat.codes
    save_history_profile(df, mappings_dir)

    return df


def standardize_features(X_train, X_test, scaler_path="modelli/scaler.pkl"):
    """
    Applica la standardizzazione alle feature del modello.
    
    Il processo prevede:

    1. Calcolo di media e deviazione standard sul set di training.
    2. Trasformazione di entrambi i set (train e test).
    3. Salvataggio dello scaler per normalizzare i futuri input dell'utente.

    Questo scaler unico servirà l'intera pipeline logistica, forestale 
    e i modelli basati su boosting (Gradient Boosting).

    :param X_train: Feature del set di addestramento.
    :param X_test: Feature del set di test.
    :param scaler_path: Percorso dove salvare l'oggetto StandardScaler serializzato.
    :return: Una tupla contenente i DataFrame (X_train_scaled, X_test_scaled).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaler_dir = os.path.dirname(scaler_path)
    if scaler_dir:
        os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)


def get_vincitori_for_nb(path_dataset):
    """
    Estrae dal dataset originale esclusivamente le righe relative ai medagliati.
    
    Questa funzione è fondamentale per il modello Naive Bayes perché:

    * Permette di calcolare la probabilità condizionata per nazione.
    * Riduce il rumore eliminando i record di chi non ha raggiunto il podio.

    :param path_dataset: Percorso del file CSV originale.
    :return: DataFrame contenente solo i record dei vincitori (Gold, Silver, Bronze).
    """
    if not os.path.exists(path_dataset):
        print(f"[!] Errore: Il file {path_dataset} non esiste.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path_dataset)
        df = _normalize_columns(df)
        _validate_columns(df, ["NOC", "Sport", "Medal"])
        vincitori = df[df["Medal"].isin(MEDAL_VALUES)].copy()
        vincitori["NOC"] = vincitori["NOC"].astype(str).str.upper().str.strip()
        vincitori["Sport"] = vincitori["Sport"].astype(str).str.strip()
        return vincitori
    except Exception as e:
        print(f"[!] Errore nel caricamento: {e}")
        return pd.DataFrame()