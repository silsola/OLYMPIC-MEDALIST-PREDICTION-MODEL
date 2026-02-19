import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))


try:
    from dataset.dataset_utils import get_vincitori_for_nb
except ImportError:
    print("[!] Errore: Impossibile trovare 'dataset/dataset_utils.py'.")
    sys.exit()


def setup_directories():
    """
    Crea le sottocartelle per organizzare i risultati.
    """
    folders = ['modelli', 'grafici', 'iperparametri/tabelle', 'iperparametri/migliori']
    for folder in folders:
        os.makedirs(os.path.join(current_dir, folder), exist_ok=True)
    print("[*] Struttura cartelle pronta.")


def prepare_data_nb(vincitori):
    """
    Prepara i dati per il modello Bayesiano.
    """
    counts = vincitori['NOC'].value_counts()
    nazioni_valide = counts[counts >= 3].index
    df_filtrato = vincitori[vincitori['NOC'].isin(nazioni_valide)].copy()
    
    print(f"[*] Nazioni analizzate: {len(nazioni_valide)}")
    
    X_dummies = pd.get_dummies(df_filtrato['Sport'])
    sport_columns = X_dummies.columns.tolist()
    
    return X_dummies.values, df_filtrato['NOC'].values, sport_columns


def train_nb_optimized(X, y):
    """
    Esegue GridSearch per ottimizzare lo Smoothing di Laplace.
    """
    model = MultinomialNB()
    param_grid = {
        'alpha': [1e-10, 0.01, 0.1, 0.5, 1.0, 2.0],
        'fit_prior': [True, False]
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def generate_visualizations(model, sport_columns, grid_results):
    """
    Genera grafici per l'analisi della dominanza statistica.
    """
    res_df = pd.DataFrame(grid_results)
    plt.figure(figsize=(8, 5))
    plt.plot(res_df['param_alpha'].astype(float), res_df['mean_test_score'], marker='o', color='navy')
    plt.xscale('log')
    plt.title("Ottimizzazione Laplace Smoothing (Alpha)")
    plt.xlabel("Alpha (Log Scale)")
    plt.ylabel("Accuratezza CV")
    plt.grid(True, linestyle='--')
    plt.savefig('grafici/ottimizzazione_nb.png')
    plt.close()

    # 2. Dominanza Sport (es. Swimming)
    sport_test = 'Swimming'
    if sport_test in sport_columns:
        idx = sport_columns.index(sport_test)
        X_test = np.zeros((1, len(sport_columns)))
        X_test[0, idx] = 1
        probs = model.predict_proba(X_test)[0]
        top_10_idx = np.argsort(probs)[-10:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(model.classes_[top_10_idx], probs[top_10_idx], color='green')
        plt.title(f"Top 10 Probabilità: {sport_test}")
        plt.tight_layout()
        plt.savefig('grafici/dominanza_swimming.png')
        plt.close()

    # 2. Dominanza Sport (es. Fencing)
    sport_test = 'Fencing'
    if sport_test in sport_columns:
        idx = sport_columns.index(sport_test)
        X_test = np.zeros((1, len(sport_columns)))
        X_test[0, idx] = 1
        probs = model.predict_proba(X_test)[0]
        top_10_idx = np.argsort(probs)[-10:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(model.classes_[top_10_idx], probs[top_10_idx], color='red')
        plt.title(f"Top 10 Probabilità: {sport_test}")
        plt.tight_layout()
        plt.savefig('grafici/dominanza_fencing.png')
        plt.close()


def main():
    setup_directories()
    
    print("\n" + "═"*60)
    print("   TRAINING PROBABILISTICO NAIVE BAYES   ")
    print("═"*60)

    path_ds = os.path.join('..', 'dataset', 'olympics_dataset.csv')
    vincitori = get_vincitori_for_nb(path_ds)
    
    if vincitori.empty:
        return

    X, y, sport_columns = prepare_data_nb(vincitori)
    
    print("[*] Ricerca Alpha ottimale (GridSearch)...")
    best_model, best_params, cv_results = train_nb_optimized(X, y)
    
    print("[*] Esportazione grafici e modelli...")
    generate_visualizations(best_model, sport_columns, cv_results)
    
    joblib.dump(best_model, 'modelli/naive_bayes.pkl')
    joblib.dump(sport_columns, 'modelli/sport_columns_nb.pkl')
    
    with open('iperparametri/migliori/nb_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    pd.DataFrame(cv_results).to_csv('iperparametri/tabelle/grid_search_nb.csv', index=False)
    
    print(f"\n[OK] Training completato. Alpha scelto: {best_params['alpha']}")
    print("═"*60 + "\n")

if __name__ == "__main__":
    main()