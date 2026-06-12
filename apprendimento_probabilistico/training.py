import os
import sys
import json
import tempfile
import joblib
import numpy as np
import pandas as pd

# Configurazione della directory cache di Matplotlib per prevenire conflitti di permessi
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "olympic_medalist_matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use('Agg')  # Backend non interattivo per prevenire errori di thread grafici in ambiente headless
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

"""
Modulo per l'addestramento del modello di apprendimento probabilistico (Naive Bayes).
Questo script gestisce l'ottimizzazione degli iperparametri tramite Grid Search, la generazione 
dei grafici di dominanza stocastica e l'esportazione automatica dello scatter plot di validazione.
"""

# Configurazione percorsi del progetto
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'olympics_dataset.csv')
MODEL_DIR = os.path.join(CURRENT_DIR, 'modelli')
GRAPH_DIR = os.path.join(CURRENT_DIR, 'grafici')
REPORT_DIR = os.path.join(CURRENT_DIR, 'report')
HYPERPARAM_TABLE_DIR = os.path.join(CURRENT_DIR, 'iperparametri', 'tabelle')
HYPERPARAM_BEST_DIR = os.path.join(CURRENT_DIR, 'iperparametri', 'migliori')

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


try:
    from dataset.dataset_utils import get_vincitori_for_nb
except ImportError:
    print("[!] Errore: Impossibile trovare 'dataset/dataset_utils.py'.")
    sys.exit()


def setup_directories():
    """
    Crea la gerarchia di cartelle necessaria per organizzare i risultati del training.
    """
    folders = [MODEL_DIR, GRAPH_DIR, REPORT_DIR, HYPERPARAM_TABLE_DIR, HYPERPARAM_BEST_DIR]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("[*] Struttura cartelle pronta.")


def prepare_data_nb(vincitori):
    """
    Trasforma il dataset in un formato adatto per il classificatore Multinomial Naive Bayes.
    """
    required_columns = {'NOC', 'Sport'}
    missing_columns = required_columns.difference(vincitori.columns)
    if missing_columns:
        raise ValueError(f"Colonne mancanti nel dataset dei medagliati: {', '.join(sorted(missing_columns))}")

    df_valid = vincitori[['NOC', 'Sport']].dropna().copy()
    df_valid['NOC'] = df_valid['NOC'].astype(str)
    df_valid['Sport'] = df_valid['Sport'].astype(str)

    counts = df_valid['NOC'].value_counts()
    nazioni_valide = counts[counts >= 3].index
    df_filtrato = df_valid[df_valid['NOC'].isin(nazioni_valide)].copy()

    if df_filtrato.empty:
        raise ValueError("Nessuna nazione con almeno 3 medaglie disponibile per Naive Bayes.")
    
    print(f"[*] Nazioni analizzate: {len(nazioni_valide)}")
    print(f"[*] Record medagliati usati: {len(df_filtrato)}")
    
    X_dummies = pd.get_dummies(df_filtrato['Sport'], dtype=int)
    sport_columns = [str(col) for col in X_dummies.columns.tolist()]
    
    return X_dummies.values, df_filtrato['NOC'].values, sport_columns


def train_nb_optimized(X, y):
    """
    Esegue la ricerca degli iperparametri (Grid Search) per il modello Naive Bayes.
    """
    model = MultinomialNB()
    param_grid = {
        'alpha': [1e-10, 0.01, 0.1, 0.5, 1.0, 2.0],
        'fit_prior': [True, False]
    }
    min_class_count = pd.Series(y).value_counts().min()
    cv_splits = min(3, int(min_class_count))
    if cv_splits < 2:
        raise ValueError("Dati insufficienti per eseguire la validazione incrociata.")

    grid_search = GridSearchCV(model, param_grid, cv=cv_splits, scoring='accuracy', n_jobs=1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def plot_sport_dominance(model, sport_columns, sport_test, filename, color):
    """
    Salva il grafico probabilistico delle nazioni più probabili per uno sport specifico.
    """
    if sport_test not in sport_columns:
        print(f"[!] Grafico dominanza saltato: sport '{sport_test}' non presente nei dati NB.")
        return

    idx = sport_columns.index(sport_test)
    X_test = np.zeros((1, len(sport_columns)))
    X_test[0, idx] = 1
    probs = model.predict_proba(X_test)[0]
    top_10_idx = np.argsort(probs)[-10:][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(model.classes_[top_10_idx], probs[top_10_idx], color=color)
    ax.set_title(f"Top 10 Probabilità: {sport_test}")
    ax.set_xlabel("Probabilità stimata")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, filename))
    plt.close(fig)
    print(f"[OK] Grafico generato nella cartella 'grafici': {filename}")


def plot_comparison_real_vs_nb(model, X, y, sport_columns):
    """
    Genera un grafico a dispersione confrontando la dominanza frequentista
    calcolata direttamente sulla matrice dei dati con le predizioni del modello.
    """
    try:
        X_df = pd.DataFrame(X, columns=sport_columns)
        y_series = pd.Series(y)

        comparison_data = []

        for sport in sport_columns:
            sport_mask = (X_df[sport] == 1)
            if not sport_mask.any():
                continue
            
            y_sport = y_series[sport_mask]
            totale_medaglie_sport = len(y_sport)
            counts = y_sport.value_counts()
            
            medaglie_leader = counts.max()
            dominanza_reale_pct = (medaglie_leader / totale_medaglie_sport) * 100

            input_vec = np.zeros((1, len(sport_columns)))
            input_vec[0, sport_columns.index(sport)] = 1
            probs = model.predict_proba(input_vec)[0]
            
            idx_leader_nb = np.argmax(probs)
            prob_leader_nb_pct = probs[idx_leader_nb] * 100

            comparison_data.append({
                "Sport": sport,
                "Dominanza_Reale": dominanza_reale_pct,
                "Probabilita_NB": prob_leader_nb_pct
            })

        merged_df = pd.DataFrame(comparison_data)

        fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
        
        ax.scatter(merged_df['Dominanza_Reale'], merged_df['Probabilita_NB'], 
                   color='#9b5de5', alpha=0.7, edgecolors='black', s=90, zorder=3, label='Discipline Sportive')
        
        ax.plot([0, 100], [0, 100], color='#ff595e', linestyle='--', linewidth=1.5, zorder=2, label='Perfetta Corrispondenza')
        
        ax.set_title("Validazione Empirica: Dominanza Storica vs Probabilità Naive Bayes", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Percentuale di Dominanza Reale dal Medagliere (%)", fontsize=10)
        ax.set_ylabel("Massima Probabilità a Posteriori del Modello (%)", fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle=':', alpha=0.6, zorder=1)
        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
        
        fig.tight_layout()
        filename = 'confronto_dominanza_reale_vs_nb.png'
        fig.savefig(os.path.join(GRAPH_DIR, filename))
        plt.close(fig)
        print(f"[OK] Grafico generato nella cartella 'grafici': {filename}")

    except Exception as e:
        print(f"[!] Errore imprevisto durante la costruzione del plot di confronto: {e}")


def generate_visualizations(model, sport_columns, grid_results):
    """
    Produce i grafici relativi all'accuratezza del modello e alle probabilità predittive.
    """
    res_df = pd.DataFrame(grid_results)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res_df['param_alpha'].astype(float), res_df['mean_test_score'], marker='o', color='navy')
    ax.set_xscale('log')
    ax.set_title("Ottimizzazione Laplace Smoothing (Alpha)")
    ax.set_xlabel("Alpha (Log Scale)")
    ax.set_ylabel("Accuratezza CV")
    ax.grid(True, linestyle='--')
    fig.tight_layout()
    filename = 'ottimizzazione_nb.png'
    fig.savefig(os.path.join(GRAPH_DIR, filename))
    plt.close(fig)
    print(f"[OK] Grafico generato nella cartella 'grafici': {filename}")

    plot_sport_dominance(model, sport_columns, 'Swimming', 'dominanza_swimming.png', 'green')
    plot_sport_dominance(model, sport_columns, 'Fencing', 'dominanza_fencing.png', 'red')


def build_sector_report(model, sport_columns):
    """
    Crea una tabella con la nazione leader stimata per ogni sport.
    """
    rows = []
    for sport in sport_columns:
        input_vec = np.zeros((1, len(sport_columns)))
        input_vec[0, sport_columns.index(sport)] = 1
        probs = model.predict_proba(input_vec)[0]
        best_idx = int(np.argmax(probs))
        rows.append({
            "Sport": sport,
            "Leader_NOC": str(model.classes_[best_idx]),
            "Dominanza": round(float(probs[best_idx]), 6)
        })
    return pd.DataFrame(rows).sort_values(["Sport", "Leader_NOC"])


def build_top_n_report(model, sport_columns, n=5):
    """
    Crea un report con le prime N nazioni più probabili per ogni sport.
    """
    rows = []
    for sport in sport_columns:
        input_vec = np.zeros((1, len(sport_columns)))
        input_vec[0, sport_columns.index(sport)] = 1
        probs = model.predict_proba(input_vec)[0]
        top_idx = np.argsort(probs)[-n:][::-1]

        for rank, class_idx in enumerate(top_idx, start=1):
            rows.append({
                "Sport": sport,
                "Rank": rank,
                "NOC": str(model.classes_[class_idx]),
                "Probabilita": round(float(probs[class_idx]), 6)
            })

    return pd.DataFrame(rows)


def main():
    """
    Coordina l'intero workflow dell'apprendimento probabilistico.
    """
    setup_directories()
    
    print("\n" + "═"*60)
    print("   TRAINING PROBABILISTICO NAIVE BAYES   ")
    print("═"*60)

    vincitori = get_vincitori_for_nb(DATASET_PATH)
    
    if vincitori.empty:
        print("[!] Dataset dei vincitori vuoto. Interruzione.")
        return

    try:
        X, y, sport_columns = prepare_data_nb(vincitori)
    except ValueError as e:
        print(f"[!] Errore preparazione dati NB: {e}")
        return
    
    print("[*] Ricerca Alpha ottimale (GridSearch) con Cross-Validation...")
    best_model, best_params, cv_results = train_nb_optimized(X, y)
    
    res_df = pd.DataFrame(cv_results)
    best_idx = res_df['mean_test_score'].idxmax()
    mean_accuracy = res_df.loc[best_idx, 'mean_test_score']
    std_accuracy = res_df.loc[best_idx, 'std_test_score']

    print("\n" + "─"*60)
    print(" RISULTATI VALIDAZIONE MODELLO PROBABILISTICO:")
    print(f"  • Configurazione ottima scelta: Alpha = {best_params['alpha']}")
    print(f"  • Fit Prior (Probabilità a priori): {best_params['fit_prior']}")
    print(f"  • Accuratezza Media di Associazione (CV): {mean_accuracy:.2%} (± {std_accuracy:.2%})")
    print("─"*60 + "\n")
    
    print("[*] Elaborazione dei grafici del modello...")
    generate_visualizations(best_model, sport_columns, cv_results)
    plot_comparison_real_vs_nb(best_model, X, y, sport_columns)
    
    sector_report = build_sector_report(best_model, sport_columns) 
    top_n_report = build_top_n_report(best_model, sport_columns, n=5)
    
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'naive_bayes.pkl'))
    joblib.dump(sport_columns, os.path.join(MODEL_DIR, 'sport_columns_nb.pkl'))
    
    with open(os.path.join(HYPERPARAM_BEST_DIR, 'nb_best_params.json'), 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    
    pd.DataFrame(cv_results).to_csv(os.path.join(HYPERPARAM_TABLE_DIR, 'grid_search_nb.csv'), index=False)
    sector_report.to_csv(os.path.join(REPORT_DIR, 'leader_storici_per_sport.csv'), index=False)
    top_n_report.to_csv(os.path.join(REPORT_DIR, 'top5_nazioni_per_sport.csv'), index=False)
    
    print("\n" + "─"*60)
    print(f"[OK] Training completato con successo.")
    print("[OK] Report leader salvato nella cartella 'report'")
    print("[OK] Report Top 5 salvato nella cartella 'report'")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()