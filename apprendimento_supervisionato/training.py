import os
import sys
import json
import tempfile
import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "olympic_medalist_matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 

"""
Modulo per l'apprendimento supervisionato del modello di predizione medaglie.
Questo script si occupa del caricamento dei dati, del preprocessing (scaling e encoding),
dell'addestramento di modelli di classificazione (Logistic Regression, Random Forest e Gradient Boosting)
e della valutazione delle performance tramite Cross-Validation e diagnostica Overfitting.
"""

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "olympics_dataset.csv")
MODEL_DIR = os.path.join(CURRENT_DIR, 'modelli')
GRAPH_DIR = os.path.join(CURRENT_DIR, 'grafici')
HYPERPARAM_TABLE_DIR = os.path.join(CURRENT_DIR, 'iperparametri', 'tabelle')
HYPERPARAM_BEST_DIR = os.path.join(CURRENT_DIR, 'iperparametri', 'migliori')
RANDOM_STATE = 42
CV_SPLITS = 3
CV_REPEATS = 2

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dataset.dataset_utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_and_prepare_data,
    standardize_features,
)


def setup_directories():
    """
    Inizializza la struttura delle cartelle necessaria per i risultati.
    """
    local_folders = [MODEL_DIR, GRAPH_DIR, HYPERPARAM_TABLE_DIR, HYPERPARAM_BEST_DIR]
    for folder in local_folders:
        os.makedirs(folder, exist_ok=True)
    print(f"[OK] Struttura cartelle pronta.")


def prepare_data():
    """
    Esegue il caricamento e il preprocessing avanzato dei dati olimpici.
    """
    try:
        df = load_and_prepare_data(DATASET_PATH, mappings_dir=MODEL_DIR)
    except FileNotFoundError:
        print(f"[!] Dataset non trovato in {DATASET_PATH}")
        sys.exit(1)
    except ValueError as e:
        print(f"[!] Errore preparazione dataset: {e}")
        sys.exit(1)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print(f"[*] Record validi: {len(df)}")
    print(f"[*] Sport codificati: {df['Sport'].nunique()} | Nazioni codificate: {df['NOC'].nunique()}")
    print(f"[*] Distribuzione target: {y.value_counts(normalize=True).sort_index().to_dict()}")

    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def make_json_serializable(value):
    """
    Converte tipi NumPy/scikit-learn in valori salvabili in JSON.
    """
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_best_params(model_name, best_params):
    """
    Salva i migliori iperparametri individuati in formato JSON.
    """
    path = os.path.join(HYPERPARAM_BEST_DIR, f'{model_name}_best_params.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4, default=make_json_serializable)
    print(f"[OK] Iperparametri per {model_name} salvati correttamente")


def save_metrics_report(results_summary, performance_summary):
    """
    Salva le metriche di Cross-Validation, Training e Test in formato JSON.
    """
    report = {
        "cross_validation_train_set": {
            model: {metric: {"mean": values[0], "std": values[1]} for metric, values in metrics.items()}
            for model, metrics in results_summary.items()
        },
        "performance_confronto_diretto": performance_summary
    }
    path = os.path.join(HYPERPARAM_TABLE_DIR, "metriche_modelli.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=make_json_serializable)
    print("[OK] Report metriche avanzato salvato in iperparametri/tabelle/metriche_modelli.json")


def save_performance_plot(results_summary):
    """
    Genera un grafico di confronto tra le metriche dei vari modelli.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(results_summary.keys())
    x = np.arange(len(metrics))
    width = 0.25 
    
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model in enumerate(models):
        means = [results_summary[model][m][0] for m in metrics]
        stds = [results_summary[model][m][1] for m in metrics]
        ax.bar(x + (i * width) - (len(models) * width / 4), means, width, yerr=stds, label=model, capsize=5, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_title(f'Confronto Performance Modelli ({CV_SPLITS}x{CV_REPEATS} Repeated CV)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, 'confronto_modelli_cv.png'))
    plt.close(fig)
    print("[OK] Grafico di confronto salvato in grafici/confronto_modelli_cv.png")


def save_overfitting_analysis(performance_summary):
    """
    Genera il grafico comparativo multi-pannello (Train vs Test) per verificare l'overfitting.
    Questa funzione crea un sotto-grafico per ciascun modello registrato.
    """
    models = list(performance_summary.keys())
    metrics = ['accuracy', 'f1', 'recall']
    x = np.arange(len(metrics))
    width = 0.35

    # Crea una griglia di sotto-grafici dinamica in base al numero di modelli (1 riga, N colonne)
    fig, axes = plt.subplots(1, len(models), figsize=(18, 5.5), sharey=True)
    fig.suptitle('Verifica Overfitting: Performance su Train Set vs Test Set', fontsize=15, fontweight='bold')

    # Se c'è solo un modello (caso raro) trasforma axes in lista per renderlo iterabile
    if len(models) == 1:
        axes = [axes]

    for i, model_name in enumerate(models):
        ax = axes[i]
        train_vals = [performance_summary[model_name]['train'][m] for m in metrics]
        test_vals = [performance_summary[model_name]['test'][m] for m in metrics]

        ax.bar(x - width/2, train_vals, width, label='Train Set', color='#4682B4', alpha=0.9)
        ax.bar(x + width/2, test_vals, width, label='Test Set', color='#FF8C00', alpha=0.9)

        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='lower left')

    fig.tight_layout()
    plot_path = os.path.join(GRAPH_DIR, 'diagnostica_overfitting_train_test.png')
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"[OK] Grafico diagnostica overfitting salvato correttamente in {plot_path}")


def save_feature_importance(model, feature_names):
    """
    Genera il grafico dell'importanza delle variabili per il modello Random Forest.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Importanza delle Variabili (Random Forest)')
    ax.barh(range(len(indices)), importances[indices], color='seagreen', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importanza Relativa')
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPH_DIR, 'feature_importance_rf.png'))
    plt.close(fig)


def evaluate_model_performance(model, X_train, y_train, X_test, y_test):
    """
    Calcola le metriche complete sia sul Train Set che sul Test Set.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return {
        "train": {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, zero_division=0)
        },
        "test": {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0)
        }
    }


def print_metrics_table(title, metrics_summary):
    """
    Stampa una tabella compatta di metriche per modello.
    """
    print("\n" + "═"*75)
    print(title)
    print("═"*75)
    for model, metrics in metrics_summary.items():
        for m_name, value in metrics.items():
            if isinstance(value, tuple):
                print(f"{model:<22} {m_name.upper():<12} {value[0]:>10.2%} {value[1]:>14.2%}")
            else:
                print(f"{model:<22} {m_name.upper():<12} {value:>10.2%}")
        print("-" * 75)


def main():
    global perf_summary 
    setup_directories()
    print("[*] Preparazione dati in corso...")
    X_train, X_test, y_train, y_test = prepare_data()

    X_train_scaled, X_test_scaled = standardize_features(
        X_train,
        X_test,
        scaler_path=os.path.join(MODEL_DIR, 'scaler.pkl')
    )

    results_summary = {}
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE
    )
    grid_cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0)
    }


    # 1. LOGISTIC REGRESSION (Baseline)
    print("[*] Esecuzione Cross-Validation Logistic Regression...")
    log_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
    cv_log = cross_validate(log_model, X_train_scaled, y_train, cv=cv, scoring=scoring_metrics)
    results_summary['Logistic Regression'] = {m: (cv_log[f'test_{m}'].mean(), cv_log[f'test_{m}'].std()) for m in scoring_metrics}
    
    log_model.fit(X_train_scaled, y_train)
    joblib.dump(log_model, os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
    save_best_params('logistic_regression', log_model.get_params())
    
    perf_summary['Logistic Regression'] = evaluate_model_performance(
        log_model, X_train_scaled, y_train, X_test_scaled, y_test
    )


    # 2. RANDOM FOREST (Modello Avanzato)
    print("[*] Esecuzione Grid Search Random Forest...")
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [10],
        'max_features': ['sqrt'],
        'max_samples': [0.5],
        'class_weight': ['balanced_subsample']
    }
    
    grid_search_rf = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid_rf,
        cv=grid_cv,
        scoring=make_scorer(f1_score, zero_division=0),
        n_jobs=1
    )
    grid_search_rf.fit(X_train_scaled, y_train)
    
    rf_best = grid_search_rf.best_estimator_
    save_feature_importance(rf_best, X_train.columns)
    save_best_params('random_forest', grid_search_rf.best_params_)
    pd.DataFrame(grid_search_rf.cv_results_).to_csv(os.path.join(HYPERPARAM_TABLE_DIR, 'grid_search_rf.csv'), index=False)
    
    cv_rf = cross_validate(rf_best, X_train_scaled, y_train, cv=cv, scoring=scoring_metrics)
    results_summary['Random Forest'] = {m: (cv_rf[f'test_{m}'].mean(), cv_rf[f'test_{m}'].std()) for m in scoring_metrics}
    
    joblib.dump(rf_best, os.path.join(MODEL_DIR, 'random_forest.pkl'))
    
    perf_summary['Random Forest'] = evaluate_model_performance(
        rf_best, X_train_scaled, y_train, X_test_scaled, y_test
    )


    # 3. GRADIENT BOOSTING (Nuovo Modello Avanzato)
    print("[*] Esecuzione Grid Search Gradient Boosting...")
    param_grid_gb = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    
    grid_search_gb = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid_gb,
        cv=grid_cv,
        scoring=make_scorer(f1_score, zero_division=0),
        n_jobs=-1
    )
    grid_search_gb.fit(X_train_scaled, y_train)
    
    gb_best = grid_search_gb.best_estimator_
    save_best_params('gradient_boosting', grid_search_gb.best_params_)
    pd.DataFrame(grid_search_gb.cv_results_).to_csv(os.path.join(HYPERPARAM_TABLE_DIR, 'grid_search_gb.csv'), index=False)
    
    cv_gb = cross_validate(gb_best, X_train_scaled, y_train, cv=cv, scoring=scoring_metrics)
    results_summary['Gradient Boosting'] = {m: (cv_gb[f'test_{m}'].mean(), cv_gb[f'test_{m}'].std()) for m in scoring_metrics}
    
    joblib.dump(gb_best, os.path.join(MODEL_DIR, 'gradient_boosting.pkl'))
    
    perf_summary['Gradient Boosting'] = evaluate_model_performance(
        gb_best, X_train_scaled, y_train, X_test_scaled, y_test
    )


    # Salvataggio complessivo dei report e grafici aggiornati a 3 modelli
    save_performance_plot(results_summary)
    save_overfitting_analysis(perf_summary)
    save_metrics_report(results_summary, perf_summary)

    print_metrics_table(f"{'MODELLO':<22} {'METRICA':<12} {'MEDIA (CV)':<15} {'DEV.STD (±)':<12}", results_summary)
    
    print("\n" + "═"*75)
    print(" CONFRONTO TRAIN VS TEST (VERIFICA OVERFITTING)")
    print("═"*75)
    for model_name in perf_summary:
        print(f"\n>>> {model_name.upper()}:")
        for metric in ['accuracy', 'f1', 'recall']:
            train_val = perf_summary[model_name]['train'][metric]
            test_val = perf_summary[model_name]['test'][metric]
            print(f"  {metric.upper():<12} -> Train: {train_val:>10.2%} | Test: {test_val:>10.2%}")
    print("═"*75 + "\n")


if __name__ == "__main__":
    perf_summary = {}
    main()