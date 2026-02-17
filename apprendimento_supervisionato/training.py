import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler


current_file_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_path, '..'))
os.chdir(current_file_path)
sys.path.append(os.path.abspath(os.path.join(current_file_path, '..')))


def setup_directories():
    """
    Crea la struttura delle cartelle per i risultati.
    """
    local_folders = ['modelli', 'grafici', 'iperparametri/tabelle', 'iperparametri/migliori']
    for folder in local_folders:
        os.makedirs(os.path.join(current_file_path, folder), exist_ok=True)
    print(f"[OK] Struttura cartelle pronta.")


def prepare_data():
    """
    Caricamento e preprocessing dei dati olimpici
    """
    path_dataset = os.path.join(project_root, "dataset", "olympics_dataset.csv")
    if not os.path.exists(path_dataset):
        print(f"[!] Dataset non trovato in {path_dataset}"); sys.exit()

    df = pd.read_csv(path_dataset)
    df['Won_Medal'] = df['Medal'].apply(lambda x: 1 if str(x) in ['Gold', 'Silver', 'Bronze'] else 0)

    features = ['Sex', 'NOC', 'Sport']
    df_clean = df[features + ['Won_Medal']].dropna()
    df_clean['Sex'] = df_clean['Sex'].map({'M': 0, 'F': 1})
    df_clean['Sport_cat'] = df_clean['Sport'].astype('category')
    df_clean['NOC_cat'] = df_clean['NOC'].astype('category')

    sport_mapping = dict(enumerate(df_clean['Sport_cat'].cat.categories))
    sport_to_id = {v: k for k, v in sport_mapping.items()}
    noc_mapping = dict(enumerate(df_clean['NOC_cat'].cat.categories))
    joblib.dump(sport_mapping, os.path.join(current_file_path, 'modelli', 'sport_mapping.pkl'))
    joblib.dump(noc_mapping, os.path.join(current_file_path, 'modelli', 'noc_mapping.pkl'))
    noc_to_id = {v: k for k, v in noc_mapping.items()}
    
    X = pd.DataFrame()
    X['Sex'] = df_clean['Sex']
    X['NOC'] = df_clean['NOC_cat'].cat.codes
    X['Sport'] = df_clean['Sport_cat'].cat.codes
    y = df_clean['Won_Medal']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def save_best_params(model_name, best_params):
    """
    Salva i migliori iperparametri in formato JSON.
    """
    path = os.path.join(current_file_path, 'iperparametri', 'migliori', f'{model_name}_best_params.json')
    with open(path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"[OK] Iperparametri salvati correttamente")


def save_performance_plot(results_summary):
    """
    Genera grafico di confronto.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(results_summary.keys())
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        means = [results_summary[model][m][0] for m in metrics]
        stds = [results_summary[model][m][1] for m in metrics]
        ax.bar(x + (i * width) - width/2, means, width, yerr=stds, label=model, capsize=5, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_title('Confronto Performance (3-Fold Cross-Validation)')
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(current_file_path, 'grafici', 'confronto_modelli_cv.png'))
    plt.close()
    print("[OK] Grafico di confronto salvato in grafici/confronto_modelli_cv.png")


def save_feature_importance(model, feature_names):
    """
    Grafico dell'importanza delle variabili per Random Forest.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title('Importanza delle Variabili (Random Forest)')
    plt.barh(range(len(indices)), importances[indices], color='seagreen', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importanza Relativa')
    plt.savefig(os.path.join(current_file_path, 'grafici', 'feature_importance_rf.png'))
    plt.close()


def main():
    setup_directories()
    print("[*] Preparazione dati in corso...")
    X_train, X_test, y_train, y_test = prepare_data()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(current_file_path, 'modelli', 'scaler.pkl'))

    results_summary = {}
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']


    # 1. LOGISTIC REGRESSION
    print("[*] Esecuzione Cross-Validation Logistic Regression...")
    log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    cv_log = cross_validate(log_model, X_train_scaled, y_train, cv=3, scoring=scoring_metrics)
    results_summary['Logistic Regression'] = {m: (cv_log[f'test_{m}'].mean(), cv_log[f'test_{m}'].std()) for m in scoring_metrics}
    
    log_model.fit(X_train_scaled, y_train)
    joblib.dump(log_model, os.path.join(current_file_path, 'modelli', 'logistic_regression.pkl'))
    save_best_params('logistic_regression', log_model.get_params())


    # 2. RANDOM FOREST
    print("[*] Esecuzione Grid Search Random Forest...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [5, 10],
        'class_weight': ['balanced']
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    rf_best = grid_search.best_estimator_
    save_feature_importance(rf_best, X_train.columns)
    
    save_best_params('random_forest', grid_search.best_params_)
    pd.DataFrame(grid_search.cv_results_).to_csv(os.path.join(current_file_path, 'iperparametri/tabelle/grid_search_rf.csv'), index=False)
    
    cv_rf = cross_validate(rf_best, X_train_scaled, y_train, cv=3, scoring=scoring_metrics)
    results_summary['Random Forest'] = {m: (cv_rf[f'test_{m}'].mean(), cv_rf[f'test_{m}'].std()) for m in scoring_metrics}
    
    joblib.dump(rf_best, os.path.join(current_file_path, 'modelli', 'random_forest.pkl'))

    save_performance_plot(results_summary)

    print("\n" + "═"*75)
    print(f"{'MODELLO':<22} {'METRICA':<12} {'MEDIA (CV)':<15} {'DEV.STD (±)':<12}")
    print("═"*75)
    for model, metrics in results_summary.items():
        for m_name, (mean, std) in metrics.items():
            print(f"{model:<22} {m_name.upper():<12} {mean:>10.2%} {std:>14.2%}")
        print("-" * 75)

if __name__ == "__main__":
    main()