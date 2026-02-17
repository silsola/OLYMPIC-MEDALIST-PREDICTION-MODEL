# OLYMPIC MEDALIST PREDICTION MODEL
Questo progetto propone un sistema di **Intelligenza Artificiale Ibrida** per l'analisi delle performance olimpiche.
Combina la potenza predittiva del **Machine Learning** (Python) con il ragionamento logico di un **Sistema Esperto** (Prolog) per fornire raccomandazioni strategiche basate sulla storia delle nazioni.

---

## Finalità del progetto

L'obiettivo didattico è applicare i concetti fondamentali dell'**Ingegneria della Conoscenza**, integrando:

- La **modellazione dei dati** storici tramite tecniche di Machine Learning.
- La **rappresentazione della conoscenza** simbolica tramite fatti e regole Prolog.
- Il **ragionamento inferenziale** per la generazione di raccomandazioni strategiche.
- L'**integrazione multi-linguaggio** (Python e Prolog) tramite bridge bidirezionale.
- La **documentazione automatizzata** tramite standard professionali (Sphinx).

---

## Contenuti e metodi

Il progetto adotta un approccio ibrido per superare i limiti dei modelli puramente statistici, aggiungendo un livello di interpretazione storica alle previsioni:

### Architettura del Sistema

| Componente | Ruolo | Tecnologie |
|------------|------|-------------|
| **Livello Predittivo** | Analisi trend e classificazione (Medaglia 1/0) | Python, Scikit-learn |
| **Livello Logico** | Validazione storica e generazione consigli | SWI-Prolog, KB |
| **Integrazione** | Scambio dati tra modelli e inferenza | PySwip |


### Logica di Valutazione (Olympic Advice)

Il sistema incrocia l'output binario del modello ML con la **Knowledge Base** per categorizzare il risultato tramite il predicato `olympic_advice/4`:

- **Dominio Storico**: Previsione positiva supportata da una consolidata tradizione d'élite.
- **Potenziale Rivelazione**: Previsione positiva per una nazione emergente (non presente nei fatti storici).
- **Sfida Difficile**: Previsione statistica negativa nonostante l'appartenenza della nazione all'élite storica dello sport.
- **Outsider**: Assenza sia di presupposti statistici che di tradizione storica.

---

## Dettagli tecnici

### Rappresentazione della Conoscenza
- **Fatti (Prolog)**: Definizione delle eccellenze storiche tramite il predicato `historical_elite(NOC, Sport)`.
- **Regole (Prolog)**: Inferenza dinamica basata sul matching tra l'output del classificatore e il database storico.
- **Modello ML**: Classificatori addestrati su dataset storico per la predizione di podi olimpici in base a parametri di performance recenti.

---

## Tecnologie utilizzate

- **Python 3.10+** – Elaborazione dati, Machine Learning e orchestrazione.
- **SWI-Prolog 9.x** – Motore di inferenza logica e Knowledge Base.
- **PySwip** – Bridge per l'integrazione tra l'ambiente Python e Prolog.
- **Pandas & Scikit-learn** – Manipolazione dataset e algoritmi predittivi.
- **Sphinx** – Generazione automatica della documentazione tecnica.

---

## Struttura del Progetto

```
OLYMPIC MEDALIST PREDICTION MODEL/
│
├── apprendimento_probabilistico/ # Moduli di ML probabilistico
│   ├── grafici/                  # Visualizzazione dei risultati
│   ├── iperparametri/            # Tuning dei modelli
│   ├── modelli/                  # Serializzazione modelli salvati
│   └── training.py               # Script di addestramento
│
├── apprendimento_supervisionato/ # Moduli di ML supervisionato
│   ├── grafici/
│   ├── iperparametri/
│   ├── modelli/
│   └── training.py
│
├── dataset/                      # Risorse dati
│   ├── dataset_utils.py          # Utility per la manipolazione dati
│   └── olympics_dataset.csv      # Dataset storico principale
│
├── docs/                         # Documentazione Sphinx
│   ├── _build/                   # Output documentazione (HTML/PDF)
│   ├── _static/ & _templates/    # Asset di configurazione
│   ├── conf.py                   # Configurazione Sphinx
│   ├── index.rst                 # Entry point documentazione
│   ├── make.bat                  # Script di build per Windows
│   └── Makefile                  # Script di build per Unix
│
├── kb/                           # Knowledge Base
│   └── rules.pl                  # Fatti e regole Prolog
│
├── analisi_medagliere.py         # Analisi statistica dei dati
├── main.py                       # Entry point: esecuzione sistema
├── README.md                     # Documentazione principale
├── report_leader_tesi.csv        # Output finale delle inferenze
└── requirements.txt              # Dipendenze del progetto
```

---

## Installazioni principali


### Prerequisiti
Assicurarsi che SWI-Prolog sia installato e configurato nel PATH di sistema
```bash
python --version  # Consigliata >= 3.12
swipl --version   # Consigliata >= 9.0
```


### Installazione dipendenze
```bash
pip install -r requirements.txt
```

---

## Esecuzione del progetto


### Avvio del sistema
Per eseguire l'analisi completa e generare il report finale: ```python main.py```

---

## Autore

**Silvia Solazzo**  
Matricola: 779231

Corso di Laurea Triennale in Informatica  
Università degli Studi di Bari "Aldo Moro"  
Anno Accademico 2025-2026

Email: s.solazzo9@studenti.uniba.it 