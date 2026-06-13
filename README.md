# OLYMPIC MEDALIST PREDICTION MODEL
Questo progetto propone un sistema di **Intelligenza Artificiale Ibrida** per l'analisi delle performance olimpiche.
Combina la potenza predittiva del **Machine Learning** (Python) con il ragionamento logico di un **Sistema Esperto** (Prolog) per fornire raccomandazioni strategiche basate sulla storia delle nazioni.

---

## Installazione

```bash
pip install -r requirements.txt
```

Per usare la Knowledge Base serve anche SWI-Prolog installato sul sistema.

## Training

```bash
python3 apprendimento_supervisionato/training.py
python3 apprendimento_probabilistico/training.py
```

Il training supervisionato usa repeated cross-validation e salva medie/deviazioni
standard in:

```text
apprendimento_supervisionato/iperparametri/tabelle/metriche_modelli.json
```

Il training probabilistico salva anche:

```text
apprendimento_probabilistico/report/leader_storici_per_sport.csv
apprendimento_probabilistico/report/top5_nazioni_per_sport.csv
```

## Valutazione Della KB

```bash
python3 kb/evaluate_kb.py
```

Il report viene salvato in:

```text
kb/report/valutazione_kb.json
```

## Esecuzione

```bash
python3 main.py
```

Il sistema richiede:

- sesso dell'atleta (`M` o `F`)
- codice NOC della nazione
- sport da analizzare

Output prodotto:

- probabilità di podio stimata dal Random Forest
- contesto storico usato dal modello
- leader storico/probabilistico dello sport tramite Naive Bayes
- consiglio strategico generato dalla Knowledge Base Prolog
- motivazioni logiche della raccomandazione

## Feature Storiche

Per evitare un modello basato solo su codici categorici, il dataset viene arricchito
con feature cumulative calcolate prima dell'anno della riga:

- medaglie storiche della nazione;
- tasso storico di medaglia della nazione;
- medaglie storiche della coppia nazione-sport;
- tasso storico della coppia nazione-sport;
- tasso globale storico dello sport.

Queste feature rappresentano la conoscenza storica numerica usata dal modello
supervisionato e sono salvate in `history_profile.pkl` per essere riutilizzate
nel `main.py`.

---

## Autore

**Silvia Solazzo**  
Matricola: 779231

Corso di Laurea Triennale in Informatica  
Università degli Studi di Bari "Aldo Moro"  
Anno Accademico 2025-2026

Email: s.solazzo9@studenti.uniba.it 
