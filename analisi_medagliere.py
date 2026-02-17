import pandas as pd
import os

def genera_report_leader():
    print("="*60)
    print("   ANALISI MEDAGLIERE: LEADER STORICI PER SPORT   ")
    print("="*60)

    file_path = 'dataset/olympics_dataset.csv'
    if not os.path.exists(file_path):
        print(f"[!] Errore: Assicurati che il file sia in {file_path}")
        return

    df = pd.read_csv(file_path)

    medaglie_valide = ['Gold', 'Silver', 'Bronze']
    vincitori = df[df['Medal'].isin(medaglie_valide)].copy()

    counts = vincitori.groupby(['Sport', 'NOC']).size().reset_index(name='Medaglie')
    leaders = counts.loc[counts.groupby('Sport')['Medaglie'].idxmax()]
    totali_per_sport = vincitori.groupby('Sport').size().reset_index(name='Totale_Medaglie_Sport')
    
    report = pd.merge(leaders, totali_per_sport, on='Sport')
    report['Percentuale_Dominanza'] = (report['Medaglie'] / report['Totale_Medaglie_Sport'] * 100).round(2)

    report = report.sort_values(by='Percentuale_Dominanza', ascending=False)
    report.to_csv('report_leader_tesi.csv', index=False)
    
    print(f"\nAnalisi completata su {len(report)} discipline sportive.")
    print("-" * 75)
    print(f"{'SPORT':<25} | {'LEADER':<6} | {'MEDAGLIE':<10} | {'DOMINANZA'}")
    print("-" * 75)
    
    for index, row in report.head(66).iterrows():
        print(f"{row['Sport']:<25} | {row['NOC']:<6} | {int(row['Medaglie']):<10} | {row['Percentuale_Dominanza']}%")
    
    print("-" * 75)
    print("\n[OK] File 'report_leader_tesi.csv' generato correttamente!")
    print("[INFO] Il report include i dati storici da Atene 1896 a Parigi 2024.")

if __name__ == "__main__":
    genera_report_leader()