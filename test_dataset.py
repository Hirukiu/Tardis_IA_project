import pandas as pd

df = pd.read_csv("ton_fichier.csv")

# ─── Colonnes à vérifier ──────────────────────────────────────────
cols_to_check = ["Gare d'arrivée", "Gare de départ", "Date"]  # adapte cette liste

# ─── Test 1 : Doublons ───────────────────────────────────────────
duplicates = df.duplicated()
if duplicates.any():
    print(f"KO: Duplicates found in cleaned dataset. ({duplicates.sum()} doublon(s))")
else:
    print("OK: No duplicates found.")

# ─── Test 2 : Valeurs manquantes (colonnes ciblées seulement) ────
missing = df[cols_to_check].isnull().sum()
cols_with_missing = missing[missing > 0]

if cols_with_missing.empty:
    print("OK: No missing values found.")
else:
    for col, count in cols_with_missing.items():
        print(f"KO: Missing values in column {col} ({count} valeur(s) manquante(s))")
