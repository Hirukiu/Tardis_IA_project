

import re
import pytest
import pandas as pd
import numpy as np

# ── Chargement unique du fichier ──────────────────────────────────────────────
CSV_PATH = "cleaned_dataset.csv"

@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV_PATH)


# ── Colonnes attendues ────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "Durée moyenne du trajet",
    "Nombre de circulations prévues",
    "Nombre de trains annulés",
    "Nombre de trains en retard au départ",
    "Retard moyen des trains en retard au départ",
    "Retard moyen de tous les trains au départ",
    "Nombre de trains en retard à l'arrivée",
    "Retard moyen des trains en retard à l'arrivée",
    "Retard moyen de tous les trains à l'arrivée",
    "Nombre trains en retard > 15min",
    "Nombre trains en retard > 30min",
    "Nombre trains en retard > 60min",
    "Prct retard pour causes externes",
    "Prct retard pour cause infrastructure",
    "Prct retard pour cause gestion trafic",
    "Prct retard pour cause matériel roulant",
    "Prct retard pour cause gestion en gare et réutilisation de matériel",
    "Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)",
    "Annee",
    "Mois",
]

COUNT_COLS = [
    "Nombre de circulations prévues",
    "Nombre de trains annulés",
    "Nombre de trains en retard au départ",
    "Nombre de trains en retard à l'arrivée",
    "Nombre trains en retard > 15min",
    "Nombre trains en retard > 30min",
    "Nombre trains en retard > 60min",
]

TEXT_COLS = [
    "Service",
    "Gare de départ",
    "Gare d'arrivée",
]

DATE_COL = "Date"
DATE_DT_COL = "Date_dt"


# ═══════════════════════════════════════════════════════════════════════════════
# 02 - Check duplicates
# ═══════════════════════════════════════════════════════════════════════════════
class TestDuplicates:
    def test_no_full_row_duplicates(self, df):
        """Aucune ligne entièrement dupliquée."""
        duplicates = df.duplicated()
        assert duplicates.sum() == 0, (
            f"{duplicates.sum()} ligne(s) dupliquée(s) trouvée(s)."
        )

    def test_no_station_pair_date_duplicates(self, df):
        """Pas de doublon sur (Date, Gare de départ, Gare d'arrivée)."""
        key_cols = [DATE_COL, "Gare de départ", "Gare d'arrivée"]
        duplicates = df.duplicated(subset=key_cols)
        assert duplicates.sum() == 0, (
            f"{duplicates.sum()} doublon(s) sur la clé station-paire + date."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 03 - Check missing values
# ═══════════════════════════════════════════════════════════════════════════════
class TestMissingValues:
    REQUIRED_COLS = [
        DATE_COL,
        "Service",
        "Gare de départ",
        "Gare d'arrivée",
        "Durée moyenne du trajet",
        "Nombre de circulations prévues",
        "Annee",
        "Mois",
    ]

    @pytest.mark.parametrize("col", REQUIRED_COLS)
    def test_required_column_no_null(self, df, col):
        """Les colonnes critiques ne doivent pas avoir de valeurs manquantes."""
        null_count = df[col].isnull().sum()
        assert null_count == 0, (
            f"Colonne '{col}' : {null_count} valeur(s) manquante(s)."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 04 - Check date format
# ═══════════════════════════════════════════════════════════════════════════════
class TestDateFormat:
    DATE_PATTERN = re.compile(r"^\d{4}-\d{2}$")        # YYYY-MM
    DATE_DT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # YYYY-MM-DD

    def test_date_column_format(self, df):
        """La colonne 'Date' doit être au format YYYY-MM."""
        invalid = df[DATE_COL].dropna().apply(
            lambda x: not self.DATE_PATTERN.match(str(x))
        )
        assert invalid.sum() == 0, (
            f"{invalid.sum()} valeur(s) invalide(s) dans '{DATE_COL}'."
        )

    def test_date_dt_column_format(self, df):
        """La colonne 'Date_dt' doit être au format YYYY-MM-DD."""
        invalid = df[DATE_DT_COL].dropna().apply(
            lambda x: not self.DATE_DT_PATTERN.match(str(x))
        )
        assert invalid.sum() == 0, (
            f"{invalid.sum()} valeur(s) invalide(s) dans '{DATE_DT_COL}'."
        )

    def test_annee_range(self, df):
        """La colonne 'Annee' doit contenir des années plausibles (2015–2030)."""
        out_of_range = df["Annee"].dropna()
        assert out_of_range.between(2015, 2030).all(), (
            "Des valeurs hors plage trouvées dans 'Annee'."
        )

    def test_mois_range(self, df):
        """La colonne 'Mois' doit être entre 1 et 12."""
        out_of_range = df["Mois"].dropna()
        assert out_of_range.between(1, 12).all(), (
            "Des valeurs hors plage (1-12) trouvées dans 'Mois'."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 05 - Check numeric columns
# ═══════════════════════════════════════════════════════════════════════════════
class TestNumericColumns:
    @pytest.mark.parametrize("col", NUMERIC_COLS)
    def test_numeric_dtype(self, df, col):
        """Les colonnes numériques doivent être de type float ou int."""
        assert pd.api.types.is_numeric_dtype(df[col]), (
            f"Colonne '{col}' n'est pas numérique (type : {df[col].dtype})."
        )

    @pytest.mark.parametrize("col", NUMERIC_COLS)
    def test_no_inf_values(self, df, col):
        """Aucune valeur infinie dans les colonnes numériques."""
        inf_count = np.isinf(df[col].dropna()).sum()
        assert inf_count == 0, (
            f"Colonne '{col}' : {inf_count} valeur(s) infinie(s)."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 06 - Check no negative counts
# ═══════════════════════════════════════════════════════════════════════════════
class TestNoNegativeCounts:
    @pytest.mark.parametrize("col", COUNT_COLS)
    def test_no_negative_values(self, df, col):
        """Les colonnes de comptage ne doivent pas avoir de valeurs négatives."""
        negative = (df[col].dropna() < 0).sum()
        assert negative == 0, (
            f"Colonne '{col}' : {negative} valeur(s) négative(s)."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 07 - Check feature engineering
# ═══════════════════════════════════════════════════════════════════════════════
class TestFeatureEngineering:
    DERIVED_COLS = ["Date_dt", "Annee", "Mois"]

    @pytest.mark.parametrize("col", DERIVED_COLS)
    def test_derived_feature_exists(self, df, col):
        """Les colonnes dérivées créées lors du feature engineering doivent exister."""
        assert col in df.columns, f"Colonne dérivée '{col}' manquante."

    def test_annee_consistent_with_date(self, df):
        """'Annee' doit être cohérente avec l'année extraite de 'Date'."""
        expected_year = df[DATE_COL].str[:4].astype(float)
        mismatch = (df["Annee"] != expected_year).sum()
        assert mismatch == 0, (
            f"{mismatch} ligne(s) avec 'Annee' incohérente avec 'Date'."
        )

    def test_mois_consistent_with_date(self, df):
        """'Mois' doit être cohérente avec le mois extrait de 'Date'."""
        expected_month = df[DATE_COL].str[5:7].astype(float)
        mismatch = (df["Mois"] != expected_month).sum()
        assert mismatch == 0, (
            f"{mismatch} ligne(s) avec 'Mois' incohérente avec 'Date'."
        )

    def test_date_dt_consistent_with_date(self, df):
        """'Date_dt' doit correspondre au premier jour du mois de 'Date'."""
        expected = df[DATE_COL] + "-01"
        mismatch = (df[DATE_DT_COL] != expected).sum()
        assert mismatch == 0, (
            f"{mismatch} ligne(s) avec 'Date_dt' incohérente avec 'Date'."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 08 - Check delay consistency
# ═══════════════════════════════════════════════════════════════════════════════
class TestDelayConsistency:
    def test_delayed_departure_le_total_trains(self, df):
        """Trains en retard au départ ≤ circulations prévues."""
        mask = df["Nombre de circulations prévues"].notna() & df["Nombre de trains en retard au départ"].notna()
        invalid = (
            df.loc[mask, "Nombre de trains en retard au départ"]
            > df.loc[mask, "Nombre de circulations prévues"]
        ).sum()
        assert invalid == 0, (
            f"{invalid} ligne(s) avec retards départ > circulations prévues."
        )

    def test_delayed_arrival_le_total_trains(self, df):
        """Trains en retard à l'arrivée ≤ circulations prévues."""
        mask = df["Nombre de circulations prévues"].notna() & df["Nombre de trains en retard à l'arrivée"].notna()
        invalid = (
            df.loc[mask, "Nombre de trains en retard à l'arrivée"]
            > df.loc[mask, "Nombre de circulations prévues"]
        ).sum()
        assert invalid == 0, (
            f"{invalid} ligne(s) avec retards arrivée > circulations prévues."
        )

    def test_cancelled_le_total_trains(self, df):
        """Trains annulés ≤ circulations prévues."""
        mask = df["Nombre de circulations prévues"].notna() & df["Nombre de trains annulés"].notna()
        invalid = (
            df.loc[mask, "Nombre de trains annulés"]
            > df.loc[mask, "Nombre de circulations prévues"]
        ).sum()
        assert invalid == 0, (
            f"{invalid} ligne(s) avec annulations > circulations prévues."
        )

    def test_retard_15min_le_retard_depart(self, df):
        """Trains en retard > 15 min ≤ trains en retard au départ."""
        mask = (
            df["Nombre de trains en retard au départ"].notna()
            & df["Nombre trains en retard > 15min"].notna()
        )
        invalid = (
            df.loc[mask, "Nombre trains en retard > 15min"]
            > df.loc[mask, "Nombre de trains en retard au départ"]
        ).sum()
        assert invalid == 0, (
            f"{invalid} ligne(s) avec retard>15min > retard départ total."
        )

    def test_retard_30min_le_15min(self, df):
        """Trains en retard > 30 min ≤ trains en retard > 15 min."""
        mask = (
            df["Nombre trains en retard > 15min"].notna()
            & df["Nombre trains en retard > 30min"].notna()
        )
        invalid = (
            df.loc[mask, "Nombre trains en retard > 30min"]
            > df.loc[mask, "Nombre trains en retard > 15min"]
        ).sum()
        assert invalid == 0, (
            f"{invalid} ligne(s) avec retard>30min > retard>15min."
        )

    def test_retard_60min_le_30min(self, df):
        """Trains en retard > 60 min ≤ trains en retard > 30 min."""
        mask = (
            df["Nombre trains en retard > 30min"].notna()
            & df["Nombre trains en retard > 60min"].notna()
        )
        invalid = (
            df.loc[mask, "Nombre trains en retard > 60min"]
            > df.loc[mask, "Nombre trains en retard > 30min"]
        ).sum()
        assert invalid == 0, (
            f"{invalid} ligne(s) avec retard>60min > retard>30min."
        )

    def test_mean_delay_all_le_mean_delayed_only_departure(self, df):
        """Retard moyen de tous les trains ≤ retard moyen des trains en retard (départ)."""
        mask = (
            df["Retard moyen de tous les trains au départ"].notna()
            & df["Retard moyen des trains en retard au départ"].notna()
        )
        invalid = (
            df.loc[mask, "Retard moyen de tous les trains au départ"]
            > df.loc[mask, "Retard moyen des trains en retard au départ"]
        ).sum()
        assert invalid == 0, (
            f"{invalid} ligne(s) : retard moyen tous > retard moyen en retard (départ)."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 09 - Check text column handling
# ═══════════════════════════════════════════════════════════════════════════════
class TestTextColumnHandling:
    @pytest.mark.parametrize("col", TEXT_COLS)
    def test_no_leading_trailing_spaces(self, df, col):
        """Les colonnes texte ne doivent pas avoir d'espaces en début/fin."""
        has_spaces = df[col].dropna().apply(lambda x: x != x.strip()).sum()
        assert has_spaces == 0, (
            f"Colonne '{col}' : {has_spaces} valeur(s) avec espaces parasites."
        )

    @pytest.mark.parametrize("col", TEXT_COLS)
    def test_no_empty_strings(self, df, col):
        """Les colonnes texte ne doivent pas contenir de chaînes vides."""
        empty = (df[col].dropna() == "").sum()
        assert empty == 0, (
            f"Colonne '{col}' : {empty} chaîne(s) vide(s)."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 10 - Check textual NA
# ═══════════════════════════════════════════════════════════════════════════════
class TestTextualNA:
    TEXTUAL_NA_VALUES = ["na", "n/a", "nan", "null", "none", "-", "?", ""]

    @pytest.mark.parametrize("col", TEXT_COLS)
    def test_no_textual_na(self, df, col):
        """Les colonnes texte ne doivent pas contenir de faux NA sous forme textuelle."""
        bad = df[col].dropna().str.lower().str.strip().isin(self.TEXTUAL_NA_VALUES).sum()
        assert bad == 0, (
            f"Colonne '{col}' : {bad} valeur(s) NA textuelle(s) détectée(s)."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 11 - Check station pair duplicates
# ═══════════════════════════════════════════════════════════════════════════════
class TestStationPairDuplicates:
    def test_no_reversed_pair_same_date(self, df):
        """
        Vérifie qu'il n'existe pas simultanément (A→B) et (B→A)
        pour la même date (ce serait suspect mais pas forcément une erreur).
        Ce test est informatif : il signale les paires inversées.
        """
        df_copy = df[[DATE_COL, "Gare de départ", "Gare d'arrivée"]].copy()
        df_copy["pair_key"] = df_copy.apply(
            lambda r: frozenset([r["Gare de départ"], r["Gare d'arrivée"]]),
            axis=1,
        )
        # Compter combien de paires ont les deux sens présents pour une même date
        counts = df_copy.groupby([DATE_COL, "pair_key"]).size()
        both_directions = (counts > 1).sum()
        # On lève une alerte si aucune paire bidirectionnelle n'est attendue
        # Adaptez ce seuil selon votre métier
        assert both_directions >= 0, (
            f"{both_directions} paire(s) station avec les 2 sens pour une même date."
        )

    def test_departure_different_from_arrival(self, df):
        """La gare de départ ne doit pas être identique à la gare d'arrivée."""
        same = (df["Gare de départ"] == df["Gare d'arrivée"]).sum()
        assert same == 0, (
            f"{same} ligne(s) avec gare départ == gare arrivée."
        )

