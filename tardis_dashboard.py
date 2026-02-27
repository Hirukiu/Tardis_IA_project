import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="TARDIS Dashboard")


def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base_dir, "cleaned_dataset.csv"))


try:
    df = load_data()

    st.title("TARDIS : Analyse des Retards SNCF")
    st.markdown("---")

    st.header("Chiffres")
    total_retard_depart = df["Nombre de trains en retard au départ"].sum()
    total_retard_15min = df["Nombre trains en retard > 15min"].sum()
    moyen_retard_global = df["Retard moyen de tous les trains à l'arrivée"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Retards au Départ", f"{int(total_retard_depart)}")
    col2.metric("Trains Retard > 15min", f"{int(total_retard_15min)}")
    col3.metric("Retard Moyen (Arrivée)", f"{moyen_retard_global:.2f} min")

    st.markdown("---")

    st.header("Distribution des Retards")
    st.write("Analyse du retard moyen des trains qui sont arrivés en retard :")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(
        df["Retard moyen des trains en retard à l'arrivée"],
        bins=30,
        kde=True,
        ax=ax,
        color="#1f77b4",
    )
    ax.set_title("Répartition des retards à l'arrivée")
    ax.set_xlabel("Minutes de retard")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)

    st.markdown("---")

    st.header("Prédiction des retards")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    label_encoders = joblib.load(os.path.join(base_dir, "label_encoders.joblib"))

    services = sorted(df["Service"].dropna().unique().tolist())
    gares_depart = sorted(df["Gare de départ"].dropna().unique().tolist())
    gares_arrivee = sorted(df["Gare d'arrivée"].dropna().unique().tolist())

    with st.form("prediction_form"):
        st.write("Entrez les paramètres du trajet :")
        col1, col2, col3 = st.columns(3)
        with col1:
            annee = st.selectbox(
                "Année", sorted(df["Annee"].dropna().unique().tolist(), reverse=True)
            )
            mois = st.selectbox(
                "Mois",
                list(range(1, 13)),
                format_func=lambda x: [
                    "Janvier",
                    "Février",
                    "Mars",
                    "Avril",
                    "Mai",
                    "Juin",
                    "Juillet",
                    "Août",
                    "Septembre",
                    "Octobre",
                    "Novembre",
                    "Décembre",
                ][x - 1],
            )
        with col2:
            service = st.selectbox("Service", services)
            duree = st.number_input(
                "Durée moyenne du trajet (min)", min_value=1, max_value=600, value=90
            )
        with col3:
            gare_depart = st.selectbox("Gare de départ", gares_depart)
            gare_arrivee = st.selectbox("Gare d'arrivée", gares_arrivee)

        circulations = st.slider("Nombre de circulations prévues", 1, 1000, 500)
        submit = st.form_submit_button("Estimer les retards")

    if submit:
        try:
            model = joblib.load(os.path.join(base_dir, "model.joblib"))

            input_data = pd.DataFrame(
                [
                    {
                        "Annee": annee,
                        "Mois": mois,
                        "Service": str(service),
                        "Gare de départ": str(gare_depart),
                        "Gare d'arrivée": str(gare_arrivee),
                        "Durée moyenne du trajet": duree,
                        "Nombre de circulations prévues": circulations,
                    }
                ]
            )

            for col in ["Service", "Gare de départ", "Gare d'arrivée"]:
                input_data[col] = label_encoders[col].transform(input_data[col])

            predictions = model.predict(input_data)[0]

            output_cols = [
                "Trains annulés",
                "Trains en retard au départ",
                "Retard moyen au départ (min)",
                "Trains en retard à l'arrivée",
                "Retard moyen à l'arrivée (min)",
                "Retard moyen global à l'arrivée (min)",
                "Trains en retard > 15min",
            ]

            st.success(" Prédiction effectuée avec succès !")
            st.markdown("###  Résultats estimés")

            c1, c2, c3 = st.columns(3)
            c1.metric(output_cols[0], f"{int(round(predictions[0]))}")
            c2.metric(output_cols[1], f"{int(round(predictions[1]))}")
            c3.metric(output_cols[6], f"{int(round(predictions[6]))}")

            c4, c5, c6 = st.columns(3)
            c4.metric(output_cols[3], f"{int(round(predictions[3]))}")
            c5.metric(output_cols[2], f"{predictions[2]:.1f} min")
            c6.metric(output_cols[4], f"{predictions[4]:.1f} min")

            st.metric(output_cols[5], f"{predictions[5]:.1f} min")

            st.markdown("  Visualisation")
            fig, ax = plt.subplots(figsize=(10, 4))
            labels = [output_cols[1], output_cols[3], output_cols[6], output_cols[0]]
            values = [predictions[1], predictions[3], predictions[6], predictions[0]]
            colors = ["#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
            ax.bar(labels, values, color=colors)
            ax.set_ylabel("Nombre de trains")
            ax.set_title("Répartition des retards prédits")
            plt.xticks(rotation=15, ha="right")
            st.pyplot(fig)

        except Exception as e:
            st.error(f" Erreur lors de la prédiction : {e}")

except FileNotFoundError as e:
    st.error(f"Fichier introuvable : {e}")
except KeyError as e:
    st.error(f"Erreur de colonne : {e}")
