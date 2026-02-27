import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

#titre de la page web
st.set_page_config(page_title="TARDIS Dashboard")



def load_data():
    """
    Fonction qui récupere le chemin du fichier cleaned_dataset.csv 
    """

    #recupere le chemin du fichier et le stock dans base_dir et retourne le
    base_dir = os.path.dirname(os.path.abspath(__file__))

    return pd.read_csv(os.path.join(base_dir, 'cleaned_dataset.csv'))

try:
    
    # df recupere le cleaned_dataset.csv
    df = load_data()

    #titre 
    st.title("TARDIS : Analyse des Retards SNCF")
    st.markdown("---")

    st.header("Chiffres")

    
    #Addition des éléments de la df et stockage dans des variables
    total_retard_depart = df["Nombre de trains en retard au départ"].sum()
    total_retard_15min = df["Nombre trains en retard > 15min"].sum()
    moyen_retard_global = df["Retard moyen de tous les trains à l'arrivée"].mean()

    #Structure de l'espace (division en 3 colonnes et affichage des col)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Retards au Départ", f"{int(total_retard_depart)}")
    col2.metric("Trains Retard > 15min", f"{int(total_retard_15min)}")
    col3.metric("Retard Moyen (Arrivée)", f"{moyen_retard_global:.2f} min")

    st.markdown("---")

    st.header("Distribution des Retards")

    st.write("Analyse du retard moyen des trains qui sont arrivés en retard :")

    #graphique
    #plusieurs graphiques sur une figure
    fig, ax = plt.subplots(figsize=(10, 4))
    #histograme
    sns.histplot(df["Retard moyen des trains en retard à l'arrivée"], bins=30, kde=True, ax=ax, color="#1f77b4")
    #titre pour l'axe
    ax.set_title("Répartition des retards à l'arrivée")
    #nom pour l'axe x
    ax.set_xlabel("Minutes de retard")
    #nom pour l'axe y
    ax.set_ylabel("Fréquence")
    #affichage de la figure 
    st.pyplot(fig)

    st.markdown("---")

    st.header("Prédiction des retards")

    #recupere le chemin du fichier label_encoders.joblib
    base_dir = os.path.dirname(os.path.abspath(__file__))

    #stockage du chemin dans label_encoders
    label_encoders = joblib.load(os.path.join(base_dir, 'label_encoders.joblib'))

    #tri l'intérieur des colonnes de A-Z  + supprime les élément vide dans la colonne (NaN)
    # garde uniquement les éléments une seul fois
    #on transforme le resultat en liste
    services = sorted(df["Service"].dropna().unique().tolist())
    gares_depart = sorted(df["Gare de départ"].dropna().unique().tolist())
    gares_arrivee = sorted(df["Gare d'arrivée"].dropna().unique().tolist())

    #formulaire de prédiction
    with st.form("prediction_form"):
        st.write("Entrez les paramètres du trajet :")
        col1, col2, col3 = st.columns(3)
        
        #organisations des formulaires, supprimer éléments NaN()/garde une fois l'élément/transforme en liste
        with col1:
            annee = st.number_input("Année", min_value=2000, max_value=2050, value=2025)
            mois = st.selectbox("Mois", list(range(1, 13)), format_func=lambda x: [
                "Janvier","Février","Mars","Avril","Mai","Juin",
                "Juillet","Août","Septembre","Octobre","Novembre","Décembre"
            ][x-1]) #janvier 0 car le premier element est 0
        with col2:
            service = st.selectbox("Service", services)
            duree = st.number_input("Durée moyenne du trajet (min)", min_value=1, max_value=600, value=90)
        with col3:
            gare_depart = st.selectbox("Gare de départ", gares_depart)
            gare_arrivee = st.selectbox("Gare d'arrivée", gares_arrivee)

        circulations = st.slider("Nombre de circulations prévues", 1, 1000, 500)
        submit = st.form_submit_button("Estimer les retards")

    #Gérer les résultats du submit
    if submit:
        try:

            #stock le chemin du fichier model.joblib dans la variable model
            model = joblib.load(os.path.join(base_dir, 'model.joblib'))

            #les inputs pour le model
            input_data = pd.DataFrame([{
                "Annee": annee,
                "Mois": mois,
                "Service": str(service),
                "Gare de départ": str(gare_depart),
                "Gare d'arrivée": str(gare_arrivee),
                "Durée moyenne du trajet": duree,
                "Nombre de circulations prévues": circulations
            }])


            #boucle pour les prédictions 

            #Traduction pour l'IA pour qu'il comprenne les éléments
            for col in ["Service", "Gare de départ", "Gare d'arrivée"]:
                input_data[col] = label_encoders[col].transform(input_data[col])

            #prédiction de l'IA stocké dans la variable prediction
            predictions = model.predict(input_data)[0]

            #nom des colonnes que le modele va renvoyer
            output_cols = [
                "Trains annulés",
                "Trains en retard au départ",
                "Retard moyen au départ (min)",
                "Trains en retard à l'arrivée",
                "Retard moyen à l'arrivée (min)",
                "Retard moyen global à l'arrivée (min)",
                "Trains en retard > 15min",
            ]

            #si prédiction reussi 
            st.success("Prédiction effectuée avec succès !")


            st.markdown("Résultats estimés")

            #réponse de l'IA 
            c1, c2, c3 = st.columns(3)
            c1.metric(output_cols[0], f"{int(round(predictions[0]))}")
            c2.metric(output_cols[1], f"{int(round(predictions[1]))}")
            c3.metric(output_cols[6], f"{int(round(predictions[6]))}")

            c4, c5, c6 = st.columns(3)
            c4.metric(output_cols[3], f"{int(round(predictions[3]))}")
            c5.metric(output_cols[2], f"{predictions[2]:.1f} min")
            c6.metric(output_cols[4], f"{predictions[4]:.1f} min")

            st.metric(output_cols[5], f"{predictions[5]:.1f} min")

            st.markdown("Visualisation")

            #Affichage du graphique de la réponse de l'IA
            fig, ax = plt.subplots(figsize=(10, 4))
            labels = [output_cols[1], output_cols[3], output_cols[6], output_cols[0]]
            values = [predictions[1], predictions[3], predictions[6], predictions[0]]
            colors = ["#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
            ax.bar(labels, values, color=colors)
            ax.set_ylabel("Nombre de trains")
            ax.set_title("Répartition des retards prédits")
            plt.xticks(rotation=15, ha='right')
            st.pyplot(fig)


            #Gestion des erreurs
        except Exception as e:
            st.error(f" Erreur lors de la prédiction : {e}")

except FileNotFoundError as e:
    st.error(f"Fichier introuvable : {e}")
except KeyError as e:
    st.error(f"Erreur de colonne : {e}")