import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(page_title="TARDIS Dashboard")

#empeche les refreshs
@st.cache_data

#recupere les donnée du fichier cdv et le met dans df
def load_data():
    """
    recupère le fichier csv cleaned_dataset.csv retourne un dataFrame.
        
    Returns:
        pd.DataFrame: Les données chargées.
    """
    return pd.read_csv('cleaned_dataset.csv')


try:
    df = load_data()

    st.title(" TARDIS : Analyse des Retards SNCF")
    st.markdown("---")

    #Statistiques 
    st.header(" Chiffres")
    
    # Utilisation des données des colonnes de la df
    total_retard_depart = df["Nombre de trains en retard au départ"].sum()
    total_retard_15min = df["Nombre trains en retard > 15min"].sum()
    moyen_retard_global = df["Retard moyen de tous les trains à l'arrivée"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Retards au Départ", f"{int(total_retard_depart)}")
    col2.metric("Trains Retard > 15min", f"{int(total_retard_15min)}")
    col3.metric("Retard Moyen (Arrivée)", f"{moyen_retard_global:.2f} min")

    st.markdown("---")

    #affichage des infos sur les retards
    st.header(" Distribution des Retards")
    st.write("Analyse du retard moyen des trains qui sont arrivés en retard :")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df["Retard moyen des trains en retard à l'arrivée"], bins=30, kde=True, ax=ax, color="#1f77b4")
    ax.set_title("Répartition des retards à l'arrivée")
    ax.set_xlabel("Minutes de retard")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)

    st.markdown("---")

    # 3. prediction des retards
    st.header(" Prédiction des retard")
    
    #formulaire 
    with st.form("prediction_form"):
        st.write("Entrez les paramètres du trajet :")


        #MathieUwU
        #Remplacement des inputs par les colones que tu as utilise pour entraine ton ia (mois, jour...)
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            st.selectbox("Mois du trajet", ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin"])
        with col_input2:
            st.slider("Nombre de circulations prévues", 1, 1000, 500)
            
        submit = st.form_submit_button("Estimer le retard")

        if submit:
            # Emplacement pour le modèle machine learning
            st.success("prediction matthieu")

except FileNotFoundError:
    st.error(" Fichier 'cleaned_dataset.csv' introuvable. Place-le dans le même dossier que ce script.")
except KeyError as e:
    st.error(f" Erreur de colonne : La colonne {e} n'existe pas dans ton fichier. Vérifie les majuscules et les espaces !")