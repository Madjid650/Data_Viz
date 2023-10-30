import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


@st.cache_data


def load_data():
    # On charge le dataset avec un lien dynamique
    df = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/7700a7b4-c1bf-4fe8-b698-4d40d1b53073',delimiter=';', low_memory=False, encoding='ISO-8859-1')

    # On supprime les colonnes vides
    df.dropna(axis=1,how='all', inplace=True)

    # On supprime les doublons
    df.drop_duplicates(keep='first', inplace=True)

    # On supprime les départements d'outre-mer
    df = df[~df['Numéro'].isin(["971", "972", "973", "974", "976", "BMPM"])]

    # On remplace "BSPP" par "75" pour le département "Paris"
    df.loc[df['Numéro'] == 'BSPP', 'Numéro'] = '75'

    return df

# La fonction permet de corriger les erreurs causées par l'espace des nombres à 5 chiffres et plus et les virgules
def preprocess_data(df, selected_category):
    if df[selected_category].dtype == 'object':
        df[selected_category] = df[selected_category].apply(lambda x: str(x).replace(' ', '').replace(',', '') if isinstance(x, str) else x)
        df[selected_category] = pd.to_numeric(df[selected_category], errors='coerce')
        df.dropna(subset=[selected_category], inplace=True)
    else:
       pass


# On créé deux catégories utilisées selon les graphiques
categories = [
    "Incendies", "Accidents sur lieux de travail", "Accidents à domicile",
    "Accidents sur voie publique", "Malaises sur lieux de travail",
    "Malaises à domicile : urgence vitale", "Intoxications", "Accidents de circulation",
    "Odeurs - fuites de gaz", "Fait dus à l'électricité", "Risques technologiques", "Fuites d'eau",
    "Inondations", "Engins explosifs","Total interventions"
]

categories1 = [
    "Incendies", "Accidents sur lieux de travail", "Accidents à domicile",
    "Accidents sur lieux de travail", "Malaises sur lieux de travail", "Malaises sur lieux de travail", "Intoxications",
    "Accidents de circulation", "Odeurs - fuites de gaz", "Fait dus à l'électricité", "Risques technologiques", "Fuites d'eau",
    "Inondations"
]




st.title(" En France métropolitaine, où est-ce que les secours sont-ils le plus intervenus et pourquoi ? ")
with st.sidebar:
    st.header("Presented by Madjid ZEHANI #DataVz2023efrei")
    st.write("**LinkedIn :** http://www.linkedin.com/in/madjid-zehani/\n")


    # Petit easter egg
    if st.button("** Clic, clic, clic !! **"):
        st.balloons()
        st.write(" Youpiii ")


# Permet d'accéder à tous les graphiques et cartes depuis la sidebar
st.sidebar.markdown('<a href="#top-section1" style="color: red;">1/ Heatmap par département</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#top-section2" style="color: red;">2/ Heatmap par région</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#top-section3" style="color: orange;">3/ Graphique à barres par département</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#top-section4" style="color: orange;">4/ Graphique à barres par région</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#top-section5" style="color: pink;">5/ Graphique en secteurs par département</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#top-section6" style="color: pink;">6/ Graphique en secteurs par région</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#top-section7" style="color: silver;">7/ Graphique à points par région</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#top-section8" style="color: yellow;">8/ Moyenne du nombre d\'interventions par type dans une région</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#top-section9" style="color: cyan;">9/ Carte des interventions par type en France métropolitaine</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#top-section10" style="color: cyan;">10/ Carte des interventions par type et par région</a>', unsafe_allow_html=True)


# Bouton pour afficher le dataset
df = load_data()
agree = st.checkbox('Afficher le Dataset')
if agree:
    st.text("Voici le dataset duquel nous avons récupéré les informations :")
    st.write(df.head())





# Titre avec une marge en haut pour éviter d'être caché par la barre de défilement
st.markdown('<a name="top-section1"></a>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.title("1/ Heatmap des interventions des secours par département")

# L'utilisateur sélectionne la catégorie
selected_category = st.selectbox("Sélectionnez une catégorie d'intervention : ", categories)

# Utilisation de la fonction de prétraitement
preprocess_data(df, selected_category)

# Groupement (groupby) et somme (sum) des interventions par département
df_by_department = df.groupby("Département")[selected_category].sum().reset_index()

# Sélection des 10 premiers départements en nombre d'interventions
top_departments = df_by_department.nlargest(12, selected_category)

# Ajustement de la taille de la figure en fonction du nombre de départements
fig_height = max(10, len(top_departments) * 0.3)

# On créé la heatmap
fig, ax = plt.subplots(figsize=(12, fig_height))
sns.heatmap(top_departments.pivot_table(index="Département", values=selected_category),
                cmap="YlOrRd", annot=True, fmt=".0f", linewidths=0.5)
plt.title(f"Heatmap pour la catégorie : {selected_category}")
plt.xlabel("Département")

# Affichage de la heatmap
st.pyplot(fig)




# Titre avec une marge en haut pour éviter d'être caché par la barre de défilement
st.markdown('<a name="top-section2"></a>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.title("2/ Heatmap des interventions des secours par région")

# L'utilisateur sélectionne la catégorie
selected_category5 = st.selectbox("Sélectionnez une catégorie d'intervention : ", categories, key="selected_category5")


# Utilisation de la fonction de prétraitement
preprocess_data(df, selected_category5)


# Groupement (groupby) et somme (sum) des interventions par région
df_by_region = df.groupby("Région")[selected_category5].sum().reset_index()


# Sélection des 10 premières regions en nombre d'interventions
top_region = df_by_region.nlargest(12, selected_category5)

# Ajustement de la taille de la figure en fonction du nombre de régions
fig_height = max(10, len(top_region) * 0.3)

# On créé la heatmap
fig, ax = plt.subplots(figsize=(12, fig_height))
sns.heatmap(top_region.pivot_table(index="Région", values=selected_category5),
                cmap="YlOrRd", annot=True, fmt=".0f", linewidths=0.5)
plt.title(f"Heatmap pour la catégorie : {selected_category5}")
plt.xlabel("Région")

# On affiche la heatmap
st.pyplot(fig)





# Titre avec une marge en haut pour éviter d'être caché par la barre de défilement
st.markdown('<a name="top-section3"></a>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.title("3/ Graphique à barres des interventions des secours par département")

# Sélection de la catégorie par l'utilisateur
selected_category = st.selectbox("Que voulez-vous afficher : ", categories)

# Utilisation de la fonction de prétraitement
preprocess_data(df, selected_category)

# Groupement (groupby) et somme (sum) des interventions par département
df_by_department = df.groupby("Département")[selected_category].sum().reset_index()

# On trie les données par ordre décroissant d'interventions
df_by_department = df_by_department.sort_values(by=selected_category, ascending=False)

# Sélection des 15 premiers départements en nombre d'interventions
top_10_departments = df_by_department.head(15)

# Création du graphique à barres
fig, ax = plt.subplots(figsize=(16, 13)) 
plt.barh(top_10_departments["Département"], top_10_departments[selected_category])
plt.xlabel("Nombre d'interventions")
plt.ylabel("Département")
plt.title(f"Graphique à barres pour la catégorie : {selected_category}")

# Affichage du graphique à barres
st.pyplot(fig)





# Titre avec une marge en haut pour éviter d'être caché par la barre de défilement
st.markdown('<a name="top-section4"></a>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.title("4/ Graphique à barres des interventions des secours par région")

# Sélection de la catégorie par l'utilisateur
selected_category6 = st.selectbox("Que voulez-vous afficher : ", categories, key="selected_category6")

# Utilisation de la fonction de prétraitement
preprocess_data(df, selected_category6)

# Groupement (groupby) et somme (sum) des interventions par région
df_by_region = df.groupby("Région")[selected_category6].sum().reset_index()

# On trie les données par ordre décroissant d'interventions
df_by_region = df_by_region.sort_values(by=selected_category6, ascending=False)

# Création du graphique à barres
fig, ax = plt.subplots(figsize=(16, 13))
plt.barh(df_by_region["Région"], df_by_region[selected_category6])
plt.xlabel("Nombre d'interventions")
plt.ylabel("Région")
plt.title(f"Graphique à barres pour la catégorie : {selected_category6}")

# Affichage du graphique à barres
st.pyplot(fig)





# Titre avec une marge en haut pour éviter d'être caché par la barre de défilement
st.markdown('<a name="top-section5"></a>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.title("5/ Graphique en secteurs des interventions des secours par département")

# L'utilisateur sélectionne la catégorie
selected_category3 = st.selectbox("Sélectionnez une catégorie d'intervention : ", categories, key="selectbox_category3")

# Utilisation de la fonction de prétraitement
preprocess_data(df, selected_category3)


# Sélection des 12 départements avec le plus grand nombre d'interventions
top_12_departments = df.nlargest(12, selected_category3)

# On créé du graphique en secteurs
fig = px.pie(top_12_departments, names='Département', values=selected_category3, title=f"Répartition des interventions par département pour la catégorie : {selected_category3}")

# On ajuste la taille du graphique
fig.update_layout(
    autosize=False,
    width=700, 
    height=700  
)
# Affichage du graphique en secteurs
st.plotly_chart(fig)





# Titre avec une marge en haut pour éviter d'être caché par la barre de défilement
st.markdown('<a name="top-section6"></a>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.title("6/ Graphique en secteurs des interventions des secours par région")

# L'utilisateur sélectionne la catégorie
selected_category7 = st.selectbox("Sélectionnez une catégorie d'intervention : ", categories, key="selectbox_category7")

# Utilisation de la fonction de prétraitement
preprocess_data(df, selected_category7)

# Création du graphique en secteurs
fig = px.pie(df, names='Région', values=selected_category7, title=f"Répartition des interventions par région pour la catégorie : {selected_category7}")

# On ajuste la taille du graphique
fig.update_layout(
    autosize=False,
    width=700,  
    height=700  
)
# Affichage du graphique en secteurs
st.plotly_chart(fig)





# Titre avec une marge en haut pour éviter d'être caché par la barre de défilement
st.markdown('<a name="top-section7"></a>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.title("7/ Graphique nuage de points des interventions des secours par région")

# L'utilisateur sélectionne la catégorie 
selected_category8 = st.selectbox("Sélectionnez une catégorie d'intervention : ", categories, key='selected_category8')

# Utilisation de la fonction de prétraitement
preprocess_data(df, selected_category8)

# Calcul de la moyenne des départements pour la région
regional_data = df.groupby('Région')[selected_category8].mean().reset_index()

# Création du graphique de dispersion
plt.figure(figsize=(12, 6))
plt.scatter(regional_data['Région'], regional_data[selected_category8], c='blue', marker='o')
plt.title(f"Diagramme de Dispersion des Interventions ({selected_category8}) par Région")
plt.xlabel("Région")
plt.ylabel(f"Moyenne des Interventions ({selected_category8})")
plt.xticks(rotation=45) 

# Affichage du graphique
st.pyplot(plt)





# Titre avec une marge en haut pour éviter d'être caché par la barre de défilement
st.markdown('<a name="top-section8"></a>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.title("8/ Moyenne du nombre d'interventions des secours par type dans une région")

# L'utilisateur choisit la région
selected_region = st.selectbox("Sélectionnez une région : ", df["Région"].unique())

# On créé un DataFrame pour stocker les moyennes par type
mean_data = pd.DataFrame()

for selected_category in categories1:
    preprocess_data(df, selected_category)
    category_mean = df[df["Région"] == selected_region][selected_category].mean()
    mean_data.at[0, selected_category] = category_mean

# On créé un graphique à barres pour la moyenne par type 
mean_data = mean_data.T.reset_index()
mean_data.columns = ["Type d'intervention", "Moyenne"]

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=mean_data, x="Type d'intervention", y="Moyenne", ax=ax)
plt.xlabel("Type d'intervention")
plt.ylabel("Moyenne du nombre d'interventions")
plt.title(f"Moyenne du nombre d'interventions des secours par type dans la région {selected_region}")
plt.xticks(rotation=45)
plt.tight_layout()

# On affiche le graphique à barres
st.pyplot(fig)





# Titre avec une marge en haut pour éviter d'être caché par la barre de défilement
st.markdown('<a name="top-section9"></a>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.title("9/ Carte des interventions des secours par type en France métropolitaine ")

# On charge les données sur les départements pour réaliser la carte
url = 'https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb'
departements = gpd.read_file(url)

# On convertit le code département en type str pour la fusion
departements['code'] = departements['code'].astype(str)

# L'utilisateur choisit la catégorie
selected_category = st.selectbox("Sélectionnez une catégorie : ", categories)

# On s'assure que les valeurs sont propres en supprimant les espaces et en remplaçant les virgules
preprocess_data(df, selected_category)

# Groupement des données par département et calcul de la moyenne de la catégorie sélectionnée
prixm = df.groupby('Numéro')[selected_category].mean().reset_index()

# On remplit les codes départementaux avec des zéros à gauche
prixm['Numéro'] = prixm['Numéro'].astype(str).apply(lambda x: x.zfill(2))

# On s'assure que les valeurs sont numériques et remplacez les chaînes vides par NaN
prixm[selected_category] = pd.to_numeric(prixm[selected_category], errors='coerce')

# On fusionne les données des départements avec les données de la catégorie sélectionnée
merge = departements.merge(prixm, left_on='code', right_on='Numéro', how='left')

# On créé la carte
fig, ax = plt.subplots(1, 1, figsize=(8,6))
merge.plot(column=selected_category, ax=ax, legend=True, cmap='coolwarm')

plt.title(f'{selected_category} par département en France métropolitaine')
st.pyplot(plt)





# Titre avec une marge en haut pour éviter d'être caché par la barre de défilement
st.markdown('<a name="top-section10"></a>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

st.title("10/ Carte des interventions des secours par type et par région")

# On charge les données sur les régions
url = 'https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb'
regions = gpd.read_file(url)

# On convertit le code de la région en type str pour la fusion
regions['code'] = regions['code'].astype(str)

# L'utilisateur sélectionne la région
available_regions = df["Région"].unique()

# On exclut la région "Corse" de la liste des régions disponibles
available_regions = [region for region in available_regions if region != "Corse"]
selected_region = st.selectbox("Sélectionnez une région : ", available_regions)

# L'utilisateur sélectionne le type d'intervention
selected_category = st.selectbox("Sélectionnez un type d'intervention : ", categories)

# On filtre les données pour n'inclure que la région sélectionnée
filtered_data = df[df["Région"] == selected_region]

# Fonction de prétraitement
preprocess_data(df, selected_category)

# Groupement des données par département et calcul de la moyenne de la catégorie sélectionnée
prixm = filtered_data.groupby('Numéro')[selected_category].mean().reset_index()

# On remplit les codes de département avec des zéros à gauche
prixm['Numéro'] = prixm['Numéro'].astype(str).apply(lambda x: x.zfill(2))

# On s'assure que valeurs sont numériques et on remplace les chaînes vides par NaN
prixm[selected_category] = pd.to_numeric(prixm[selected_category], errors='coerce')

# On fusionne les données des régions avec les données de la catégorie sélectionnée
merge = regions.merge(prixm, left_on='code', right_on='Numéro', how='left')

# On créé la carte
fig, ax = plt.subplots(1, 1, figsize=(5,4))
merge.plot(column=selected_category, ax=ax, legend=True, cmap='coolwarm')

plt.title(f'{selected_category} pour la région {selected_region}')
st.pyplot(plt)