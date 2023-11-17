import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import spacy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Cargar el tokenizador de spaCy
nlp = spacy.load("en_core_web_sm")

# Cargar datos de relaciones (donde se encuentra la columna 'abstract')
df_relations = pd.read_csv('relations_train.csv', delimiter='\t')

# Unir datos de relaciones con datos de abstractos utilizando 'abstract_id'
df_abstracts = pd.read_csv('abstracts_train.csv', delimiter='\t')
df_combined = pd.merge(df_relations, df_abstracts, on='abstract_id')

# Limpiar la columna 'abstract' convirtiendo todos los elementos a cadenas de texto
df_combined['abstract'] = df_combined['abstract'].astype(str)

# Disminuir la presencia de las clases
df_combined = shuffle(df_combined, random_state=42)
association_subset = df_combined[df_combined['type'] == 'Association'].head(len(df_combined) // 90)
df_combined = pd.concat([association_subset, df_combined[df_combined['type'] != 'Association']])
pp_subtet = df_combined[df_combined['type'] == 'Positive_Correlation'].head(len(df_combined) // 80)
df_combined = pd.concat([pp_subtet, df_combined[df_combined['type'] != 'Positive_Correlation']])
nn_subtet = df_combined[df_combined['type'] == 'Positive_Correlation'].head(len(df_combined) // 70)
df_combined = pd.concat([nn_subtet, df_combined[df_combined['type'] != 'Negative_Correlation']])

# Separar datos en conjunto de entrenamiento y conjunto de prueba
train_data, test_data = train_test_split(df_combined, test_size=0.2, random_state=42)

# Función para lematizar y procesar texto con spaCy
def process_text(text):
    doc = nlp(text)
    # Extraer lemas de las palabras y eliminar stopwords
    lemmas = [token.lemma_ for token in doc if not token.is_stop]
    return lemmas

# Preprocesamiento de datos de entrenamiento para el modelo NLP
X_train_nlp = train_data['abstract'].apply(process_text)
y_train_nlp = train_data['type']

# Preprocesamiento de datos de prueba para el modelo NLP
X_test_nlp = test_data['abstract'].apply(process_text)
y_test_nlp = test_data['type']

# Utilizar spaCy para lematizar texto y extraer características para el modelo NLP
vectorizer_nlp = CountVectorizer()
X_train_vectorized_nlp = vectorizer_nlp.fit_transform(X_train_nlp.apply(lambda x: ' '.join(x)))
X_test_vectorized_nlp = vectorizer_nlp.transform(X_test_nlp.apply(lambda x: ' '.join(x)))

# Entrenar un clasificador de Bayes ingenuo con modelo de Dirichlet
alpha_dirichlet = 0.1  # Hiperparámetro de suavizado de Dirichlet
model_dirichlet = MultinomialNB(alpha=alpha_dirichlet)
model_dirichlet.fit(X_train_vectorized_nlp, y_train_nlp)

# Entrenar un clasificador de Random Forest para el modelo NLP
model_rf = RandomForestClassifier()
model_rf.fit(X_train_vectorized_nlp, y_train_nlp)

# Predecir en datos de prueba
predictions_dirichlet = model_dirichlet.predict(X_test_vectorized_nlp)
predictions_rf = model_rf.predict(X_test_vectorized_nlp)

# Calcular exactitud
accuracy_dirichlet = accuracy_score(y_test_nlp, predictions_dirichlet)
accuracy_rf = accuracy_score(y_test_nlp, predictions_rf)

# Crear la aplicación con Streamlit
st.title('Clasificación de Relaciones en Textos Biomédicos')

# Sidebar para selección de algoritmos
selected_algorithms = st.sidebar.multiselect('Seleccione los algoritmos', ['Modelo Dirichlet', 'Modelo NLP', 'Ambos'])

# Mostrar resultados según algoritmos seleccionados
if 'Modelo Dirichlet' in selected_algorithms or 'Ambos' in selected_algorithms:
    st.header('Modelo Dirichlet')
    st.subheader('Matriz de Confusión:')
    st.write(confusion_matrix(y_test_nlp, predictions_dirichlet))
    st.subheader('Informe de Clasificación:')
    st.write(classification_report(y_test_nlp, predictions_dirichlet))
    st.subheader('Exactitud:')
    st.write(accuracy_dirichlet)

if 'Modelo NLP' in selected_algorithms or 'Ambos' in selected_algorithms:
    st.header('Modelo NLP')
    st.subheader('Matriz de Confusión:')
    st.write(confusion_matrix(y_test_nlp, predictions_rf))
    st.subheader('Informe de Clasificación:')
    st.write(classification_report(y_test_nlp, predictions_rf))
    st.subheader('Exactitud:')
    st.write(accuracy_rf)

# Aplicación para ingresar nuevos datos y clasificar
st.header('Clasificación de Nuevos Datos')
new_text = st.text_area('Ingrese el texto para clasificar:')
if st.button('Clasificar'):
    # Preprocesar el texto ingresado
    new_text_processed = process_text(new_text)
    # Vectorizar el texto
    new_text_vectorized = vectorizer_nlp.transform([' '.join(new_text_processed)])
    # Clasificar con ambos modelos
    prediction_dirichlet = model_dirichlet.predict(new_text_vectorized)
    prediction_rf = model_rf.predict(new_text_vectorized)
    
    st.subheader('Resultados:')
    st.write('Modelo Dirichlet:', prediction_dirichlet[0])
    st.write('Modelo NLP:', prediction_rf[0])

# Gráficos interactivos del rendimiento de los modelos
st.header('Rendimiento de los Modelos en Datos de Prueba')
# Gráfico circular para el Modelo Dirichlet
fig_dirichlet = px.pie(names=['Correctas', 'Incorrectas'],
                       values=[accuracy_dirichlet, 1 - accuracy_dirichlet],
                       title='Exactitud del Modelo Dirichlet',
                       labels={'names': 'Clasificación', 'values': 'Exactitud'})

# Configuración adicional para el gráfico circular del Modelo Dirichlet
fig_dirichlet.update_traces(textinfo='percent+label', pull=[0.1, 0])

# Mostrar el gráfico circular del Modelo Dirichlet
st.plotly_chart(fig_dirichlet)

# Gráfico circular para el Modelo NLP
fig_nlp = px.pie(names=['Correctas', 'Incorrectas'],
                 values=[accuracy_rf, 1 - accuracy_rf],
                 title='Exactitud del Modelo NLP',
                 labels={'names': 'Clasificación', 'values': 'Exactitud'})

# Configuración adicional para el gráfico circular del Modelo NLP
fig_nlp.update_traces(textinfo='percent+label', pull=[0.1, 0])

# Mostrar el gráfico circular del Modelo NLP
st.plotly_chart(fig_nlp)
