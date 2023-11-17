import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import spacy
from sklearn.utils import shuffle

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
pp_subtet = df_combined[df_combined['type'] == 'Positive_Correlation'].head(len(df_combined) // 90)
df_combined = pd.concat([pp_subtet, df_combined[df_combined['type'] != 'Positive_Correlation']])
nn_subtet = df_combined[df_combined['type'] == 'Positive_Correlation'].head(len(df_combined) // 90)
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

# Realizar predicciones para el modelo de Dirichlet
predictions_dirichlet = model_dirichlet.predict(X_test_vectorized_nlp)

# Evaluar el rendimiento del modelo de Dirichlet
print("=== Modelo Dirichlet ===")
print("Matriz de Confusión:")
print(confusion_matrix(y_test_nlp, predictions_dirichlet))
print("\nInforme de Clasificación:")
print(classification_report(y_test_nlp, predictions_dirichlet))

# Entrenar un clasificador de Random Forest para el modelo NLP
model_rf = RandomForestClassifier()
model_rf.fit(X_train_vectorized_nlp, y_train_nlp)

# Realizar predicciones para el modelo NLP
predictions_rf = model_rf.predict(X_test_vectorized_nlp)

# Evaluar el rendimiento del modelo NLP
print("=== Modelo NLP ===")
print("Matriz de Confusión:")
print(confusion_matrix(y_test_nlp, predictions_rf))
print("\nInforme de Clasificación:")
print(classification_report(y_test_nlp, predictions_rf))
