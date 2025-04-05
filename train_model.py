import pandas as pd
import os
import joblib
import re
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

os.makedirs('model', exist_ok=True)

def detect_columns(df):
    text_candidates = ['body', 'content', 'text', 'message', 'email_body']
    label_candidates = ['label', 'class', 'spam', 'type']
    text_col = next((col for col in text_candidates if col in df.columns), None)
    label_col = next((col for col in label_candidates if col in df.columns), None)
    return text_col, label_col

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL_TOKEN', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL_TOKEN', text)
    text = re.sub(r'\d+', 'NUM_TOKEN', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words and len(word) > 2])

main_files = [f for f in os.listdir('dataset') if f.endswith('.csv') and '_vectorized' not in f]
dfs = []
for file in main_files:
    file_path = os.path.join('dataset', file)
    df_temp = None
    try:
        df_temp = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip', engine='python')
    except Exception as e:
        try:
            df_temp = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python')
        except Exception as e2:
            print(f"⚠️ Error en {file}: {str(e2)[:50]}")
            continue
    if df_temp is None or df_temp.shape[1] < 2:
        print(f"⚠️ {file} no tiene suficientes columnas")
        continue
    text_col, label_col = detect_columns(df_temp)
    if not text_col or not label_col:
        print(f"⚠️ Columnas no detectadas en {file}")
        continue
    df_temp = df_temp.rename(columns={text_col: 'text', label_col: 'label'})
    df_temp = df_temp[['text', 'label']].copy()
    df_temp['text'] = df_temp['text'].apply(clean_text)
    df_temp = df_temp[df_temp['text'].str.len() > 10]
    df_temp['label'] = df_temp['label'].apply(lambda x: 1 if str(x).lower() in ['spam', '1', 'phishing'] else 0)
    dfs.append(df_temp)
    print(f"✅ {file}: {len(df_temp)} registros (Cols: {text_col}/{label_col})")

if not dfs:
    print("❌ No se encontraron datos válidos")
    exit()

df = pd.concat(dfs, ignore_index=True)
print(f"\nDataset final: {len(df)} registros")
print("Distribución de clases:\n", df['label'].value_counts())

# Usamos HashingVectorizer para reducir el uso de memoria
vectorizer = HashingVectorizer(
    n_features=2**15,  # ajusta según tus recursos
    alternate_sign=False,
    ngram_range=(1, 2)
)
X = vectorizer.transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Usamos SGDClassifier para un entrenamiento más rápido
model = SGDClassifier(loss='hinge', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nExactitud del modelo: {accuracy:.4f}")
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))

# Guardar modelo y vectorizador (observa que HashingVectorizer no guarda vocabulario)
joblib.dump(model, 'model/spam_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("\n✅ Modelo entrenado y evaluado exitosamente!")
