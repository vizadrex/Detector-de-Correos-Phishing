# 📧 Detector de Correos Phishing

Clasificador inteligente de correos electrónicos para detectar phishing usando Machine Learning (SVM + TF-IDF).

![Interfaz del detector](https://via.placeholder.com/800x400.png?text=Captura+del+Detector+Phishing)

## 🚀 Características Principales
- **Clasificación en tiempo real**: Analiza textos de correos al instante
- **Nivel de confianza**: Muestra el porcentaje de certeza de la predicción
- **Modelo eficiente**: Precisión del 97% en pruebas con dataset real
- **Interfaz simple**: Web intuitiva para usuarios no técnicos

## ⚙️ Instalación
1. Clonar el repositorio:
```bash
git clone https://github.com/vizadrex/Detector-de-Correos-Phishing.git
cd phishing-detector
    
Instalar dependencias:

bash
Copy
pip install -r requirements.txt
Descargar dataset:

Obtener spam_emails.zip de Kaggle

Colocar el ZIP en la raíz del proyecto

Entrenar modelo:

bash
Copy
python train_model.py
🖥️ Uso
Iniciar la aplicación:

bash
Copy
python app.py
Acceder en tu navegador:

Copy
http://localhost:5000
Pegar el texto del correo electrónico y hacer clic en "Analizar"

📊 Dataset
Fuente: Spam Mails Dataset

Contenido:

85,000+ correos etiquetados (phishing/legítimos)

Textos en inglés (funciona bien con español)

Preprocesamiento:

Eliminación de URLs y emails

Normalización de texto

Remoción de stopwords

🧠 Modelo ML
Característica	Detalle
Algoritmo	SVM (Máquinas de Vectores de Soporte)
Vectorización	TF-IDF con bigramas
Parámetros clave	max_features=5000, ngram_range=(1,2)
Precisión	96.8% (test set)
🌐 API
http
Copy
POST /predict
Content-Type: text/plain

"Estimado cliente, su cuenta requiere verificación inmediata: http://phishy-site.com"
Respuesta:

json
Copy
{
  "prediction": "PHISHING",
  "confidence": 98.7,
  "text_preview": "Estimado cliente, su cuenta requiere..."
}

[image](https://github.com/user-attachments/assets/ce592f4b-408c-4649-9157-8064ef9bb26e)
[image](https://github.com/user-attachments/assets/fb84190e-73e8-4dc4-9d55-89e1b8f2ac54)

[image](https://github.com/user-attachments/assets/537fb5d9-4307-4ff1-9bbd-9ac85b33c71a)

[image](https://github.com/user-attachments/assets/74905cac-04ad-4adf-afad-c123ae26a33b)
