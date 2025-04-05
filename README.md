# 📧 Detector de Correos Phishing con Machine Learning

Proyecto completo para clasificar correos electrónicos como phishing o legítimos usando un modelo SVM entrenado con múltiples datasets públicos.

![Interfaz del detector](https://via.placeholder.com/800x400.png?text=Interfaz+Web+del+Detector)

## 🌟 Características Principales
- **Clasificación en tiempo real** con porcentaje de confianza
- **Modelo eficiente** (SVM + TF-IDF) con 96-97% de precisión
- **Interfaz web intuitiva** con Flask
- **Detección de patrones** comunes de phishing:
  - URLs sospechosas
  - Lenguaje urgente
  - Solicitudes de información personal
  - Errores gramaticales

## 📥 Instalación

### Requisitos
- Python 3.10+
- Pip
- Dataset de [Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)

### Pasos
1. Clonar repositorio:
git clone [https://github.com/tu-usuario/phishing-detector.git](https://github.com/vizadrex/Detector-de-Correos-Phishing.git)
cd phishing-detector
Instalar dependencias:

bash
Copy
pip install -r requirements.txt
Colocar el dataset (spam_emails.zip) en la raíz del proyecto

Entrenar el modelo:

bash
Copy
python train_model.py
Iniciar la aplicación:

bash
Copy
python app.py
🖥️ Uso
Acceder a http://localhost:5000

Pegar el texto del correo electrónico en el área de texto

Hacer clic en "Analizar"

Ejemplo de entrada:

Copy
¡Ganador! Su cuenta ha sido seleccionada para un premio de $1,000,000. 
Haga clic aquí: http://premio-falso.com/claim?user=123
Salida esperada:

Copy
✅ Resultado:
- Clasificación: PHISHING
- Confianza: 98.3%
🧠 Estructura del Proyecto
Copy
phishing-detector/
├── app.py              # Aplicación Flask
├── train_model.py      # Script de entrenamiento
├── model/              # Modelo entrenado
│   ├── spam_model.pkl
│   └── vectorizer.pkl
├── templates/          # Vistas HTML
│   └── index.html
├── static/             # Estilos CSS
│   └── style.css
└── requirements.txt    # Dependencias
🔧 Configuración del Modelo
Componente	Detalle
Algoritmo	Support Vector Machine (SVM)
Vectorización	TF-IDF con bigramas
Parámetros clave	max_features=5000, min_df=2
Precisión	96.8% (validación)
Tiempo de respuesta	< 100ms
📊 Dataset
Fuente: Spam Mails Dataset

Contenido:

85,000+ correos etiquetados

Combinación de múltiples fuentes (Enron, TREC, etc.)

Preprocesamiento:

Eliminación de stopwords

Normalización de texto

Detección automática de columnas

🌐 API
http
Copy
POST /predict HTTP/1.1
Content-Type: text/plain

"Estimado cliente: Su cuenta PayPal necesita verificación urgente. Acceda aquí: http://paypal-fake.com/verify"
Respuesta:

json
Copy
{
  "prediction": "PHISHING",
  "confidence": 97.45,
  "text_preview": "Estimado cliente: Su cuenta PayPal necesita..."
}
🛠️ Troubleshooting
Error común: No such file or directory: 'model/spam_model.pkl'
Solución:

bash
Copy
# 1. Eliminar archivos antiguos
rmdir /s /q dataset model

# 2. Volver a entrenar el modelo
python train_model.py
🤝 Contribuir
Haz fork del proyecto

Crea tu feature branch:

bash
Copy
git checkout -b feature/nueva-funcionalidad
Haz commit de tus cambios:

bash
Copy
git commit -m 'Agrega nueva funcionalidad'
Push a la branch:

bash
Copy
git push origin feature/nueva-funcionalidad
Abre un Pull Request
