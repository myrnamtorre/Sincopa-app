# 💃 Síncopa: Asistente Coreográfico IA

**Síncopa** es un agente conversacional interactivo impulsado por Inteligencia Artificial y Machine Learning, diseñado para asistir a bailarines, coreógrafos y entrenadores en el análisis de pistas musicales, clasificación de géneros, estructuración de rutinas y recomendaciones de acondicionamiento físico y vestuario.

---

## 🚀 Características Principales

* 🎵 **Análisis y Clasificación en Tiempo Real:** Identifica el género musical (Bachata, Salsa, Quebradita) y calcula el tempo aproximado (BPM) e intensidad métrica.
* 🔗 **Procesamiento de Enlaces Externos:** Soporta URLs de **Spotify, YouTube, SoundCloud y Apple Music**, extrayendo automáticamente el título real de la canción mediante scraping (*BeautifulSoup*).
* 📊 **Evaluación Multi-Modalidad:** Calcula métricas de exigencia física (escala 1 a 10) adaptadas según si la rutina se baila en **Pareja, Grupo/Compañía o Solista**.
* 🏋️ **Acondicionamiento Físico Personalizado:** Sugiere rutinas de ejercicio específicas (pliometría, disociación corporal, agilidad de pies) para aguantar el ritmo de cada pista.
* 👗 **Asesoría de Vestuario y Calzado:** Responde preguntas contextuales sobre la vestimenta y el calzado ideal según el ritmo analizado.
* 🎶 **Motor de Sugerencias con Filtros:** Recomienda temas similares filtrando por velocidad o estilo (ej. *bachatas lentas*, *salsas rápidas*).

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.10+
* **Interfaz de Usuario:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Procesamiento de Datos:** Pandas, NumPy
* **Extracción Web (Scraping):** BeautifulSoup4, Requests
* **Serialización:** Joblib

---

## 📂 Estructura del Repositorio

```text
├── app.py                   # Aplicación principal de Streamlit y lógica del bot
├── modelo_sincopa_rf.joblib # Modelo Random Forest entrenado para clasificación
├── Sincopa.ipynb            # Notebook con el flujo de EDA, extracción y entrenamiento
├── requirements.txt         # Lista de dependencias para el entorno de producción
└── README.md                # Documentación del proyecto
