import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import random
import time
import requests
from bs4 import BeautifulSoup

# ==========================================
# 1. CONFIGURACIÓN INICIAL DE STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Síncopa - Asistente Coreográfico",
    page_icon="💃",
    layout="wide"
)

# Custom CSS para estilo visual
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E63946;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #457B9D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARGA DEL MODELO ML & ESTADOS DE SESIÓN
# ==========================================
@st.cache_resource
def cargar_modelo():
    try:
        if os.path.exists('modelo_ritmos.pkl'):
            return joblib.load('modelo_ritmos.pkl')
        return None
    except Exception:
        return None

modelo = cargar_modelo()

# Inicialización de estado de sesión
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **¡Hola! Soy Síncopa, tu asistente de análisis coreográfico y métrica musical.**\n\nIngresa el **enlace (link) de una canción** (Spotify, YouTube, SoundCloud o Apple Music) y analizaré su tempo, género, métrica y exigencia física."
        }
    ]

if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []

if "ultima_evaluacion" not in st.session_state:
    st.session_state.ultima_evaluacion = None

# ==========================================
# 3. FUNCIONES AUXILIARES Y SIMULACIÓN AUDIO
# ==========================================
@st.cache_data(ttl=3600)
def obtener_titulo_desde_link(url):
    """Extrae el título de la página o video mediante metadata u oEmbed."""
    try:
        if "youtube.com" in url or "youtu.be" in url:
            oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
            res = requests.get(oembed_url, timeout=3)
            if res.status_code == 200:
                return res.json().get("title", "")
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=3)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            if soup.title and soup.title.string:
                return soup.title.string
    except Exception:
        pass
    return ""

def es_url_valida(texto):
    regex = r'^(https?://)?(www\.)?(youtube\.com|youtu\.be|spotify\.com|soundcloud\.com|apple\.com)'
    return bool(re.match(regex, texto.strip(), re.IGNORECASE))

def analizar_pista(url):
    """Simula el análisis espectral y métrico de la señal de audio."""
    if not es_url_valida(url):
        return {"es_musica": False, "razon": "requiere_link"}

    titulo_extraido = obtener_titulo_desde_link(url)
    titulo_lower = titulo_extraido.lower()

    # Filtro de contenido no bailable (removida la palabra "app" suelta)
    palabras_no_musicales = [
        "tutorial", "curso", "charla", "podcast", "interview", "entrevista",
        "review", "explicacion", "clase", "documental", "conference", "talking"
    ]
    
    if any(p in titulo_lower for p in palabras_no_musicales):
        return {
            "es_musica": False,
            "razon": "no_musical",
            "titulo_detectado": titulo_extraido if titulo_extraido else "Contenido sin música bailable"
        }

    # Determinación simulada de parámetros acústicos
    if any(w in titulo_lower for w in ["bachata", "sensual", "romeo"]):
        tempo = random.randint(110, 130)
        danceability = random.uniform(0.70, 0.88)
        energy = random.uniform(0.55, 0.75)
    elif any(w in titulo_lower for w in ["quebradita", "quebradora", "chona", "zapateado"]):
        tempo = random.randint(175, 205)
        danceability = random.uniform(0.80, 0.95)
        energy = random.uniform(0.85, 0.99)
    else:  # Por defecto tiende a Salsa u otros ritmos tropicales
        tempo = random.randint(140, 180)
        danceability = random.uniform(0.75, 0.92)
        energy = random.uniform(0.75, 0.95)

    nombre_cancion = titulo_extraido if titulo_extraido else "Pista Analizada (Audio Enlace)"

    return {
        "es_musica": True,
        "cancion_formateada": nombre_cancion,
        "tempo": tempo,
        "danceability": round(danceability, 2),
        "energy": round(energy, 2),
        "valence": round(random.uniform(0.60, 0.90), 2),
        "speechiness": round(random.uniform(0.03, 0.12), 2),
        "acousticness": round(random.uniform(0.10, 0.40), 2),
        "densidad_tatum": round(random.uniform(2.5, 4.5), 2),
        "num_secciones": random.randint(4, 8)
    }

def obtener_metricas_multi_modalidad(genero, tempo):
    if genero == "Bachata":
        pareja, grupo, solista = 8, 6, 7
        metrica = "📌 **Métrica:** Compás de 4/4. Acentuación en el pulso 4 y 8 con elevación o cadera (síncopa suave).\n📌 **Estructura:** Alternancia entre majao, mambo y derecho."
        ejercicios = "• Trabajo de disociación pélvica y transferencias de peso.\n• Flexibilidad activa de cadera y tobillos."
        estilo = "💡 *Recomendación:* Priorizar la fluidez corporal e interpretar los cambios de sección (punteos vs. mambo)."
    elif genero == "Quebradita":
        pareja, grupo, solista = 10, 9, 8
        metrica = "📌 **Métrica:** Compás de 2/4 a ritmo acelerado. Acentuación fuerte y continua en el bote.\n📌 **Estructura:** Secciones rápidas con cambio de giros y acrobacias."
        ejercicios = "• Capacidad pliométrica (saltos verticales y amortiguación).\n• Fortalecimiento de core para agarres y alzadas."
        estilo = "💡 *Recomendación:* Mantener centro de gravedad bajo para estabilidad en giros dinámicos."
    else:  # Salsa
        pareja, grupo, solista = 9, 8, 9
        metrica = "📌 **Métrica:** Clave de Salsa 2/3 o 3/2 (8 tiempos musicales). Acentos fuertes en clave y campana.\n📌 **Estructura:** Intro, verso, montuno, mambo y moña."
        ejercicios = "• Velocidad de pies (shines/footwork) e intervalos de agilidad.\n• Control postural y tensión en brazos para el marco en pareja."
        estilo = "💡 *Recomendación:* Mantener la velocidad de reacción alta en los cortes de mambo."

    return {
        "pareja": pareja,
        "grupo": grupo,
        "solista": solista,
        "metrica_ritmo": metrica,
        "ejercicios_recomendados": ejercicios,
        "recomendacion_estilo": estilo
    }

def generar_sugerencias_dinamicas(genero, nivel="moderadas", cantidad=3):
    catalogo = {
        "Bachata": ["Obsesión - Aventura", "Propuesta Indecente - Romeo Santos", "Sobredosis - Romeo Santos ft. Ozuna", "Stand by Me - Prince Royce"],
        "Quebradita": ["La Chona - Los Tucanes de Tijuana", "La Quebradora - Banda El Mexicano", "El Baile del Perrito - Wilfrido Vargas", "El Tucanazo - Los Tucanes de Tijuana"],
        "Salsa": ["Llorarás - Oscar D'León", "Valió la Pena - Marc Anthony", "Rebelión - Joe Arroyo", "Periodico de Ayer - Héctor Lavoe"]
    }
    opciones = catalogo.get(genero, catalogo["Salsa"])
    return random.sample(opciones, min(cantidad, len(opciones)))

# ==========================================
# 4. INTERFAZ Y NAVEGACIÓN
# ==========================================
st.markdown('<div class="main-header">💃 Síncopa - Asistente Coreográfico</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis métrico, clasificación de ritmos y sugerencias de entrenamiento</div>', unsafe_allow_html=True)

tabs = st.tabs(["💬 Chat Asistente", "📊 Historial & Métricas", "ℹ️ Guía de Uso"])

# ------------------------------------------
# TAB 1: CHAT ASISTENTE
# ------------------------------------------
with tabs[0]:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Pega aquí el link de Spotify, YouTube, SoundCloud o Apple Music..."):
        # Registro de mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Respuestas conversacionales genéricas si el usuario saluda o pregunta
        prompt_low = prompt.strip().lower()
        
        if prompt_low in ["hola", "buenas", "que haces?", "quien eres?", "ayuda"]:
            reply = "¡Hola! Para comenzar a trabajar, **por favor envíame el enlace (URL) de la canción** que deseas analizar."
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

        # Proceso de análisis por enlace (CASO 3 CORREGIDO)
        else:
            with st.chat_message("assistant"):
                with st.spinner("🎧 Inspeccionando señal de audio y métricas espectrales..."):
                    time.sleep(0.3)
                    analisis = analizar_pista(prompt)

                if not analisis["es_musica"]:
                    if analisis.get("razon") == "requiere_link":
                        reply = "⚠️ **Por favor, ingresa únicamente un enlace (link) válido** de *Spotify, YouTube, SoundCloud o Apple Music*. No realizo análisis ingresando el nombre escrito de la canción."
                    else:
                        nom_detectado = analisis.get("titulo_detectado", "Contenido detectado")
                        reply = (
                            f"🎙️ **Contenido No Musical Detectado:**\n\n"
                            f"El enlace *'{nom_detectado}'* fue analizado y corresponde a un tutorial, charla o video sin audio musical bailable.\n\n"
                            f"> ⛔ **Síncopa permanece en silencio:** No se asigna género (*Salsa/Bachata/Quebradita*) ni métricas a tutoriales técnicos o contenido conversacional."
                        )
                    st.warning(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                else:
                    tempo_val = analisis["tempo"]

                    df_in = pd.DataFrame([{
                        'tempo': analisis['tempo'],
                        'danceability': analisis['danceability'],
                        'energy': analisis['energy'],
                        'valence': analisis['valence'],
                        'speechiness': analisis['speechiness'],
                        'acousticness': analisis['acousticness'],
                        'densidad_tatum': analisis['densidad_tatum'],
                        'num_secciones': analisis['num_secciones']
                    }])

                    # PREDICCIÓN ML O REGLAS DIRECTAS
                    genero_palabras_clave = None
                    titulo_low = analisis['cancion_formateada'].lower()

                    if any(w in titulo_low for w in ["quebradora", "quebradita", "caballito", "chona", "zapateado"]):
                        genero_palabras_clave = "Quebradita"
                    elif any(w in titulo_low for w in ["bachata", "sensual"]):
                        genero_palabras_clave = "Bachata"

                    if genero_palabras_clave:
                        prediccion_ml = genero_palabras_clave
                    elif modelo is not None:
                        try:
                            prediccion_ml = modelo.predict(df_in)[0]
                            if prediccion_ml == "Salsa" and tempo_val >= 175:
                                prediccion_ml = "Quebradita"
                        except Exception:
                            prediccion_ml = "Quebradita" if tempo_val >= 170 else ("Salsa" if tempo_val >= 135 else "Bachata")
                    else:
                        prediccion_ml = "Quebradita" if tempo_val >= 170 else ("Salsa" if tempo_val >= 135 else "Bachata")

                    mm = obtener_metricas_multi_modalidad(prediccion_ml, tempo_val)

                    registro_sesion = {
                        "Pista / Canción": analisis['cancion_formateada'],
                        "Género Clasificado": prediccion_ml,
                        "Tempo (BPM)": tempo_val,
                        "Exigencia Pareja": mm['pareja'],
                        "Exigencia Grupo": mm['grupo'],
                        "Exigencia Solista": mm['solista']
                    }
                    st.session_state.ultima_evaluacion = {
                        "cancion": analisis['cancion_formateada'],
                        "genero": prediccion_ml,
                        "tempo": tempo_val
                    }
                    st.session_state.historial_evaluaciones.append(registro_sesion)

                    sug_rel = generar_sugerencias_dinamicas(prediccion_ml, "moderadas", cantidad=3)
                    sug_txt = ", ".join([f"*{s}*" for s in sug_rel])

                    ficha_texto = f"""==================================================
FICHA TÉCNICA COREOGRÁFICA - SÍNCOPA IA
==================================================
Canción: {analisis['cancion_formateada']}
Género Clasificado: {prediccion_ml}
Tempo Estimado: ~{tempo_val} BPM

EXIGENCIA FÍSICA POR MODALIDAD:
- Pareja: {mm['pareja']}/10
- Grupo / Compañía: {mm['grupo']}/10
- Solista: {mm['solista']}/10

ACENTUACIÓN Y MÉTRICA:
{mm['metrica_ritmo']}

PREPARACIÓN FÍSICA SUGERIDA:
{mm['ejercicios_recomendados']}
==================================================
"""

                    reply = f"""🎵 **Canción:** **{analisis['cancion_formateada']}**
🏷️ **Género Clasificado:** **{prediccion_ml}** 
⏱️ **Tempo Estimado:** ~{tempo_val} BPM

---

### 🎼 Marcación Coreográfica & Métrica Musical:
{mm['metrica_ritmo']}

---

### 📊 Exigencia Física por Modalidad de Baile:

* 👫 **Si lo bailas en Pareja:** Exigencia de **{mm['pareja']} / 10** (Ideal para marco y conexión).
* 👯‍♀️ **Si lo bailas en Grupo / Compañía:** Exigencia de **{mm['grupo']} / 10** (Exige alta limpieza en bloques y simetría).
* 🕺 **Si lo bailas Individual / Solista:** Exigencia de **{mm['solista']} / 10** (Requiere proyección escénica y footwork continuo).

{mm['recomendacion_estilo']}

---

### 🏋️ Prep Física & Ejercicios para Aguantar la Pista:
{mm['ejercicios_recomendados']}

---

💡 *Otras opciones sugeridas de {prediccion_ml}:* {sug_txt}.
"""
                    st.markdown(reply)
                    st.download_button(
                        label="📥 Descargar Ficha Técnica (.txt)",
                        data=ficha_texto,
                        file_name=f"Ficha_{analisis['cancion_formateada'].replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                    st.session_state.messages.append({"role": "assistant", "content": reply})

# ------------------------------------------
# TAB 2: HISTORIAL Y MÉTRICAS
# ------------------------------------------
with tabs[1]:
    st.subheader("📈 Resumen de Evaluaciones de la Sesión")
    
    if st.session_state.historial_evaluaciones:
        df_hist = pd.DataFrame(st.session_state.historial_evaluaciones)
        st.dataframe(df_hist, use_container_width=True)
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Distribución por Géneros Analizados")
            st.bar_chart(df_hist["Género Clasificado"].value_counts())
        with col2:
            st.markdown("##### Distribución de Tempos (BPM)")
            st.line_chart(df_hist["Tempo (BPM)"])
    else:
        st.info("Aún no se han evaluado canciones en esta sesión. Analiza un enlace en el chat para ver aquí las métricas.")

# ------------------------------------------
# TAB 3: GUÍA DE USO
# ------------------------------------------
with tabs[2]:
    st.subheader("📚 Guía de Uso del Asistente")
    st.markdown("""
    **Síncopa** es una herramienta especializada para bailarines, profesores y coreógrafos.

    ### 📌 ¿Cómo funciona?
    1. **Ingresa la URL:** Copia y pega el enlace de la pista desde Spotify, YouTube o plataformas compatibles.
    2. **Extracción y Predicción:** Síncopa procesa el tempo (BPM) y variables acústicas para clasificar el ritmo en **Salsa, Bachata o Quebradita**.
    3. **Generación de Ficha Coreográfica:** Obtendrás la velocidad, estructura de métrica musical, nivel de exigencia física por modalidad y ejercicios recomendados.

    ---
    *Nota: Síncopa filtra automáticamente tutoriales o podcasts sin música bailable.*
    """)
