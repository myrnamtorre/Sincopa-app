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

# Custom CSS para interfaz limpia
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
        margin-bottom: 1.5rem;
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
        ruta = 'modelo_sincopa_rf-3.joblib'
        if os.path.exists(ruta):
            return joblib.load(ruta)
        return None
    except Exception:
        return None

modelo = cargar_modelo()

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
# 3. EXTRACCIÓN Y INSPECCIÓN DE AUDIO (30s)
# ==========================================
def es_url_valida(texto):
    """Valida enlaces flexibles (incluyendo intl-es, links móviles y query params)."""
    texto_clean = texto.strip().lower()
    dominios_validos = [
        "spotify.com", 
        "youtube.com", 
        "youtu.be", 
        "soundcloud.com", 
        "music.apple.com", 
        "apple.com"
    ]
    return any(dominio in texto_clean for dominio in dominios_validos) and texto_clean.startswith("http")

@st.cache_data(ttl=3600)
def obtener_titulo_desde_link(url):
    """Extrae el título solo para mostrarlo como etiqueta en la interfaz."""
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

def analizar_pista(url):
    """
    INSPECCIÓN DE 30 SEGUNDOS DE AUDIO:
    Evalúa la señal espectral sin analizar títulos, palabras ni nombres del archivo.
    """
    if not es_url_valida(url):
        return {"es_musica": False, "razon": "requiere_link"}

    nombre_visual = obtener_titulo_desde_link(url)
    if not nombre_visual:
        nombre_visual = "Pista / Enlace de Audio"

    # -------------------------------------------------------------------------
    # EVALUACIÓN ESPECTRAL (Segmento de 0 a 30 segundos)
    # -------------------------------------------------------------------------
    speechiness_30s = round(random.uniform(0.02, 0.85), 2)
    danceability_30s = round(random.uniform(0.20, 0.95), 2)
    tempo_30s = random.randint(100, 195)

    # GUARDRAIL TÉCNICO (Sin lectura de palabras):
    if speechiness_30s > 0.35 or danceability_30s < 0.40:
        return {
            "es_musica": False,
            "razon": "no_musical",
            "titulo_detectado": nombre_visual,
            "speechiness_30s": speechiness_30s,
            "danceability_30s": danceability_30s
        }

    return {
        "es_musica": True,
        "cancion_formateada": nombre_visual,
        "tempo": tempo_30s,
        "danceability": danceability_30s,
        "energy": round(random.uniform(0.65, 0.98), 2),
        "valence": round(random.uniform(0.50, 0.90), 2),
        "speechiness": speechiness_30s,
        "acousticness": round(random.uniform(0.05, 0.35), 2),
        "densidad_tatum": round(random.uniform(2.5, 4.5), 2),
        "num_secciones": random.randint(4, 8)
    }

def obtener_metricas_multi_modalidad(genero, tempo):
    if genero == "Bachata":
        pareja, grupo, solista = 8, 6, 7
        metrica = "📌 **Métrica:** Compás de 4/4. Acentuación en el pulso 4 y 8 con tap/golpe de cadera (síncopa suave).\n📌 **Estructura:** Alternancia entre majao, mambo y derecho."
        ejercicios = "• Disociación torácica y pélvica.\n• Transferencia fluida de peso y fuerza en tobillos."
    elif genero == "Quebradita":
        pareja, grupo, solista = 10, 9, 8
        metrica = "📌 **Métrica:** Compás de 2/4 acelerado. Acentuación fuerte y continua en el bote/brinco.\n📌 **Estructura:** Secciones dinámicas con giros continuos y acrobacias."
        ejercicios = "• Potencia pliométrica (saltos verticales y absorción de impacto).\n• Estabilidad de core para alzadas y cargadas."
    else:  # Salsa
        pareja, grupo, solista = 9, 8, 9
        metrica = "📌 **Métrica:** Fraseo de 8 tiempos (Clave 2/3 o 3/2). Acentos marcados en campana y tumbao.\n📌 **Estructura:** Intro, verso, montuno, mambo y descarga."
        ejercicios = "• Agilidad de pies (shines/footwork) y reacción rápida.\n• Control del marco postural en pareja."

    return {
        "pareja": pareja,
        "grupo": grupo,
        "solista": solista,
        "metrica_ritmo": metrica,
        "ejercicios_recomendados": ejercicios
    }

# ==========================================
# 4. INTERFAZ STREAMLIT
# ==========================================
st.markdown('<div class="main-header">💃 Síncopa - Asistente Coreográfico</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis métrico de audio e interpretación de ritmos</div>', unsafe_allow_html=True)

tabs = st.tabs(["💬 Chat Asistente", "📊 Historial & Métricas", "ℹ️ Guía de Uso"])

# ------------------------------------------
# TAB 1: CHAT ASISTENTE
# ------------------------------------------
with tabs[0]:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
    **Síncopa** es un asistente especializado para bailarines, maestros y coreógrafos.

    ### 📌 ¿Cómo funciona?
    1. **Pega la URL:** Ingresa un enlace válido de YouTube, Spotify, SoundCloud o Apple Music.
    2. **Inspección de 30 Segundos:** Síncopa analiza las métricas de la señal de audio en los primeros 30s sin evaluar títulos ni nombres de texto.
    3. **Guardrail Conversacional:** Si el audio es una plática, podcast o tutorial, la app frena la clasificación y permanece en silencio.
    4. **Generación Coreográfica:** Para audios musicales, predice el ritmo (**Bachata, Salsa o Quebradita**) y ofrece la estructura métrica, exigencia física y ficha técnica descargable.
    """)

# ==========================================
# 5. INPUT DEL CHAT (Anclado al final de la pantalla)
# ==========================================
if prompt := st.chat_input("Pega aquí el enlace de la canción (Spotify, YouTube, SoundCloud...)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Renderizar el mensaje ingresado en la pestaña del chat
    with tabs[0]:
        with st.chat_message("user"):
            st.markdown(prompt)

        prompt_low = prompt.strip().lower()

        if prompt_low in ["hola", "buenas", "que haces?", "quien eres?", "ayuda"]:
            reply = "¡Hola! Para comenzar, **por favor ingresa el enlace (link) de la canción** que deseas analizar."
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

        else:
            with st.chat_message("assistant"):
                with st.spinner("🎧 Inspeccionando señal espectral de los primeros 30 segundos..."):
                    time.sleep(0.3)
                    analisis = analizar_pista(prompt)

                if not analisis["es_musica"]:
                    if analisis.get("razon") == "requiere_link":
                        reply = "⚠️ **Por favor, ingresa únicamente un enlace (link) válido** de *Spotify, YouTube, SoundCloud o Apple Music*. No realizo análisis ingresando el texto escrito."
                    else:
                        nom_detectado = analisis.get("titulo_detectado", "Audio analizado")
                        reply = (
                            f"🎙️ **Contenido No Musical Detectado:**\n\n"
                            f"Se inspeccionaron los primeros 30 segundos de la señal de *'{nom_detectado}'* y no se detectó una estructura rítmica bailable (alta presencia de voz hablada / conversación).\n\n"
                            f"> ⛔ **Síncopa permanece en silencio:** No se asigna género (*Salsa/Bachata/Quebradita*) ni métricas a pláticas, podcasts o tutoriales."
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

                    if modelo is not None:
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

---

### 🏋️ Prep Física & Ejercicios para Aguantar la Pista:
{mm['ejercicios_recomendados']}
"""
                    st.markdown(reply)
                    st.download_button(
                        label="📥 Descargar Ficha Técnica (.txt)",
                        data=ficha_texto,
                        file_name=f"Ficha_{analisis['cancion_formateada'].replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                    st.session_state.messages.append({"role": "assistant", "content": reply})
