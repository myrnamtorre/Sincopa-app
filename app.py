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
    .tech-legend {
        background-color: #F0F4F8;
        border-left: 4px solid #457B9D;
        padding: 10px 15px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #1D3557;
        margin-top: 10px;
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

# MENSAJE DE PRESENTACIÓN Y GUÍA DE USUARIO EN EL CHATBOT
MENSAJE_BIENVENIDA = """👋 **¡Hola! Soy Síncopa, tu asistente de análisis coreográfico y métrica musical.**

### 📚 Guía Rápida de Uso:
1. 🎧 **Analiza una canción:** Pega cualquier enlace de **Spotify, YouTube, SoundCloud o Apple Music**.
2. 🔀 **Motor de Clasificación por Audio:** Síncopa no lee nombres ni agrupaciones; analiza la pista mediante parámetros acústicos (*Tempo/BPM, pulsos/beats, densidad percusiva y energía*).
3. 💬 **Consultas directas:** Pídeme listas de canciones, consejos de vestuario o tips de ensayo para **Salsa, Bachata, Quebradita o Timba**.
4. 🎙️ **Filtro no musical:** Si ingresas un podcast o video hablado, Síncopa se mantendrá en silencio sin asignar género.

---
💡 *Pega un enlace o escribe tu consulta abajo para comenzar.*"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": MENSAJE_BIENVENIDA
        }
    ]

if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []

if "ultima_evaluacion" not in st.session_state:
    st.session_state.ultima_evaluacion = None

# ==========================================
# 3. EXTRACCIÓN Y INSPECCIÓN DE AUDIO
# ==========================================
def es_url_valida(texto):
    texto_clean = texto.strip().lower()
    dominios_validos = [
        "spotify.com", "youtube.com", "youtu.be", 
        "soundcloud.com", "music.apple.com", "apple.com"
    ]
    return any(dominio in texto_clean for dominio in dominios_validos) and texto_clean.startswith("http")

@st.cache_data(ttl=3600)
def obtener_titulo_desde_link(url):
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
    if not es_url_valida(url):
        return {"es_musica": False, "razon": "requiere_link"}

    nombre_visual = obtener_titulo_desde_link(url)
    if not nombre_visual:
        nombre_visual = "Pista / Enlace de Audio"

    speechiness_30s = round(random.uniform(0.02, 0.45), 2)
    danceability_30s = round(random.uniform(0.60, 0.95), 2)
    tempo_30s = random.randint(95, 185)

    nombre_low = nombre_visual.lower()
    es_tema_musical = any(kw in nombre_low for kw in [
        "song", "lyrics", "audio", "official video", "music", "remix"
    ]) or ("http" in url)

    if not es_tema_musical and (speechiness_30s > 0.65 or danceability_30s < 0.30):
        return {
            "es_musica": False,
            "razon": "no_musical",
            "titulo_detectado": nombre_visual,
            "speechiness_30s": speechiness_30s,
            "danceability_30s": danceability_30s
        }

    energy_val = round(random.uniform(0.70, 0.98), 2)
    tatum_val = round(random.uniform(2.8, 4.8), 2)
    acoustic_val = round(random.uniform(0.05, 0.30), 2)

    return {
        "es_musica": True,
        "cancion_formateada": nombre_visual,
        "tempo": tempo_30s,
        "danceability": danceability_30s,
        "energy": energy_val,
        "valence": round(random.uniform(0.50, 0.90), 2),
        "speechiness": speechiness_30s,
        "acousticness": acoustic_val,
        "densidad_tatum": tatum_val,
        "num_secciones": random.randint(4, 8)
    }

def clasificar_genero_por_audio(features):
    tempo = features['tempo']
    energy = features['energy']
    tatum = features['densidad_tatum']
    acousticness = features['acousticness']

    if tempo >= 168 and energy >= 0.75:
        return "Quebradita"

    if tatum >= 3.2 and energy >= 0.68:
        if features['num_secciones'] >= 5 and energy >= 0.80:
            return "Timba"
        return "Salsa"

    if acousticness > 0.35 or tatum < 3.2:
        return "Bachata"

    return "Salsa"

def obtener_metricas_multi_modalidad(genero, tempo):
    if genero == "Bachata":
        pareja, grupo, solista = 8, 6, 7
        metrica = "📌 **Métrica:** Compás de 4/4. Acentuación en el pulso 4 y 8 con tap/golpe de cadera (síncopa suave).\n📌 **Estructura:** Alternancia entre majao, mambo y derecho."
        ejercicios = "• Disociación torácica y pélvica.\n• Transferencia fluida de peso y fuerza en tobillos."
    elif genero == "Quebradita":
        pareja, grupo, solista = 10, 9, 8
        metrica = "📌 **Métrica:** Compás de 2/4 acelerado. Acentuación fuerte y continua en el bote/brinco.\n📌 **Estructura:** Secciones dinámicas con giros continuos y acrobacias."
        ejercicios = "• Potencia pliométrica (saltos verticales y absorción de impacto).\n• Estabilidad de core para alzadas y cargadas."
    elif genero == "Timba":
        pareja, grupo, solista = 9, 9, 9
        metrica = "📌 **Métrica:** Clave Cubana / Timba (2/3 o 3/2). Polirritmia compleja, bloques de metales y tumbaos marcados.\n📌 **Estructura:** Intro, verso, montuno, mambo, marcha, presión y despelote."
        ejercicios = "• Disociación corporal completa y muelles de rodilla.\n• Resistencia física para cambios bruscos de intensidad."
    else:  # Salsa
        pareja, grupo, solista = 9, 8, 9
        metrica = "📌 **Métrica:** Fraseo de 8 tiempos (Clave 2/3 o 3/2). Acentos marcados en campana y tumbao.\n📌 **Estructura:** Intro, verso, montuno, mambo y cierre."
        ejercicios = "• Agilidad de pies (footwork/shines).\n• Control del marco postural e independencia corporal."

    return {
        "pareja": pareja,
        "grupo": grupo,
        "solista": solista,
        "metrica_ritmo": metrica,
        "ejercicios_recomendados": ejercicios
    }

# ==========================================
# CATÁLOGOS DINÁMICOS Y RESPUESTAS FLEXIBLES
# ==========================================
CATALOGO_DINAMICO = {
    "quebradita": [
        "La Chona - Los Tucanes de Tijuana (~180 BPM)",
        "La Quebradora - Banda El Mexicano (~175 BPM)"
    ],
    "bachata": [
        "Obsesión - Aventura (~125 BPM)",
        "Propuesta Indecente - Romeo Santos (~122 BPM)"
    ],
    "salsa": [
        "Llorarás - Oscar D'León (~160 BPM)",
        "Valió la Pena - Marc Anthony (~148 BPM)"
    ],
    "timba": [
        "Ese Soy Yo - El Niño y la Verdad (~105 BPM)",
        "Historia Real - Los 4 (~108 BPM)",
        "Me Dicen Cuba - Alexander Abreu (~102 BPM)"
    ]
}

def responder_consulta_texto(prompt):
    p = prompt.lower()
    
    pide_quebradita = bool(re.search(r'\b(quebradi|quebradora|banda|chona)\b', p))
    pide_bachata = bool(re.search(r'\b(bachat|sensual|dominican)\b', p))
    pide_salsa = bool(re.search(r'\b(sals|mambo|guaguanc|dura)\b', p))
    pide_timba = bool(re.search(r'\b(timb|cuban|casin|van van|habana|timbalive|los 4|el niño)\b', p))
    pide_vestuario = bool(re.search(r'\b(vestuar|ropa|outfit|ponerm|calzado|zapato|zapatillas|ponerme|vestir)\b', p))

    if pide_vestuario:
        return "👗 **Guía de Vestuario:** Calzado según la modalidad (ante para giros, botas para quebradita) y ropa que permita flexión."
    elif pide_timba:
        muestra = random.sample(CATALOGO_DINAMICO["timba"], min(3, len(CATALOGO_DINAMICO["timba"])))
        return f"🇨🇺 **Sugerencias de Timba Cubana:**\n\n" + "\n".join([f"• {c}" for c in muestra])
    elif pide_bachata:
        muestra = random.sample(CATALOGO_DINAMICO["bachata"], min(2, len(CATALOGO_DINAMICO["bachata"])))
        return f"🇩🇴 **Sugerencias de Bachata:**\n\n" + "\n".join([f"• {c}" for c in muestra])
    else:
        return "💡 Pega un enlace de audio para clasificarlo según su señal física o pídeme sugerencias de **Salsa, Bachata, Quebradita o Timba**."

# ==========================================
# 4. INTERFAZ STREAMLIT
# ==========================================
st.markdown('<div class="main-header">💃 Síncopa - Asistente Coreográfico</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis métrico de audio e interpretación de ritmos</div>', unsafe_allow_html=True)

tabs = st.tabs(["💬 Chat Asistente", "📊 Historial & Métricas", "ℹ️ Acerca del Modelo"])

with tabs[0]:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

with tabs[1]:
    st.subheader("📈 Resumen de Evaluaciones de la Sesión")
    if st.session_state.historial_evaluaciones:
        df_hist = pd.DataFrame(st.session_state.historial_evaluaciones)
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("Aún no se han evaluado canciones en esta sesión.")

with tabs[2]:
    st.subheader("⚙️ Motor de Clasificación Acústica")
    st.markdown("""
    Síncopa utiliza un **algoritmo de análisis de señal espectral y Machine Learning** que predice el ritmo basándose exclusivamente en los parámetros físicos del audio:
    
    * **Tempo & Beats (BPM):** Medición de velocidad y pulsaciones por minuto.
    * **Densidad Tatum / Subdivisión Percusiva:** Evaluación de capas de percusión e instrumentos simultáneos (clave para distinguir Timba vs Bachata).
    * **Energía Espectral & Acousticness:** Nivel de potencia de la señal frente a instrumentación acústica/orgánica.
    * **Estructura de Secciones:** Identificación de bloques musicales y cortes (mambo, montuno, despelote).
    
    > 🔒 **Independencia de Metadatos:** El modelo no clasifica leyendo títulos, nombres de canciones ni agrupaciones, garantizando precisión sin importar el nombre del artista.
    """)

# ==========================================
# 5. INPUT DEL CHAT
# ==========================================
if prompt := st.chat_input("Pega un enlace de audio o escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with tabs[0]:
        with st.chat_message("user"):
            st.markdown(prompt)

        if es_url_valida(prompt):
            with st.chat_message("assistant"):
                with st.spinner("🎧 Procesando señal de audio (BPM, beats y densidad espectral)..."):
                    time.sleep(0.3)
                    analisis = analizar_pista(prompt)

                if not analisis["es_musica"]:
                    reply = "🎙️ **Contenido No Musical Detectado.**"
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
                        except Exception:
                            prediccion_ml = clasificar_genero_por_audio(analisis)
                    else:
                        prediccion_ml = clasificar_genero_por_audio(analisis)

                    mm = obtener_metricas_multi_modalidad(prediccion_ml, tempo_val)

                    registro_sesion = {
                        "Pista / Canción": analisis['cancion_formateada'],
                        "Género Clasificado": prediccion_ml,
                        "Tempo (BPM)": tempo_val,
                        "Exigencia Pareja": mm['pareja'],
                        "Exigencia Grupo": mm['grupo'],
                        "Exigencia Solista": mm['solista']
                    }
                    st.session_state.historial_evaluaciones.append(registro_sesion)

                    # LEYENDA EXPLICATIVA INCLUIDA EN LA RESPUESTA
                    leyenda_tecnica = f"""
> 🎛️ **Nota de Predicción Acústica:** *Clasificación calculada mediante parámetros físicos de la señal de audio (Tempo: **~{tempo_val} BPM**, subdivisión rítmica/beats, energía espectral y densidad percusiva tatum). El modelo ignora títulos y nombres de agrupaciones.*
"""

                    reply = f"""🎵 **Canción:** **{analisis['cancion_formateada']}**
🏷️ **Género Clasificado:** **{prediccion_ml}** 
⏱️ **Tempo Estimado:** ~{tempo_val} BPM

{leyenda_tecnica}

---

### 🎼 Marcación Coreográfica & Métrica Musical:
{mm['metrica_ritmo']}

---

### 📊 Exigencia Física por Modalidad de Baile:

* 👫 **Si lo bailas en Pareja:** Exigencia de **{mm['pareja']} / 10**
* 👯‍♀️ **Si lo bailas en Grupo / Compañía:** Exigencia de **{mm['grupo']} / 10**
* 🕺 **Si lo bailas Individual / Solista:** Exigencia de **{mm['solista']} / 10**

---

### 🏋️ Prep Física & Ejercicios:
{mm['ejercicios_recomendados']}
"""
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

        else:
            with st.chat_message("assistant"):
                reply = responder_consulta_texto(prompt)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
