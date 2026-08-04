import os
import tempfile
import librosa
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yt_dlp

# ==========================================
# 1. CONFIGURACIÓN INICIAL
# ==========================================
st.set_page_config(page_title="Síncopa - Asistente Coreográfico", page_icon="💃", layout="wide")
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #E63946; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #457B9D; text-align: center; margin-bottom: 1.5rem; }
    .stChatMessage { border-radius: 12px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ENTRENAMIENTO DINÁMICO (FEATURES REALES)
# ==========================================
@st.cache_resource
def cargar_modelo_en_memoria():
    # Features: [tempo, rmse, zcr, flatness, beat_strength, mfcc1, mfcc2]
    # Se entrena con perfiles acústicos distintivos, incluyendo una clase específica para "Podcast"
    X_train = np.array([
        # Quebradita (Rápido, alta energía, metales fuertes)
        [175.0, 0.25, 0.08, 0.010, 1.8, -100, 120],
        [180.0, 0.28, 0.09, 0.015, 1.9, -90,  115],
        # Bachata (Velocidad media, acústico, ataque percusivo claro)
        [125.0, 0.18, 0.06, 0.005, 1.5, -150, 140],
        [130.0, 0.20, 0.07, 0.006, 1.6, -140, 135],
        # Salsa (Rápido, brillante, polirritmia densa)
        [160.0, 0.22, 0.07, 0.008, 1.7, -120, 130],
        [165.0, 0.24, 0.08, 0.009, 1.8, -110, 125],
        # Timba (Más lento o doble tiempo, bajo pesado, percusión compleja)
        [105.0, 0.26, 0.06, 0.007, 1.9, -115, 110],
        [110.0, 0.27, 0.07, 0.008, 2.0, -105, 105],
        # Podcast / Voz (ZCR alto por consonantes, RMS bajo/variable, fuerza de beat nula)
        [110.0, 0.05, 0.15, 0.050, 0.5, -250, 80],
        [150.0, 0.08, 0.18, 0.060, 0.6, -230, 75],
        [90.0,  0.04, 0.12, 0.040, 0.4, -260, 85]
    ])
    
    y_train = np.array([
        "Quebradita", "Quebradita", 
        "Bachata", "Bachata", 
        "Salsa", "Salsa", 
        "Timba", "Timba", 
        "Podcast", "Podcast", "Podcast"
    ])

    modelo_optimo = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced")
    modelo_optimo.fit(X_train, y_train)
    return modelo_optimo

modelo = cargar_modelo_en_memoria()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 **¡Hola! Síncopa - Clasificación Inteligente.**\nPega un enlace de audio para extraer features reales (MFCC, RMS, ZCR) y clasificar el género."}]
if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []

def es_url_valida(texto):
    dominios = ["youtube.com", "youtu.be", "soundcloud.com"]
    return any(d in texto.strip().lower() for d in dominios) and texto.startswith("http")

@st.cache_data(ttl=3600)
def obtener_titulo_desde_link(url):
    try:
        if "youtube.com" in url or "youtu.be" in url:
            res = requests.get(f"https://www.youtube.com/oembed?url={url}&format=json", timeout=3)
            if res.status_code == 200: return res.json().get("title", "Audio")
    except: pass
    return "Pista Analizada"

# ==========================================
# 3. EXTRACCIÓN REAL DE FEATURES (FEATURE ENGINEERING)
# ==========================================
def analizar_audio_para_modelo(url):
    nombre_visual = obtener_titulo_desde_link(url)
    fd, ruta_salida = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "outtmpl": ruta_salida.replace(".mp3", ""),
        "quiet": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        archivo_final = ruta_salida.replace(".mp3", "") + ".mp3"
        # Analizamos 45 segundos para tener una muestra estadística robusta
        y, sr = librosa.load(archivo_final, duration=45.0, sr=22050)
        if os.path.exists(archivo_final): os.remove(archivo_final)

        # Extracción de variables acústicas reales
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
        
        rmse = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strength = float(np.mean(onset_env))
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
        mfcc1 = float(np.mean(mfccs[0]))
        mfcc2 = float(np.mean(mfccs[1]))

        return {
            "cancion_formateada": nombre_visual,
            "features": [tempo_val, rmse, zcr, flatness, beat_strength, mfcc1, mfcc2],
            "tempo": round(tempo_val, 1),
            "densidad_tatum": round(beat_strength * 2, 2) # Proxy visual para la UI
        }

    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 4. LÓGICA DE UI Y RESPUESTAS
# ==========================================
def obtener_detalles_coreograficos(genero):
    datos = {
        "Bachata": (8, 6, 7, "Compás 4/4. Acento en pulso 4 y 8 con tap/cadera.", "Conexión corporal y marco fluido.", "Ropa estilizada."),
        "Quebradita": (10, 9, 8, "Compás 2/4. Acento constante en el bote.", "Acrobacias y giros veloces.", "Ropa vaquera y botas."),
        "Timba": (9, 9, 9, "Clave Cubana (2/3 o 3/2). Polirritmia compleja.", "Nudos Casino y despelote.", "Ropa urbana deportiva."),
        "Salsa": (9, 8, 9, "Fraseo 8 tiempos. Acentos en campana.", "Shines rápidos y giros en eje.", "Ropa semi-formal.")
    }
    return datos.get(genero, (0,0,0,"","",""))

CATALOGO_ENTRENAMIENTO = {
    "Quebradita": "🔥 **Bloque HIIT:** Tabata (20s/10s) Jump Squats y Burpees. Fortalecimiento de gemelos.",
    "Bachata": "🔥 **Bloque Core:** Tabata Planchas con rotación. Puente de glúteos unilateral.",
    "Salsa": "🔥 **Bloque Agilidad:** 5x45s skipping alto. Desplantes dinámicos alternados.",
    "Timba": "🔥 **Bloque Polirritmia:** Sentadillas sumo con toque. Planchas tocando hombros."
}

st.markdown('<div class="main-header">💃 Síncopa - Asistente Coreográfico</div>', unsafe_allow_html=True)
tabs = st.tabs(["💬 Chat Asistente", "📊 Historial", "⚙️ Modelo"])

with tabs[0]:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

with tabs[1]:
    if st.session_state.historial_evaluaciones:
        st.dataframe(pd.DataFrame(st.session_state.historial_evaluaciones), use_container_width=True)

with tabs[2]:
    st.json({"Algoritmo": "RandomForestClassifier", "Features": ["tempo", "rmse", "zcr", "flatness", "beat_strength", "mfcc1", "mfcc2"], "Clases": list(modelo.classes_)})

if prompt := st.chat_input("Pega un enlace de YouTube..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with tabs[0]:
        with st.chat_message("user"): st.markdown(prompt)

        if es_url_valida(prompt):
            with st.chat_message("assistant"):
                with st.spinner("🎧 Extrayendo MFCCs y variables acústicas..."):
                    resultado = analizar_audio_para_modelo(prompt)
                
                if "error" in resultado:
                    st.error(f"Error procesando audio: {resultado['error']}")
                else:
                    X_input = np.array([resultado["features"]])
                    prediccion = modelo.predict(X_input)[0]

                    if prediccion == "Podcast":
                        reply = f"⚠️ **Contenido No Musical Detectado**\n\n🎵 **Pista:** *{resultado['cancion_formateada']}*\n\n*(El clasificador identificó firmas acústicas de voz hablada/podcast: alta tasa de cruce por cero y baja fuerza de pulso rítmico).* "
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                    else:
                        par, grp, sol, metrica, aprovechamiento, vestuario = obtener_detalles_coreograficos(prediccion)
                        rutina = CATALOGO_ENTRENAMIENTO.get(prediccion, "")
                        
                        reply = f"""🎵 **Pista:** **{resultado['cancion_formateada']}**
🏷️ **Clasificación del Modelo:** **{prediccion}** 
⏱️ **Tempo Estimado:** ~{resultado['tempo']} BPM
📊 **Fuerza de Pulso (Proxy Densidad):** {resultado['densidad_tatum']}

---
### 🎼 Marcación Coreográfica:
{metrica}

### 📊 Calificación:
* 👫 Pareja: {par}/10 | 👯‍♀️ Grupo: {grp}/10 | 🕺 Solista: {sol}/10

### 💡 Aprovechamiento:
{aprovechamiento}

---
{rutina}
"""
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        st.session_state.historial_evaluaciones.append({"Canción": resultado['cancion_formateada'], "Género": prediccion, "Tempo": resultado['tempo']})
        else:
            with st.chat_message("assistant"):
                reply = "💡 Por favor, pega un enlace válido de YouTube o SoundCloud."
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
