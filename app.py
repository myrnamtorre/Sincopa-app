import os
import tempfile
import librosa
import numpy as np
import pandas as pd
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
    X_train = np.array([
        [175.0, 0.25, 0.08, 0.010, 1.8, -100, 120],
        [180.0, 0.28, 0.09, 0.015, 1.9, -90,  115],
        [125.0, 0.18, 0.06, 0.005, 1.5, -150, 140],
        [130.0, 0.20, 0.07, 0.006, 1.6, -140, 135],
        [160.0, 0.22, 0.07, 0.008, 1.7, -120, 130],
        [165.0, 0.24, 0.08, 0.009, 1.8, -110, 125],
        [105.0, 0.26, 0.06, 0.007, 1.9, -115, 110],
        [110.0, 0.27, 0.07, 0.008, 2.0, -105, 105],
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
    st.session_state.messages = [{"role": "assistant", "content": "👋 **¡Hola! Síncopa - Clasificación Inteligente.**\nPega cualquier enlace de música en el chat para analizar su matriz acústica."}]
if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []

def es_url_valida(texto):
    dominios = ["youtube.com", "youtu.be", "soundcloud.com", "spotify.com", "apple.com"]
    return any(d in texto.strip().lower() for d in dominios) and texto.startswith("http")

# ==========================================
# 3. EXTRACCIÓN Y PROCESAMIENTO DIRECTO
# ==========================================
def analizar_audio_por_enlace(url):
    fd, ruta_salida = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    
 ydl_opts = 
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "outtmpl": ruta_salida.replace(".mp3", ""),
        "quiet": True,
        "nocheckcertificate": True,
        "default_search": "auto", 
        "noplaylist": True,
        # CAMBIO CLAVE: Disfrazamos el cliente como un reproductor web incrustado o iOS para saltar el firewall
        "extractor_args": {"youtube": {"player_client": ["web_embedded", "ios", "android_creator"]}},
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
        }
    

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Extraemos el título real del archivo descargado directamente
            if 'entries' in info:
                info = info['entries'][0]
            titulo_pista = info.get('title', 'Pista Analizada')
        
        archivo_final = ruta_salida.replace(".mp3", "") + ".mp3"
        y, sr = librosa.load(archivo_final, duration=45.0, sr=22050)
        
        if os.path.exists(archivo_final): 
            os.remove(archivo_final)

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
            "cancion_formateada": titulo_pista,
            "features": [tempo_val, rmse, zcr, flatness, beat_strength, mfcc1, mfcc2],
            "tempo": round(tempo_val, 1),
            "densidad_tatum": round(beat_strength * 2, 2)
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
    "Quebradita": "🔥 **Bloque HIIT:** Tabata (20s/10s) Jump Squats y Burpees.",
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
    st.json({"Algoritmo": "RandomForestClassifier", "Features": ["tempo", "rmse", "zcr", "flatness", "beat_strength", "mfcc1", "mfcc2"]})

if prompt := st.chat_input("Pega el enlace de la pista a evaluar..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with tabs[0]:
        with st.chat_message("user"): st.markdown(prompt)

        if es_url_valida(prompt):
            with st.chat_message("assistant"):
                with st.spinner("🎧 Descargando audio y procesando matriz acústica..."):
                    resultado = analizar_audio_por_enlace(prompt)
                
                if "error" in resultado:
                    st.error(f"❌ **Enlace no disponible o bloqueado:** {resultado['error']}\n\n*Por favor, intenta con otro link de la misma canción que sea 100% público.*")
                else:
                    X_input = np.array([resultado["features"]])
                    prediccion = modelo.predict(X_input)[0]

                    if prediccion == "Podcast":
                        reply = f"⚠️ **Audio Rechazado (Contenido No Musical)**\n🎵 *{resultado['cancion_formateada']}*\n\nEl sistema detectó un archivo de voz/podcast. No se realizará la evaluación coreográfica."
                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                    else:
                        par, grp, sol, metrica, aprovechamiento, _ = obtener_detalles_coreograficos(prediccion)
                        rutina = CATALOGO_ENTRENAMIENTO.get(prediccion, "")
                        
                        reply = f"""🎵 **Pista:** **{resultado['cancion_formateada']}**
🏷️ **Clasificación del Modelo:** **{prediccion}** 
⏱️ **Tempo Estimado:** ~{resultado['tempo']} BPM

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
                reply = "💡 Por favor, ingresa un enlace válido."
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
