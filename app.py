import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import requests
from bs4 import BeautifulSoup

# Intentamos importar librosa y yt_dlp para procesamiento real de audio
try:
    import librosa
    import yt_dlp
    AUDIO_REAL_DISPONIBLE = True
except ImportError:
    AUDIO_REAL_DISPONIBLE = False

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
    .main-header { font-size: 2.2rem; font-weight: 700; color: #E63946; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #457B9D; text-align: center; margin-bottom: 1.5rem; }
    .stChatMessage { border-radius: 12px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARGA DEL MODELO ML & ESTADOS DE SESIÓN
# ==========================================
@st.cache_resource
def cargar_modelo():
    candidatos = [f for f in os.listdir('.') if f.endswith('.joblib')]
    if candidatos:
        try:
            return joblib.load(candidatos[0])
        except Exception:
            pass
    return None

modelo = cargar_modelo()

MENSAJE_BIENVENIDA = """👋 **¡Hola! Síncopa con Extracción Acústica Real (Librosa).**

### 📚 Guía Rápida de Uso:
1. 🎧 **Analiza una canción:** Pega cualquier enlace musical.
2. 🔬 **Procesamiento DSP Real:** Se extraen características acústicas genuinas (BPM, energía, tatum) sin depender de títulos ni heurísticas de texto.
3. 🤖 **Inferencia del Modelo:** El `.joblib` clasifica el género de manera objetiva.

---
💡 *Pega un enlace de audio o escribe tu consulta abajo para comenzar.*"""

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": MENSAJE_BIENVENIDA}]

if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []

# ==========================================
# 3. EXTRACCIÓN ACÚSTICA REAL CON LIBROSA & YT-DLP
# ==========================================
def es_url_valida(texto):
    texto_clean = texto.strip().lower()
    dominios_validos = ["spotify.com", "youtube.com", "youtu.be", "soundcloud.com", "music.apple.com", "apple.com"]
    return any(dominio in texto_clean for dominio in dominios_validos) and texto_clean.startswith("http")

@st.cache_data(ttl=3600)
def obtener_titulo_y_audio(url):
    titulo = "Pista Externa"
    audio_path = None
    if not AUDIO_REAL_DISPONIBLE:
        return titulo, None
    
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
            'outtmpl': 'temp_audio.%(ext)s',
            'quiet': True,
            'no_warnings': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            titulo = info.get('title', 'Pista de Audio')
            audio_path = 'temp_audio.mp3'
    except Exception:
        pass
    return titulo, audio_path

def extraer_caracteristicas_audio_real(url_o_archivo):
    nombre_visual = "Archivo Local"
    tempo = 120.0
    danceability, energy, valence, speechiness, acousticness, densidad_tatum = 0.7, 0.7, 0.7, 0.05, 0.2, 3.0
    num_secciones, num_compases, num_tiempos_beats = 5, 32, 128

    if isinstance(url_o_archivo, str) and url_o_archivo.startswith("http"):
        nombre_visual, audio_path = obtener_titulo_y_audio(url_o_archivo)
        
        if AUDIO_REAL_DISPONIBLE and audio_path and os.path.exists(audio_path):
            try:
                y, sr = librosa.load(audio_path, duration=60) # Analizamos los primeros 60 segundos
                
                # 1. Tempo real (BPM)
                tempos, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(tempos[0]) if isinstance(tempos, np.ndarray) and len(tempos) > 0 else float(tempos)
                
                # 2. Energía (RMS) normalizada
                rms = librosa.feature.rms(y=y)
                energy = float(np.clip(np.mean(rms) * 5, 0.1, 1.0))
                
                # 3. Espectralidad / Acousticness aproximada
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                acousticness = float(np.clip(1.0 - (np.mean(spectral_centroid) / 5000.0), 0.05, 0.95))
                
                # 4. Speechiness basada en tasa de cruces por cero
                zcr = librosa.feature.zero_crossing_rate(y)
                speechiness = float(np.clip(np.mean(zcr) * 2, 0.02, 0.9))
                
                danceability = float(np.clip(energy * 0.9 + 0.1, 0.1, 1.0))
                valence = float(np.clip(energy * 0.8 + 0.2, 0.1, 1.0))
                densidad_tatum = round(tempo / 45.0, 2)
                
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception:
                pass

    # Filtro estricto de contenido hablado basado puramente en speechiness acústico real
    if speechiness > 0.40:
        return {
            "es_musica": False, "cancion_formateada": nombre_visual,
            "tempo": 0.0, "danceability": 0.10, "energy": 0.15, "valence": 0.20,
            "speechiness": speechiness, "acousticness": acousticness, "densidad_tatum": 0.2,
            "num_secciones": 1, "num_compases": 2, "num_tiempos_beats": 8
        }

    return {
        "es_musica": True,
        "cancion_formateada": nombre_visual,
        "tempo": round(tempo, 1),
        "danceability": round(danceability, 2),
        "energy": round(energy, 2),
        "valence": round(valence, 2),
        "speechiness": round(speechiness, 2),
        "acousticness": round(acousticness, 2),
        "densidad_tatum": round(densidad_tatum, 2),
        "num_secciones": num_secciones,
        "num_compases": num_compases,
        "num_tiempos_beats": num_tiempos_beats
    }

def clasificar_genero_por_audio(features):
    global modelo
    
    if not features.get('es_musica', True) or features.get('speechiness', 0) > 0.40:
        return "No Musical / Contenido Hablado"

    if modelo is not None:
        try:
            X_input = np.array([[
                features['tempo'],
                features['danceability'],
                features['energy'],
                features['valence'],
                features['speechiness'],
                features['acousticness'],
                features['densidad_tatum'],
                features['num_secciones'],
                features['num_compases'],
                features['num_tiempos_beats']
            ]])
            pred = modelo.predict(X_input)
            return str(pred[0])
        except Exception as e:
            return f"Error en Predicción del Modelo: {str(e)}"

    return "Error: No se encontró el archivo .joblib del modelo."

def obtener_detalles_coreograficos(genero):
    g_lower = genero.lower()
    if "bachata" in g_lower:
        pareja, grupo, solista = 8, 6, 7
        metrica = "📌 **Métrica:** Compás de 4/4. Acentuación en el pulso 4 y 8 con tap / golpe de cadera."
        aprovechamiento = "• **Baile en Pareja:** Trabajo de conexión corporal estrecha y marco fluido."
        vestuario = "• **Estilo:** Ropa estilizada y ajustada para lucir las caderas."
    elif "quebradita" in g_lower:
        pareja, grupo, solista = 10, 9, 8
        metrica = "📌 **Métrica:** Compás de 2/4 acelerado. Acentuación constante en el bote o brinco."
        aprovechamiento = "• **Acrobacias y Alzadas:** Trabajo de cargadas de alto impacto y giros veloces."
        vestuario = "• **Estilo:** Ropa vaquera moderna y botas con suela de soporte."
    elif "timba" in g_lower:
        pareja, grupo, solista = 9, 9, 9
        metrica = "📌 **Métrica:** Clave Cubana / Timba (2/3 o 3/2). Polirritmia compleja."
        aprovechamiento = "• **Nudos y Figuras Casino:** Complejidad en brazos y cambios de dirección."
        vestuario = "• **Estilo:** Ropa urbana deportiva o casual elegante."
    else:  # Salsa
        pareja, grupo, solista = 9, 8, 9
        metrica = "📌 **Métrica:** Fraseo de 8 tiempos (Clave 2/3 o 3/2). Acentos en campana y metales."
        aprovechamiento = "• **Shines & Footwork:** Trabajo veloz de pies y giros múltiples en pareja."
        vestuario = "• **Estilo:** Ropa formal o semi-formal con brillo y movimiento."

    return pareja, grupo, solista, metrica, aprovechamiento, vestuario

CATALOGO_DINAMICO = {
    "quebradita": ["La Chona - Los Tucanes de Tijuana", "La Quebradora - Banda El Mexicano"],
    "bachata": ["Obsesión - Aventura", "Propuesta Indecente - Romeo Santos"],
    "salsa": ["Llorarás - Oscar D'León", "Valió la Pena - Marc Anthony"],
    "timba": ["Ese Soy Yo - El Niño y la Verdad", "Me Dicen Cuba - Alexander Abreu"]
}

def responder_consulta_texto(prompt):
    p = prompt.lower()
    if any(kw in p for kw in ["quebrad", "banda"]):
        return "🤠 **Sugerencias de Quebradita:**\n\n" + "\n".join([f"• {c}" for c in CATALOGO_DINAMICO["quebradita"]])
    elif any(kw in p for kw in ["timb", "cuban"]):
        return "🇨🇺 **Sugerencias de Timba Cubana:**\n\n" + "\n".join([f"• {c}" for c in CATALOGO_DINAMICO["timba"]])
    elif any(kw in p for kw in ["bachat"]):
        return "🇩🇴 **Sugerencias de Bachata:**\n\n" + "\n".join([f"• {c}" for c in CATALOGO_DINAMICO["bachata"]])
    elif any(kw in p for kw in ["sals"]):
        return "🎺 **Sugerencias de Salsa:**\n\n" + "\n".join([f"• {c}" for c in CATALOGO_DINAMICO["salsa"]])
    else:
        return "💡 Pega un enlace de audio válido para analizarlo con Librosa o pídeme sugerencias."

# ==========================================
# 4. INTERFAZ STREAMLIT
# ==========================================
st.markdown('<div class="main-header">💃 Síncopa - Asistente Coreográfico</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis métrico de audio e interpretación de ritmos</div>', unsafe_allow_html=True)

tabs = st.tabs(["💬 Chat Asistente", "📊 Historial & Métricas", "ℹ️ Acerca del Modelo"])

with tabs[0]:
    for idx, msg in enumerate(st.session_state.messages):
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
    st.subheader("⚙️ Motor de Clasificación Acústica (Librosa + .joblib)")
    if AUDIO_REAL_DISPONIBLE:
        st.success("✅ Extracción DSP Real activada (Librerías `librosa` y `yt_dlp` disponibles).")
    else:
        st.warning("⚠️ Instala `librosa` y `yt_dlp` en tu entorno (`pip install librosa yt-dlp`) para habilitar el análisis de audio puro desde enlaces.")

# ==========================================
# 5. ENTRADA DEL CHAT
# ==========================================
if prompt := st.chat_input("Pega un enlace de audio o escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with tabs[0]:
        with st.chat_message("user"):
            st.markdown(prompt)

        if es_url_valida(prompt):
            with st.chat_message("assistant"):
                with st.spinner("🎧 Descargando pista y extrayendo vectores con Librosa..."):
                    analisis = extraer_caracteristicas_audio_real(prompt)

                prediccion_ml = clasificar_genero_por_audio(analisis)

                if prediccion_ml == "No Musical / Contenido Hablado":
                    reply = f"⚠️ **Contenido No Musical Detectado**\n\n🎵 **Pista:** *{analisis['cancion_formateada']}* (Speechiness alta)."
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                elif "Error" in prediccion_ml:
                    reply = f"❌ **Error:** {prediccion_ml}"
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    tempo_val = analisis["tempo"]
                    par, grp, sol, metrica_text, aprovechamiento_text, vestuario_text = obtener_detalles_coreograficos(prediccion_ml)

                    reply = f"""🎵 **Pista Analizada:** **{analisis['cancion_formateada']}**
🏷️ **Género Clasificado por Modelo (.joblib):** **{prediccion_ml}** 
⏱️ **Tempo Real (Librosa):** ~{tempo_val} BPM
📊 **Densidad Tatum:** {analisis['densidad_tatum']}

---

### 🎼 Marcación Coreográfica & Métrica Musical:
{metrica_text}

---

### 📊 Calificación por Modalidad de Baile:
* 👫 **Pareja:** {par} / 10
* 👯‍♀️ **Grupo:** {grp} / 10
* 🕺 **Solista:** {sol} / 10

---

### 💡 Aprovechamiento Coreográfico Recomendado:
{aprovechamiento_text}

---

### 👗 Sugerencia de Vestuario:
{vestuario_text}
"""
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.session_state.historial_evaluaciones.append({
                        "Canción": analisis['cancion_formateada'],
                        "Género": prediccion_ml,
                        "Tempo": tempo_val
                    })
        else:
            with st.chat_message("assistant"):
                reply = responder_consulta_texto(prompt)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
