import streamlit as st
import yt_dlp
import librosa
import numpy as np
import re
import os
import tempfile
import hashlib
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Síncopa - Clasificador de Ritmos Tropicales",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Síncopa: Análisis y Clasificación de Audio")
st.subheader("Salsa | Bachata | Quebradita")
st.markdown("---")

# ==========================================
# 2. ENTRENAMIENTO DE MODELO DE RESPALDO (MOCK ML)
# ==========================================
# Se entrena un RandomForest dummy para predecir si el tempo/audio
# coincide con las características de los tres géneros o si debe descartarse.
@st.cache_resource
def cargar_modelo():
    # Características ficticias: [BPM, variabilidad_espectral, energia]
    # Clases: 0: Bachata, 1: Salsa, 2: Quebradita, 3: Habla/Otro
    X_train = np.array([
        # Bachata (~100-130 BPM)
        [105, 1.2, 0.4], [115, 1.1, 0.5], [125, 1.3, 0.45],
        # Salsa (~150-200 BPM)
        [160, 2.5, 0.8], [175, 2.8, 0.85], [190, 2.4, 0.9],
        # Quebradita (>210 BPM)
        [225, 3.1, 0.95], [235, 3.5, 0.98], [245, 3.2, 0.92],
        # Voz / Hablado / Podcasts (BPM irregular / fuera de rango)
        [80, 0.4, 0.2], [140, 0.5, 0.3], [200, 0.3, 0.2]
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

modelo_rf = cargar_modelo()

# ==========================================
# 3. EXTRACCIÓN Y VALIDACIÓN DE METADATOS (yt-dlp)
# ==========================================
def analizar_metadatos_link(url):
    """
    Usa yt-dlp para inspeccionar categorías, duración y títulos antes de descargar.
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            
            titulo = info.get('title', 'Desconocido')
            duracion = info.get('duration', 0)  # en segundos
            categorias = info.get('categories', [])
            
            # Filtro 1: Si la plataforma expresamente indica que no es música y dura > 10 min
            if "Music" not in categorias and duracion > 600:
                return False, titulo, "Contenido extenso de tipo podcast/hablado (Categoría no musical)."

            # Filtro 2: Duración excesiva para un track tropical de competencia/práctica
            if duracion > 900:  # Mayor a 15 minutos
                return False, titulo, "El audio excede la duración típica de una canción (máximo 15 mins)."

            return True, titulo, "Ok"
        except Exception as e:
            return False, "Enlace no válido", f"Error al procesar el enlace: {str(e)}"

# ==========================================
# 4. DESCARGA Y ANÁLISIS DE AUDIO CON LIBROSA
# ==========================================
def descargar_muestra_audio(url):
    """
    Descarga únicamente un fragmento pequeño de audio (primeros 30 seg)
    para análisis rítmico rápido.
    """
    temp_dir = tempfile.mkdtemp()
    out_path = os.path.join(temp_dir, 'sample.mp3')
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    return out_path

def analizar_audio_caracteristicas(ruta_audio):
    """
    Procesa el audio descargado para calcular tempo (BPM) y energía espectral.
    """
    # Cargar los primeros 30 segundos del audio
    y, sr = librosa.load(ruta_audio, duration=30)
    
    # Estimación de Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]
        
    # Calcular Centroide Espectral (variabilidad timbríca) y RMS (Energía)
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    
    # Limpiar archivo temporal
    if os.path.exists(ruta_audio):
        os.remove(ruta_audio)
        
    return float(tempo), float(spec_cent / 1000.0), float(rms * 10)

# ==========================================
# 5. INTERFAZ DE USUARIO STREAMLIT
# ==========================================
query = st.text_input("Ingresa la URL del audio/video (YouTube, SoundCloud, Spotify, etc.):", "")

if st.button("Analizar Pista"):
    if not query.strip():
        st.warning("⚠️ Por favor, ingresa una URL antes de continuar.")
    else:
        # Validación inicial de formato de URL
        match = re.search(r'https?://[^\s]+', query)
        if not match:
            st.error("⚠️ **Formato no válido:** Por favor, ingresa únicamente un enlace (link) válido de audio o video.")
        else:
            url_detectada = match.group(0)
            
            with st.spinner("🔍 Analizando metadatos del enlace..."):
                es_valido_meta, titulo, razon_meta = analizar_metadatos_link(url_detectada)
            
            if not es_valido_meta:
                st.error(f"🎙️ **Pista no válida:** {razon_meta}")
            else:
                st.success(f"📌 **Título encontrado:** {titulo}")
                
                with st.spinner("🎧 Extrayendo y analizando características de la señal de audio..."):
                    try:
                        ruta_temp = descargar_muestra_audio(url_detectada)
                        bpm, spec_cent, rms = analizar_audio_caracteristicas(ruta_temp)
                        
                        # Evaluar en el modelo Random Forest
                        vector_features = np.array([[bpm, spec_cent, rms]])
                        probabilidades = modelo_rf.predict_proba(vector_features)[0]
                        max_prob = np.max(probabilidades)
                        clase_predicha = np.argmax(probabilidades)
                        
                        # Nombres de géneros mapeados
                        mapa_generos = {0: "Bachata", 1: "Salsa", 2: "Quebradita", 3: "Otro / Hablado"}
                        genero_detectado = mapa_generos.get(clase_predicha, "Desconocido")
                        
                        # --- REGLA DE UMBRAL DE CONFIANZA ---
                        # Si la clase es "Otro" o la certeza del modelo es menor al 60%
                        if clase_predicha == 3 or max_prob < 0.60:
                            st.warning(
                                "⚠️ **Pista fuera de alcance:** El audio analizado no presenta una estructura rítmica clara de **Salsa, Bachata o Quebradita** "
                                "(posiblemente se trate de voz hablada, podcast o un género musical no soportado)."
                            )
                        else:
                            st.subheader("📊 Resultado del Análisis")
                            st.markdown(f"* **Género Detectado:** `{genero_detectado}`")
                            st.markdown(f"* **Tempo Estimado:** `{bpm:.1f} BPM`")
                            st.markdown(f"* **Confianza del Modelo:** `{max_prob * 100:.1f}%`")
                            
                            # Muestra visual de desglose
                            st.progress(int(max_prob * 100))
                            
                    except Exception as e:
                        # Fallback seguro en caso de error en descarga o procesamiento
                        st.warning(
                            "⚠️ **No se pudo procesar la onda de audio directamente.** "
                            "Asegúrate de que la fuente permita reproducción pública de audio."
                        )
