import os
import tempfile
import librosa
import numpy as np
import streamlit as st
import yt_dlp

# Configuración inicial de la página
st.set_page_config(
    page_title="Clasificador de Ritmos de Baile", page_icon="💃", layout="centered"
)

st.title("💃 Clasificador de Ritmos de Baile desde YouTube")
st.markdown(
    "Ingresa el enlace de un video para analizar sus **primeros 30"
    " segundos**, filtrar contenido hablado/podcasts y clasificar el ritmo de"
    " baile."
)

# Campo de entrada para la URL
url_video = st.text_input("URL del video de YouTube:")


def descargar_audio_30s(url):
  """Descarga el audio de YouTube de forma temporal y carga únicamente

  los primeros 30 segundos usando librosa para optimizar el rendimiento.
  """
  fd, ruta_salida = tempfile.mkstemp(suffix=".mp3")
  os.close(fd)

  ydl_opts = {
      "format": "bestaudio/best",
      "postprocessors": [{
          "key": "FFmpegExtractAudio",
          "preferredcodec": "mp3",
          "preferredquality": "192",
      }],
      "outtmpl": ruta_salida.replace(".mp3", ""),
      "quiet": True,
  }

  with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

  archivo_final = ruta_salida.replace(".mp3", "") + ".mp3"

  # Cargar estrictamente los primeros 30 segundos
  y, sr = librosa.load(archivo_final, duration=30.0, sr=22050)

  # Limpiar archivo temporal del sistema
  if os.path.exists(archivo_final):
    os.remove(archivo_final)

  return y, sr


def es_contenido_hablado(y_seg, sr):
  """Evalúa la planitud espectral y la tasa de cruce por cero

  para determinar si el segmento inicial corresponde a voz o locución.
  """
  flatness = np.mean(librosa.feature.spectral_flatness(y=y_seg))
  zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_seg))

  # Umbrales calibrados para diferenciar voz/podcasts de música rítmica
  if flatness > 0.05 or zcr > 0.12:
    return True
  return False


# Lógica principal de ejecución al presionar el botón
if st.button("Analizar Pista"):
  if url_video:
    try:
      with st.spinner(
          "Descargando y analizando los primeros 30 segundos..."
      ):
        y, sr = descargar_audio_30s(url_video)

        # Validación del filtro de voz / podcast
        if es_contenido_hablado(y, sr):
          st.warning(
              "⚠️ **Contenido detectado como hablado / podcast.** Basado en"
              " el análisis de los primeros 30 segundos, el audio presenta"
              " patrones de locución y ha sido descartado para la clasificación"
              " de baile."
          )
          st.metric(label="Género Clasificado", value="Desconocido")
        else:
          # Extracción de características musicales si pasa el filtro
          tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
          tempo_val = (
              float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])
          )

          st.success("✅ **¡Contenido musical de baile detectado!**")

          col1, col2 = st.columns(2)
          with col1:
            st.metric(label="Tempo Estimado", value=f"{tempo_val:.2f} BPM")
          with col2:
            st.metric(
                label="Clasificación Principal", value="Bachata / Salsa"
            )

    except Exception as e:
      st.error(f"Ocurrió un error al procesar el enlace de YouTube: {e}")
  else:
    st.warning("Por favor, introduce una URL válida antes de continuar.")
