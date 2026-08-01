import streamlit as st
import yt_dlp
import librosa
import numpy as np
import pandas as pd
import re
import os
import tempfile
import hashlib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Síncopa - Análisis Coreográfico y Musical",
    page_icon="💃",
    layout="wide"
)

if "historico" not in st.session_state:
    st.session_state.historico = []

st.title("💃 Síncopa: Análisis Musical y Asistente Coreográfico")
st.markdown("Plataforma de clasificación de ritmos tropicales (**Salsa, Bachata, Quebradita**) basada en métricas de audio y machine learning.")
st.markdown("---")

# ==========================================
# 2. MODELO ENTRENADO Y BASE DE DATOS DE SUGERENCIAS
# ==========================================
@st.cache_resource
def cargar_modelo():
    # Métricas de entrenamiento: [BPM, Centroide Espectral (timbre), RMS (energía)]
    X_train = np.array([
        # Bachata (~100-130 BPM)
        [105, 1.2, 0.4], [115, 1.1, 0.5], [125, 1.3, 0.45],
        # Salsa (~150-200 BPM)
        [160, 2.5, 0.8], [175, 2.8, 0.85], [190, 2.4, 0.9],
        # Quebradita (>210 BPM)
        [225, 3.1, 0.95], [235, 3.5, 0.98], [245, 3.2, 0.92],
        # Voz Hablada / Podcasts / Ruidos
        [80, 0.4, 0.2], [140, 0.5, 0.3], [200, 0.3, 0.2]
    ])
    # Clases: 0: Bachata, 1: Salsa, 2: Quebradita, 3: Fuera de Alcance / Habla
    y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

modelo_rf = cargar_modelo()

DATOS_RECOMENDACIONES = {
    "Salsa": {
        "subestilos": ["Salsa On1 / Lineal", "Salsa On2 / Mambo", "Salsa Caleña", "Rueda de Casino"],
        "vestuario": "Trajes ajustados con flecos o lentejuelas que destaquen el movimiento de cadera; zapatos con suela de gamuza flexible.",
        "pistas_similares": [
            "Marc Anthony - Tu Amor Me Hace Bien",
            "Grupo Niche - Cali Pachanguero",
            "Héctor Lavoe - Periódico de Ayer"
        ]
    },
    "Bachata": {
        "subestilos": ["Bachata Sensual", "Bachata Dominicana / Tradicional", "Bachata Urbana"],
        "vestuario": "Líneas elegantes y fluidas; telas elásticas para movimiento corporal continuo y acoplamiento en pareja.",
        "pistas_similares": [
            "Romeo Santos - Propuesta Indecente",
            "Aventura - Dile al Amor",
            "Prince Royce - Darte un Beso"
        ]
    },
    "Quebradita": {
        "subestilos": ["Quebradita Tradicional / Baile de Caballito", "Quebradita Acrobática"],
        "vestuario": "Atuendo vaquero o norteño: sombrero, botas ligeras reforzadas para amortiguación, hebillas decorativas y jeans cómodos.",
        "pistas_similares": [
            "Banda Machos - La Secretaria",
            "Banda Arkángel R-15 - El Bailador",
            "Mi Banda El Mexicano - No Bailes de Caballito"
        ]
    }
}

# ==========================================
# 3. EXTRACCIÓN DE METADATOS Y ANÁLISIS DE AUDIO
# ==========================================
def obtener_metadatos_link(url):
    ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False) or {}
            titulo = info.get('title', 'Pista Desconocida')
            duracion = info.get('duration', 0) or 0
            categorias = info.get('categories') or []  # Protección contra NoneType
            return True, titulo, duracion, categorias
        except Exception:
            return False, "Pista / Enlace", 0, []

def validar_alcance_pista(query, titulo, duracion, categorias):
    cadena_eval = (query + " " + titulo).lower()

    # 1. Filtro de Podcasts y Voz Hablada
    tokens_no_musica = [
        "podcast", "episodio", "ep.", "episode", "entrevista", "interview", 
        "vlog", "hablado", "conferencia", "noticias", "discurso", "talking", 
        "chat", "conversacion", "review", "tutorial", "explicacion", "resumen"
    ]
    if any(t in cadena_eval for t in tokens_no_musica) or ("Music" not in categorias and duracion > 600):
        return False, "podcast"

    # 2. Filtro de Palabras de Ritmos Soportados
    tokens_quebradita = ["quebradita", "quebradora", "banda", "zapateado", "brinco", "caballito", "arkangel", "machos"]
    tokens_bachata = ["bachata", "sensual", "romeo", "aventura", "prince royce", "darte", "obsesion"]
    tokens_salsa = ["salsa", "mambo", "guaguanco", "timba", "marc anthony", "lavoe", "niche"]

    tiene_ritmo = any(w in cadena_eval for w in (tokens_quebradita + tokens_bachata + tokens_salsa))

    if not tiene_ritmo and "Music" not in categorias and len(categorias) > 0:
        return False, "fuera_de_alcance"

    return True, "ok"

def descargar_y_analizar_audio(url):
    temp_dir = tempfile.mkdtemp()
    out_path = os.path.join(temp_dir, 'sample.mp3')
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_path,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    y, sr = librosa.load(out_path, duration=30)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]
        
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    
    if os.path.exists(out_path):
        os.remove(out_path)
        
    return float(tempo), float(spec_cent / 1000.0), float(rms * 10)

def generar_pdf_ficha(titulo, genero, bpm, confianza, detalles):
    buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(buffer.name, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#1DB954"))
    story.append(Paragraph("Síncopa - Ficha Técnica de Análisis Musical", title_style))
    story.append(Spacer(1, 12))

    data = [
        ["Parámetro", "Detalle"],
        ["Pista / Título:", titulo],
        ["Ritmo / Género:", genero],
        ["Tempo Estimado:", f"{bpm:.1f} BPM"],
        ["Certeza del Análisis:", f"{confianza:.1f}%"],
        ["Fecha de Análisis:", datetime.now().strftime("%Y-%m-%d %H:%M")]
    ]
    t = Table(data, colWidths=[150, 350])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#333333")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 18))

    story.append(Paragraph("<b>Estilos y Sugerencias de Baile:</b>", styles['Heading2']))
    for sub in detalles['subestilos']:
        story.append(Paragraph(f"• {sub}", styles['BodyText']))
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Recomendación de Vestuario:</b>", styles['Heading2']))
    story.append(Paragraph(detalles['vestuario'], styles['BodyText']))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Pistas Similares Sugeridas:</b>", styles['Heading2']))
    for p in detalles['pistas_similares']:
        story.append(Paragraph(f"• {p}", styles['BodyText']))

    doc.build(story)
    return buffer.name

# ==========================================
# 4. INTERFAZ Y NAVEGACIÓN
# ==========================================
tab_analisis, tab_dashboard = st.tabs(["🎧 Análisis de Pista", "📊 Dashboard Histórico"])

# ------------------------------------------
# TAB 1: ANÁLISIS, ALERTAS Y SUGERENCIAS
# ------------------------------------------
with tab_analisis:
    query = st.text_input("Ingresa la URL del audio/video (YouTube, SoundCloud, Spotify, etc.):", "")

    if st.button("Analizar Pista"):
        # WARNING 1: Entrada vacía
        if not query.strip():
            st.warning("⚠️ **Por favor, ingresa una URL antes de continuar.**")
        else:
            # WARNING 2: No es un enlace
            match = re.search(r'https?://[^\s]+', query)
            if not match:
                st.error("⚠️ **Formato no válido:** Por favor, ingresa únicamente un enlace (link) válido de *Spotify, YouTube, SoundCloud o Apple Music*.")
            else:
                url_detectada = match.group(0)
                
                with st.spinner("🔍 Validando metadatos del enlace..."):
                    exito_meta, titulo, duracion, categorias = obtener_metadatos_link(url_detectada)
                
                es_valido, razon = validar_alcance_pista(query, titulo, duracion, categorias)
                
                # WARNING 3: Contenido no musical / Podcast
                if not es_valido and razon == "podcast":
                    st.error("🎙️ **Contenido No Musical Detectado:** El enlace ingresado parece ser un podcast, entrevista u otro contenido hablado. Síncopa solo analiza pistas musicales de Salsa, Bachata y Quebradita.")
                
                # WARNING 4: Pista fuera de alcance
                elif not es_valido and razon == "fuera_de_alcance":
                    st.warning("⚠️ **Pista fuera de alcance:** El enlace proporcionado no parece pertenecer a los géneros soportados (**Salsa, Bachata o Quebradita**).")
                
                else:
                    st.success(f"📌 **Título encontrado:** {titulo}")
                    
                    with st.spinner("🎧 Procesando métricas de audio (BPM, energía y timbre)..."):
                        try:
                            # Cálculo real de métricas
                            bpm, spec_cent, rms = descargar_y_analizar_audio(url_detectada)
                            
                            # Evaluación con las métricas del modelo entrenado
                            vector_features = np.array([[bpm, spec_cent, rms]])
                            probabilidades = modelo_rf.predict_proba(vector_features)[0]
                            max_prob = np.max(probabilidades)
                            clase_predicha = np.argmax(probabilidades)
                            
                            mapa_generos = {0: "Bachata", 1: "Salsa", 2: "Quebradita", 3: "Otro / Hablado"}
                            genero_detectado = mapa_generos.get(clase_predicha, "Desconocido")
                            
                            # WARNING 5: Modelo rechaza por baja confianza o clase no musical
                            if clase_predicha == 3 or max_prob < 0.60:
                                st.warning("⚠️ **Pista fuera de alcance:** Las métricas de audio extraídas no se ajustan al perfil de **Salsa, Bachata o Quebradita**.")
                            else:
                                detalles = DATOS_RECOMENDACIONES[genero_detectado]
                                
                                # Guardar en Histórico
                                registro = {
                                    "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                    "Título": titulo,
                                    "Género": genero_detectado,
                                    "BPM": round(bpm, 1),
                                    "Confianza": round(max_prob * 100, 1)
                                }
                                st.session_state.historico.append(registro)
                                
                                # Mostrar Resultados y Sugerencias de Baile / Vestuario
                                st.markdown("---")
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    st.subheader("📊 Métricas de Clasificación")
                                    st.markdown(f"* **Género Predicho:** `{genero_detectado}`")
                                    st.markdown(f"* **Tempo Estimado (BPM):** `{bpm:.1f} BPM`")
                                    st.markdown(f"* **Certeza del Modelo:** `{max_prob * 100:.1f}%`")
                                    st.progress(int(max_prob * 100))
                                    
                                    st.subheader("👗 Vestuario Recomendado")
                                    st.info(detalles['vestuario'])

                                with col2:
                                    st.subheader("🕺 Subestilos Recomendados")
                                    for sub in detalles['subestilos']:
                                        st.write(f"• {sub}")
                                        
                                    st.subheader("🎵 Pistas Similares Sugeridas")
                                    for p in detalles['pistas_similares']:
                                        st.write(f"• {p}")

                                # Ficha Técnica PDF
                                st.markdown("---")
                                pdf_path = generar_pdf_ficha(titulo, genero_detectado, bpm, max_prob * 100, detalles)
                                with open(pdf_path, "rb") as pdf_file:
                                    st.download_button(
                                        label="📄 Descargar Ficha Técnica en PDF",
                                        data=pdf_file,
                                        file_name=f"Ficha_Tecnica_{genero_detectado}.pdf",
                                        mime="application/pdf"
                                    )

                        except Exception as e:
                            st.warning("⚠️ **Pista fuera de alcance:** No se pudo extraer la onda de audio para calcular las métricas rítmicas.")

# ------------------------------------------
# TAB 2: DASHBOARD HISTÓRICO
# ------------------------------------------
with tab_dashboard:
    st.subheader("📈 Dashboard de Consultas e Histórico")
    
    if not st.session_state.historico:
        st.info("Aún no has analizado pistas en esta sesión. Realiza un análisis en la pestaña anterior para poblar el histórico.")
    else:
        df_hist = pd.DataFrame(st.session_state.historico)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total de Analizadas", len(df_hist))
        m2.metric("Ritmo Predominante", df_hist['Género'].mode()[0] if not df_hist.empty else "N/A")
        m3.metric("BPM Promedio", f"{df_hist['BPM'].mean():.1f}")
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("**Distribución por Género:**")
            st.bar_chart(df_hist['Género'].value_counts())
            
        with col_chart2:
            st.markdown("**Variación de Tempo (BPM) en el Tiempo:**")
            st.line_chart(df_hist.set_index('Fecha')['BPM'])
            
        st.markdown("---")
        st.markdown("**Tabla de Registros:**")
        st.dataframe(df_hist, use_container_width=True)
