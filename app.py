import os
import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import streamlit.components.v1 as components

import sincopa_core as sc

# ==========================================
# 1. CONFIGURACIÓN Y ESTILOS
# ==========================================
st.set_page_config(page_title="Síncopa - Asistente Coreográfico", page_icon="💃", layout="wide")
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #E63946; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #457B9D; text-align: center; margin-bottom: 1.5rem; }
    .stChatMessage { border-radius: 12px; }
    .stChatInput { position: fixed; bottom: 0; left: 0; right: 0; padding: 1rem; background: rgba(255, 255, 255, 0.95); z-index: 100; }
    .stChatInput textarea { height: 90px !important; font-size: 1rem !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARGA DEL DATASET REAL Y DEL MODELO — ÚNICA FUENTE DE VERDAD
#    Si existe modelo_sincopa_rf.joblib (producido por entrenar_modelo.py
#    sobre dataset_bachata_salsa_quebradita.csv) lo carga tal cual. Si no
#    existe, entrena en memoria con EXACTAMENTE la misma función
#    (sc.entrenar_random_forest) sobre el mismo dataset real.
# ==========================================
RUTA_MODELO = os.path.join(os.path.dirname(__file__), "modelo_sincopa_rf.joblib")


@st.cache_resource
def cargar_todo():
    df = sc.cargar_dataset()
    if os.path.exists(RUTA_MODELO):
        modelo, origen = joblib.load(RUTA_MODELO), "archivo (modelo_sincopa_rf.joblib)"
    else:
        modelo, _ = sc.entrenar_random_forest(df)
        origen = "entrenado en memoria (no se encontró el .joblib)"
    rangos_genero = sc.calcular_rangos_desde_dataset(df)
    rango_ambiguo = sc.calcular_rango_ambiguo(rangos_genero)
    palabras_genero = sc.construir_palabras_genero(df)
    return df, modelo, origen, rangos_genero, rango_ambiguo, palabras_genero


df_canciones, modelo, origen_modelo, RANGOS_GENERO, RANGO_AMBIGUO, PALABRAS_GENERO = cargar_todo()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 **¡Hola! Soy Síncopa**, tu asistente coreográfico.\n\n"
                   "🎯 **Puedo ayudarte a evaluar pistas de:** Bachata, Salsa (incluye Timba cubana) y Quebradita.\n\n"
                   "⚠️ **Cómo trabajo:**\n"
                   "1. Si tu canción está en mi base de datos real (482 pistas), uso sus características exactas.\n"
                   "2. Si no la reconozco, estimo un perfil rítmico plausible dentro del rango real observado "
                   "para el género que detecte por palabras clave — no analizo audio real.\n"
                   "3. Distingo si un video **es una canción**, si **contiene música pero no es una pista "
                   "dedicada** (tráiler, gameplay, resumen deportivo, etc.) o si **no tiene música** (podcast, "
                   "tutorial, entrevista)."
    }]
if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []
if "sugerencias_usadas" not in st.session_state:
    st.session_state.sugerencias_usadas = {g: [] for g in list(sc.GENEROS) + ["Timba"]}
if "entrenamientos_usados" not in st.session_state:
    st.session_state.entrenamientos_usados = {g: [] for g in list(sc.GENEROS) + ["Timba"]}
if "ultimo_genero_evaluado" not in st.session_state:
    st.session_state.ultimo_genero_evaluado = None

# ==========================================
# 3. EXTRACCIÓN DE TÍTULO DESDE ENLACES
# ==========================================
@st.cache_data(ttl=3600)
def extraer_titulo_link(url):
    try:
        if "youtube.com" in url or "youtu.be" in url:
            res = requests.get(f"https://www.youtube.com/oembed?url={url}&format=json", timeout=3)
            if res.status_code == 200:
                return res.json().get("title", url)
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            if soup.title and soup.title.string:
                titulo = soup.title.string.strip()
                titulo = titulo.replace(" - song and lyrics by", "").replace(" | Spotify", "").strip()
                return titulo
    except Exception:
        pass
    return url


# ==========================================
# 4. ANÁLISIS DE ENTRADA
#    Etapa A -> sc.clasificar_tipo_contenido: ¿es canción, tiene música
#               sin ser pista dedicada, o no es musical?
#    Etapa B -> si es canción: ¿coincide con el dataset real? Si sí, usa
#               SUS features reales. Si no, estima dentro del rango real
#               del género detectado por palabra clave.
# ==========================================
def analizar_entrada(entrada):
    if entrada.strip().startswith("http"):
        titulo = extraer_titulo_link(entrada)
    else:
        titulo = entrada

    tipo = sc.clasificar_tipo_contenido(titulo, PALABRAS_GENERO, df=df_canciones)

    if tipo == "no_musical":
        return {"tipo": "no_musical", "titulo": titulo}
    if tipo == "musica_no_dedicada":
        return {"tipo": "musica_no_dedicada", "titulo": titulo}
    if tipo == "no_parece_cancion":
        return {"tipo": "no_parece_cancion", "titulo": titulo}

    # ¿Coincide con una canción real del dataset?
    fila_real = sc.buscar_en_dataset(titulo, df_canciones)
    if fila_real is not None:
        features = {f: fila_real[f] for f in sc.FEATURES}
        fuente = "real"
        genero_kw = fila_real["genero_etiqueta"]
    else:
        genero_kw = sc.detectar_genero_por_palabra_clave(titulo.lower(), PALABRAS_GENERO)
        features = sc.estimar_features(genero_kw, RANGOS_GENERO, RANGO_AMBIGUO)
        fuente = "estimado"

    X_input = np.array([[features[f] for f in sc.FEATURES]])
    probabilidades = modelo.predict_proba(X_input)[0]
    clases = modelo.classes_
    max_prob = np.max(probabilidades)
    prediccion = clases[np.argmax(probabilidades)]

    umbral = 0.55 if (fuente == "real" or genero_kw) else 0.65
    if max_prob < umbral:
        return {"tipo": "sin_certeza", "titulo": titulo}

    # Variante Timba: solo etiqueta de "sabor" dentro de Salsa, no cambia el modelo.
    es_timba = prediccion == "Salsa" and sc.es_variante_timba(titulo)
    genero_catalogo = "Timba" if es_timba else prediccion

    return {
        "tipo": "cancion", "titulo": titulo, "fuente": fuente,
        "tempo": round(float(features["tempo"]), 1),
        "genero_modelo": prediccion, "genero_catalogo": genero_catalogo,
        "certidumbre": round(max_prob * 100, 1),
        "num_secciones": features.get("num_secciones"),
    }


# ==========================================
# 5. UI
# ==========================================
st.markdown('<div class="main-header">💃 Síncopa - Asistente Coreográfico</div>', unsafe_allow_html=True)
tabs = st.tabs(["💬 Chat Asistente", "📊 Historial y Descarga", "⚙️ Modelo"])

with tabs[0]:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Pega el enlace, pide sugerencias, vestuario o entrenamiento..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            prompt_lower = prompt.lower()

            generos_catalogo = list(sc.GENEROS) + ["Timba"]
            pide_sugerencia = any(k in prompt_lower for k in
                                   ["sugerencia", "dame", "recomienda", "canción", "canciones", "lista"])
            pide_entrenamiento = any(k in prompt_lower for k in
                                      ["entrenamiento", "ejercicio", "entrena", "rutina física", "workout"])
            genero_mencionado = next((g for g in generos_catalogo if g.lower() in prompt_lower), None)

            if genero_mencionado and pide_sugerencia:
                item = sc.elegir_sin_repetir(genero_mencionado, sc.CATALOGO_SUGERENCIAS,
                                              st.session_state.sugerencias_usadas)
                reply = f"🎶 **Sugerencia dinámica para {genero_mencionado}:**\nTe recomiendo probar con: **{item}**."

            elif pide_entrenamiento:
                genero_ent = genero_mencionado or st.session_state.ultimo_genero_evaluado
                if not genero_ent:
                    reply = ("⚠️ ¿Para qué género quieres el entrenamiento? Evalúa una canción primero o "
                              "dime el género directamente (Bachata, Salsa, Timba o Quebradita).")
                else:
                    item = sc.elegir_sin_repetir(genero_ent, sc.CATALOGO_ENTRENAMIENTOS,
                                                  st.session_state.entrenamientos_usados)
                    nombre, ejercicio, motivo = item
                    reply = (f"⚡ **Entrenamiento de {nombre} para {genero_ent}:**\n{ejercicio}\n\n"
                              f"> *¿Por qué?* {motivo}")

            elif "vestuario" in prompt_lower:
                gen_vest = genero_mencionado or st.session_state.ultimo_genero_evaluado
                if gen_vest:
                    prefijo = "" if genero_mencionado else f"ℹ️ *Basado en la última canción evaluada ({gen_vest}):*\n\n"
                    reply = prefijo + sc.CATALOGO_VESTUARIO.get(gen_vest, "Vestuario general de competencia.")
                else:
                    reply = ("⚠️ Aún no has evaluado ninguna canción en esta sesión. Evalúa una pista primero "
                              "o dime el género para darte opciones de vestuario de competencia.")

            else:
                resultado = analizar_entrada(prompt)
                tipo = resultado["tipo"]

                if tipo == "no_musical":
                    reply = (f"🗣️ **Contenido no musical detectado**\n📄 *{resultado['titulo']}*\n\n"
                              "Esto parece ser voz hablada (podcast, entrevista, tutorial, etc.), sin estructura "
                              "métrica musical. No hay nada que evaluar coreográficamente.")

                elif tipo == "musica_no_dedicada":
                    reply = (f"🎬 **Video con música, pero no es una pista dedicada para bailar**\n"
                              f"📄 *{resultado['titulo']}*\n\n"
                              "Detecté que este contenido probablemente tiene música de fondo (tráiler, "
                              "gameplay, resumen deportivo, anuncio, etc.), pero no es una canción pensada "
                              "para evaluar coreográficamente. Pégame el enlace o título de la canción "
                              "específica si quieres que la analice.")

                elif tipo == "no_parece_cancion":
                    reply = (f"🤔 **No reconozco esto como una canción**\n📄 *{resultado['titulo']}*\n\n"
                              "No encontré ni una palabra clave de género (Bachata, Salsa, Quebradita) ni un "
                              "formato de título tipo *Artista - Canción*, ni coincide con mi base de datos. "
                              "Si es una canción real, escríbela como `Artista - Título` o pégame el enlace.")

                elif tipo == "sin_certeza":
                    reply = (f"⚠️ **Certeza insuficiente**\n📄 *{resultado['titulo']}*\n\n"
                              "El modelo no alcanzó suficiente confianza para asignar un género de baile "
                              "soportado con certeza. Prueba con el enlace directo o con el nombre completo "
                              "del artista.")

                else:  # "cancion"
                    genero_modelo = resultado["genero_modelo"]
                    genero_catalogo = resultado["genero_catalogo"]
                    certidumbre = resultado["certidumbre"]
                    st.session_state.ultimo_genero_evaluado = genero_catalogo
                    par, grp, sol, metrica, aprovechamiento = sc.DETALLES_COREOGRAFICOS[genero_catalogo]

                    nota_fuente = ("📌 *Encontré esta pista en mi base de datos real — usé sus características "
                                   "medidas, no una estimación.*" if resultado["fuente"] == "real" else
                                   "📌 *No encontré esta pista en mi base de datos: el perfil rítmico es una "
                                   "estimación dentro del rango real observado para este género.*")

                    etiqueta_genero = genero_catalogo if genero_catalogo != genero_modelo else genero_modelo
                    nota_subgenero = (f"\n*(Clasificada por el modelo como Salsa; identifiqué el subgénero "
                                       f"Timba por palabras clave del título/artista.)*" if genero_catalogo == "Timba" else "")

                    reply = f"""🎵 **Pista / Enlace:** **{resultado['titulo']}**
🏷️ **Clasificación:** **{etiqueta_genero}**{nota_subgenero}
🎯 **Certidumbre del Pronóstico:** **{certidumbre}%**
⏱️ **Tempo Estimado:** ~{resultado['tempo']} BPM

{nota_fuente}

---
### 🎼 Marcación Coreográfica:
{metrica}

### 📊 Calificación:
* 👫 Pareja: {par}/10 | 👯‍♀️ Grupo: {grp}/10 | 🕺 Solista: {sol}/10

### 💡 Aprovechamiento:
{aprovechamiento}

---
💬 Pide "entrenamiento" o "vestuario" para {genero_catalogo} cuando quieras.
"""
                    st.session_state.historial_evaluaciones.append({
                        "Canción": resultado["titulo"], "Género": genero_catalogo,
                        "Certidumbre (%)": certidumbre, "Tempo": resultado["tempo"],
                        "Fuente": resultado["fuente"],
                    })

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    components.html(
        """
        <script>
            const doc = window.parent.document;
            setTimeout(() => { doc.window.scrollTo(0, doc.body.scrollHeight); }, 100);
        </script>
        """,
        height=0,
    )

with tabs[1]:
    st.subheader("📊 Historial de Evaluaciones y Exportación")
    if st.session_state.historial_evaluaciones:
        df_historial = pd.DataFrame(st.session_state.historial_evaluaciones)
        st.dataframe(df_historial, use_container_width=True)
        csv_data = df_historial.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar Evaluaciones en CSV", data=csv_data,
                            file_name="historial_evaluaciones_sincopa.csv", mime="text/csv")
    else:
        st.info("Aún no hay canciones evaluadas en el historial.")

with tabs[2]:
    st.json({
        "Algoritmo": "RandomForestClassifier",
        "Origen del modelo cargado": origen_modelo,
        "Dataset": "dataset_bachata_salsa_quebradita.csv (482 canciones reales)",
        "Features": sc.FEATURES,
        "Clases del modelo": list(modelo.classes_),
        "Nota": "Timba se identifica por palabras clave como variante de Salsa para elegir catálogo de "
                "sugerencias/entrenamiento/vestuario, pero el modelo solo predice las 3 clases presentes "
                "en el dataset real.",
    })
