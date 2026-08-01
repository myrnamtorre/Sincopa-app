import streamlit as st
import numpy as np
import time
import joblib
import os
import pandas as pd
import hashlib
import requests
from bs4 import BeautifulSoup
import re
import random

st.set_page_config(
    page_title="Síncopa • Asistente Coreográfico IA",
    page_icon="💃",
    layout="wide"
)

st.title("💃 Síncopa: Asistente Coreográfico IA")
st.caption("🤖 Agente Conversacional para Análisis Coreográfico, Ritmo y Entrenamiento")
st.markdown("---")

# 1. CARGA DEL MODELO ML
ruta_modelo = 'modelo_sincopa_rf.joblib'
modelo = None
if os.path.exists(ruta_modelo):
    try:
        modelo = joblib.load(ruta_modelo)
        st.sidebar.success("🤖 Backend ML: Random Forest Activo")
    except Exception as e:
        st.sidebar.error(f"Error al cargar el modelo: {e}")

# 2. BANCO DE DATOS BASE PARA GENERACIÓN DINÁMICA
ARTISTAS_Y_ESTILOS = {
    "Quebradita": {
        "artistas": ["Banda Machos", "Los Tucanes de Tijuana", "Banda Arkangel R-15", "Banda Yuri", "Mi Banda El Mexicano", "Banda Maguey", "Banda Cuisillos", "Banda Pequeños Musical"],
        "canciones_rapidas": ["La Culebra", "El Baile del Caballito", "Al Gato y al Ratón", "La Chona", "El Tucanazo", "No Bailes de Caballito", "La Niña Fresa"],
        "canciones_moderadas": ["Vampiro", "La Roncona", "El Apagón", "Eva María", "Ramito de Violetas", "El Sonidito (Versión Banda)"],
        "canciones_lentas": ["Lindo Michoacán", "Un Indio Quiere Llorar", "Corrido de los Pérez", "Casas de Madera"],
        "canciones_principiantes": ["La Roncona", "El Apagón", "La Chona (a tiempo base)", "Ramito de Violetas"]
    },
    "Salsa": {
        "artistas": ["Héctor Lavoe", "Joe Arroyo", "Marc Anthony", "Roberto Roena", "Willie Colón", "Eddie Santiago", "Los Hermanos Lebrón", "Gilberto Santa Rosa", "Frankie Ruiz", "El Gran Combo de Puerto Rico", "Grupo Niche", "Ray Barretto"],
        "canciones_rapidas": ["Aguanile", "La Rebelión", "Que Se Sepa", "Rebelión", "Indestructible", "Anacaona", "Cali Pachanguero", "Mambo Gozón"],
        "canciones_moderadas": ["Valió la Pena", "Flor Pálida", "Marea de la Mar", "Gotas de Lluvia", "Llorarás", "Deseándote", "Tú Con Él"],
        "canciones_lentas": ["Lluvia", "Gitana", "Sobredosis", "Con Conciencia", "Ven Devórame Otra Vez", "Casi Un Hechizo", "Aquel Viejo Motel"],
        "canciones_principiantes": ["Flor Pálida", "Valió la Pena", "Gitana", "Idilio", "Tu Amor Me Hace Bien"]
    },
    "Bachata": {
        "artistas": ["Romeo Santos", "Prince Royce", "Juan Luis Guerra", "Aventura", "Monchy & Alexandra", "Daniel Santacruz", "Kewin Cosmos", "Ataca & La Alemana Selection", "Zacarías Ferreira", "Frank Reyes"],
        "canciones_rapidas": ["Propuesta Indecente", "Darte un Beso", "Eres Mía", "Obsesión", "Su Veneno", "Carita de Inocente", "La Diabla"],
        "canciones_moderadas": ["Stand by Me", "El Perdedor", "Hilito", "Incondicional", "Deja Vu", "Sobredosis de Bachata"],
        "canciones_lentas": ["Burbujas de Amor", "Infidelidades", "Perdidos", "Dile al Amor", "Enséñame a Olvidar", "Hoja en Blanco", "Por Un Segundo"],
        "canciones_principiantes": ["Stand by Me", "Darte un Beso", "Burbujas de Amor", "Corazón Sin Cara", "Mi Corazoncito"]
    }
}

# 3. INICIALIZACIÓN DE ESTADOS DE SESIÓN
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **¡Hola! Soy Síncopa, tu asistente de Inteligencia Artificial especializado en preparación coreográfica.**\n\n"
                "Para aprovechar al máximo nuestra conversación, ten en cuenta lo que puedo hacer por ti:\n\n"
                "### 🎯 ¿Qué puedes pedirme?:\n"
                "1. **Análisis de Canción o Link:** Ingresa el título o pega un enlace de *Spotify, YouTube, SoundCloud o Apple Music* para clasificar su género (**Bachata, Salsa o Quebradita**) y calcular su tempo estimado (BPM).\n"
                "2. **Exigencia Física & Modalidad:** Descubre el nivel de exigencia física (1 a 10) si la rutina se baila en **Pareja, Grupo/Compañía o Solista**.\n"
                "3. **Acondicionamiento Físico:** Pídeme rutinas de ejercicio específicas (pliometría, disociación, agilidad) para aguantar el ritmo de la pista.\n"
                "4. **Recomendaciones de Vestuario y Calzado:** Pregúntame qué ropa o calzado es el ideal para el género analizado.\n"
                "5. **Sugerencias Dinámicas de Canciones:** Pídeme listas personalizadas (ej. *'bachatas lentas, salsas rápidas y quebraditas para principiantes'*).\n\n"
                "--- \n"
                "### 🛑 Límites del servicio:\n"
                "* ⚠️ *No se procesan archivos locales subidos en formato audio (.mp3/.wav).* Por favor comparte el enlace o título.\n"
                "* ⚠️ *Especializado exclusivamente en género tropical y latino:* **Bachata, Salsa y Quebradita**.\n"
                "* ⚠️ *Las métricas y tempos (BPM) son estimaciones algorítmicas de orientación pedagógica y entrenamiento.*"
            )
        }
    ]

if "ultima_evaluacion" not in st.session_state:
    st.session_state.ultima_evaluacion = None
if "ultimo_genero_sugerido" not in st.session_state:
    st.session_state.ultimo_genero_sugerido = "Bachata"
if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []

# 4. FUNCIONES DE EXTRACCIÓN Y GENERACIÓN DINÁMICA
def obtener_titulo_desde_link(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=4)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                titulo = og_title["content"]
            elif soup.title and soup.title.string:
                titulo = soup.title.string
            else:
                titulo = "Enlace de Audio/Video"

            sufijos = [" | Spotify", " - YouTube", " on SoundCloud", " en Apple Music", " - Apple Music", " - Topic"]
            for s in sufijos:
                titulo = titulo.replace(s, "")

            return titulo.strip()
    except Exception:
        pass
    
    return "Canción por Enlace Externo"

def analizar_pista(query):
    q = query.lower().strip()
    
    tokens_no_musica = ["podcast", "entrevista", "interview", "vlog", "hablado", "conferencia", "noticias", "discurso"]
    if any(t in q for t in tokens_no_musica):
        return {"es_musica": False, "razon": "Contenido No Musical / Voz Hablada"}

    es_link = any(domain in q for domain in ["spotify.com", "youtube.com", "youtu.be", "soundcloud.com", "apple.com", "drive.google", ".mp3", ".wav"])
    
    if es_link:
        match = re.search(r'https?://[^\s]+', query)
        url_detectada = match.group(0) if match else query
        cancion_nombre = obtener_titulo_desde_link(url_detectada)
    else:
        cancion_nombre = query.title()

    hash_val = int(hashlib.md5(q.encode('utf-8')).hexdigest(), 16)

    tokens_quebradita = ["quebradita", "quebraditas", "banda", "zapateado", "brinco", "fast", "roncona", "culebra", "caballito", "vaquero", "machos", "arkangel", "tucanes", "vampiro"]
    tokens_bachata = ["bachata", "bachatas", "sensual", "bolero", "slow", "suave", "romantica", "romeo", "aventura", "prince", "royce", "guerra"]
    tokens_salsa = ["salsa", "salsas", "mambo", "guaguanco", "son", "timba", "marc anthony", "lavoe", "colon", "arroyo"]

    cadena_eval = (query + " " + cancion_nombre).lower()

    if any(w in cadena_eval for w in tokens_quebradita):
        tempo_base = 240.0 + (hash_val % 15)
        secciones_base = 12
    elif any(w in cadena_eval for w in tokens_bachata):
        tempo_base = 122.0 + (hash_val % 12)
        secciones_base = 7
    elif any(w in cadena_eval for w in tokens_salsa):
        tempo_base = 178.0 + (hash_val % 20)
        secciones_base = 9
    else:
        tempo_base = 135.0 + (hash_val % 50)
        secciones_base = 8

    return {
        "es_musica": True,
        "es_link": es_link,
        "tempo": round(tempo_base, 1),
        "secciones": secciones_base,
        "cancion_formateada": cancion_nombre
    }

def obtener_metricas_multi_modalidad(genero_predicho):
    if genero_predicho == "Quebradita":
        base = 8.5
        recom = "⭐ **Sugerencia:** ¡Ideal para **Grupo / Compañía** por el impacto visual de los lanzamientos y bloques sincronizados!"
        ejercicios = (
            "* 🦘 **Pliometría & Potencia:** Jump squats y salto de cuerda rápido (3 series x 45 seg).\n"
            "* 🦶 **Fuerza de Tobillo y Gemelos:** Elevaciones de talón para zapateado continuo.\n"
            "* 🫁 **Resistencia Cardiovascular HIIT:** Intervalos de 30s esfuerzo / 15s descanso."
        )
        metrica_ritmo = "⏱️ **Estructura Métrica:** Compás 2/4 rápido.\n👉 **Acentuación:** Marcación rápida constante. ¡Sincroniza los saltos e impulsos en los tiempos fuertes **1 y 2**!"
    elif genero_predicho == "Salsa":
        base = 7.0
        recom = "⭐ **Sugerencia:** Funciona excelente tanto en **Pareja** (turn patterns) como en **Solista** para lucir Shines/Footwork."
        ejercicios = (
            "* ⚡ **Agilidad de Pies (Footwork):** Escalera de agilidad para rapidez en shines.\n"
            "* 🔄 **Estabilidad de Core & Giros:** Planchas dinámicas y giros spot focalizados.\n"
            "* 🦵 **Movilidad de Cadera:** Disociación pélvica y fortalecimiento de abductores."
        )
        metrica_ritmo = "⏱️ **Estructura Métrica:** Fraseo de 8 tiempos (Clave 2/3 o 3/2).\n👉 **Acentuación:** Paso básico en **1, 2, 3** (pausa en 4) y **5, 6, 7** (pausa en 8). Opciones de acento On1 u On2 según el arreglo."
    else: # Bachata
        base = 5.0
        recom = "⭐ **Sugerencia:** Perfecta para **Pareja** por la conexión y fluidez en ondas/sensual, o **Solista** para disociación."
        ejercicios = (
            "* 🌊 **Disociación Corporal:** Aislación torácica y ondas de torso.\n"
            "* 🧘 **Flexibilidad & Control:** Estiramientos de isquiotibiales y movilidad de cadera.\n"
            "* 🛡️ **Fuerza de Postura (Marco):** Remo con liga e isometría de deltoides."
        )
        metrica_ritmo = "⏱️ **Estructura Métrica:** Compás 4/4 (Fraseo de 8 tiempos).\n👉 **Acentuación:** Pasos en **1, 2, 3** con **Tap / Golpe de Cadera** en el tiempo **4** (y **5, 6, 7** con Tap en **8**)."

    return {
        "pareja": round(base, 1),
        "grupo": round(min(10.0, base + 1.5), 1),
        "solista": round(min(10.0, base + 1.0), 1),
        "recomendacion_estilo": recom,
        "ejercicios_recomendados": ejercicios,
        "metrica_ritmo": metrica_ritmo
    }

def generar_sugerencias_dinamicas(genero, filtro, cantidad=3):
    """Genera combinaciones de canciones de forma dinámica e impredecible."""
    gen_data = ARTISTAS_Y_ESTILOS.get(genero, ARTISTAS_Y_ESTILOS["Bachata"])
    artistas = gen_data["artistas"]
    
    clave_cancion = f"canciones_{filtro}"
    if clave_cancion in gen_data:
        canciones = gen_data[clave_cancion]
    else:
        # Si pide algo variado o sin filtro exacto, junta todas las canciones disponibles
        canciones = (
            gen_data.get("canciones_rapidas", []) + 
            gen_data.get("canciones_lentas", []) + 
            gen_data.get("canciones_moderadas", [])
        )

    # Muestreo aleatorio dinámico
    canciones_seleccionadas = random.sample(canciones, min(len(canciones), cantidad))
    artistas_seleccionados = random.sample(artistas, min(len(artistas), cantidad))

    resultados = []
    for c, a in zip(canciones_seleccionadas, artistas_seleccionados):
        resultados.append(f"{c} - {a}")
    
    return resultados

def procesar_sub_peticion(sub_prompt):
    sp = sub_prompt.lower().strip()
    
    if "quebradita" in sp or "quebraditas" in sp:
        gen = "Quebradita"
    elif "salsa" in sp or "salsas" in sp:
        gen = "Salsa"
    elif "bachata" in sp or "bachatas" in sp:
        gen = "Bachata"
    else:
        gen = st.session_state.ultimo_genero_sugerido

    if any(w in sp for w in ["principiante", "principiantes", "facil", "fácil", "basica", "básica", "iniciacion"]):
        filtro = "principiantes"
        etiqueta = "para Principiantes / Ritmo Claro"
    elif any(w in sp for w in ["lenta", "lentas", "suave", "suaves", "romantica", "románticas", "sensual", "despacio"]):
        filtro = "lentas"
        etiqueta = "Lentas / Románticas"
    elif any(w in sp for w in ["rapida", "rápidas", "rapidas", "movida", "movidas", "prendida", "prendidas", "fast", "fuerte"]):
        filtro = "rapidas"
        etiqueta = "Rápidas / Alta Intensidad"
    elif any(w in sp for w in ["moderada", "moderadas", "intermedia", "intermedias"]):
        filtro = "moderadas"
        etiqueta = "Velocidad Moderada"
    else:
        filtro = "variadas"
        etiqueta = "Variadas y Populares"

    # Generar de forma 100% aleatoria y dinámica en cada petición
    canciones_dinamicas = generar_sugerencias_dinamicas(gen, filtro)

    return gen, etiqueta, canciones_dinamicas

# 5. PESTAÑAS PRINCIPALES (CHAT VS HISTORIAL)
tab_chat, tab_historial = st.tabs(["💬 Asistente Conversacional", "📊 Historial de Análisis"])

with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe una canción, pega un link o pide listas (ej. bachatas lentas, salsas rápidas)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        p_lower = prompt.lower()

        palabras_vestuario = ["vestuario", "ropa", "ropa sugerida", "que me pongo", "qué me pongo", "outfit", "traje"]
        palabras_calzado = ["calzado", "zapatos", "tenis", "zapatillas", "suela"]
        palabras_entrenamiento = ["ejercicio", "ejercicios", "entrenamiento", "rutina", "preparacion", "preparación", "footwork", "pasos", "aguantar", "fisico", "físico"]

        es_pregunta_vestuario = any(w in p_lower for w in palabras_vestuario)
        es_pregunta_calzado = any(w in p_lower for w in palabras_calzado)
        es_pregunta_entrenamiento = any(w in p_lower for w in palabras_entrenamiento)

        if (es_pregunta_vestuario or es_pregunta_calzado or es_pregunta_entrenamiento) and st.session_state.ultima_evaluacion:
            eval_previa = st.session_state.ultima_evaluacion
            gen_previo = eval_previa["genero"]
            cancion_previa = eval_previa["cancion"]
            mm_prev = obtener_metricas_multi_modalidad(gen_previo)

            if es_pregunta_vestuario:
                if gen_previo == "Quebradita":
                    recom_v = "👗 **Vestuario Sugerido para Quebradita:**\n* Vestuario vaquero estilizado pero ligero con strech flexible y flecos con movimiento."
                elif gen_previo == "Salsa":
                    recom_v = "👗 **Vestuario Sugerido para Salsa:**\n* Flecos, pedrería o trajes de corte estilizado que acentúen la rotación de cadera y hombros."
                else:
                    recom_v = "👗 **Vestuario Sugerido para Bachata:**\n* Prendas entalladas de tela suave/elástica que permitan apreciar la disociación corporal."

                reply = f"💃 **Recomendación de Vestuario para *{cancion_previa}* ({gen_previo}):**\n\n{recom_v}"

            elif es_pregunta_calzado:
                if gen_previo == "Quebradita":
                    recom_c = "👟 **Calzado:** Botas flex con suela de amortiguación o tenis deportivos con soporte en tobillos."
                elif gen_previo == "Salsa":
                    recom_c = "👠 **Calzado:** Zapatos de baile latino con suela de ante/gamuza para giros rápidos."
                else:
                    recom_c = "👠 **Calzado:** Zapatos de bachata de flexión completa o tenis de baile urbano."

                reply = f"👠 **Calzado Recomendado para *{cancion_previa}* ({gen_previo}):**\n\n{recom_c}"

            elif es_pregunta_entrenamiento:
                reply = f"🏋️ **Acondicionamiento Físico Recomendado para *{cancion_previa}* ({gen_previo}):**\n\n{mm_prev['ejercicios_recomendados']}"

            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

        elif any(w in p_lower for w in ["sugerencia", "sugerencias", "sugieres", "sugiere", "recomienda", "opciones", "lista", "listas", "bachata", "bachatas", "salsa", "salsas", "quebradita", "quebraditas"]):
            partes = re.split(r',| y | e ', p_lower)
            
            bloques_respuesta = []
            for parte in partes:
                if parte.strip():
                    gen, etiqueta, canciones = procesar_sub_peticion(parte)
                    if canciones:
                        items = "\n".join([f"  * 🎶 **{c}**" for c in canciones])
                        bloques_respuesta.append(f"### 🎶 {gen} ({etiqueta}):\n{items}")

            if bloques_respuesta:
                reply = "¡Claro! Aquí tienes tus sugerencias dinámicas personalizadas:\n\n" + "\n\n".join(bloques_respuesta) + "\n\n*¿Te gustaría analizar alguna de estas en detalle? Solo escribe su nombre o pega su enlace.*"
            else:
                reply = "🎶 No encontré sugerencias exactas para esa combinación, pero puedes pedirme listas como *'bachatas lentas'*, *'salsas para principiantes'* o *'quebraditas rápidas'*."

            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

        else:
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analizando pista y clasificando género..."):
                    time.sleep(0.3)
                    analisis = analizar_pista(prompt)

            if not analisis["es_musica"]:
                reply = "⚠️ La entrada parece ser un contenido hablado o no musical. Por favor, ingresa el nombre de una canción o pega un enlace de audio/video."
            else:
                tempo_val = analisis["tempo"]
                secciones_val = analisis["secciones"]

                if modelo is not None:
                    df_in = pd.DataFrame({'tempo': [tempo_val], 'num_secciones': [secciones_val]})
                    prediccion_ml = modelo.predict(df_in)[0]
                else:
                    prediccion_ml = "Quebradita" if tempo_val > 220 else ("Bachata" if tempo_val < 140 else "Salsa")

                mm = obtener_metricas_multi_modalidad(prediccion_ml)

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

                # Opciones dinámicas relacionadas
                sug_rel = generar_sugerencias_dinamicas(prediccion_ml, "variadas", cantidad=3)
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

            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
                if analisis["es_musica"]:
                    st.download_button(
                        label="📥 Descargar Ficha Técnica (.txt)",
                        data=ficha_texto,
                        file_name=f"Ficha_{analisis['cancion_formateada'].replace(' ', '_')}.txt",
                        mime="text/plain"
                    )

# 6. PESTAÑA DE HISTORIAL Y MÉTRICAS ACUMULADAS
with tab_historial:
    st.subheader("📈 Resumen de Pistas Analizadas en esta Sesión")
    if len(st.session_state.historial_evaluaciones) > 0:
        df_historial = pd.DataFrame(st.session_state.historial_evaluaciones)
        st.dataframe(df_historial, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Pistas Analizadas", len(df_historial))
        with col2:
            st.metric("Promedio de Tempo (BPM)", f"{df_historial['Tempo (BPM)'].mean():.1f}")
            
        csv_data = df_historial.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exportar Historial Completo a CSV",
            data=csv_data,
            file_name="historial_sincopa.csv",
            mime="text/csv"
        )
    else:
        st.info("Aún no has analizado ninguna pista durante esta sesión. ¡Inicia en el chat enviando el nombre o link de una canción!")
