import streamlit as st
import numpy as np
import time
import joblib
import os
import pandas as pd
import hashlib

st.set_page_config(
    page_title="Síncopa • Asistente Coreográfico IA",
    page_icon="💃",
    layout="centered"
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

# 2. BASE DE DATOS AMPLIADA DE SUGERENCIAS
SUGERENCIAS_GENERO = {
    "Quebradita": [
        "La Culebra - Banda Machos",
        "La Roncona - Banda Arkangel R-15",
        "El Apagón - Banda Yuri",
        "El Baile del Caballito - Banda Machos",
        "Vampiro - Banda Machos",
        "La Chona - Los Tucanes de Tijuana",
        "Al Gato y al Ratón - Banda Machos"
    ],
    "Salsa": [
        "Aguanile - Héctor Lavoe",
        "Valió la Pena - Marc Anthony",
        "Lluvia - Eddie Santiago",
        "La Rebelión - Joe Arroyo",
        "Gitana - Willie Colón",
        "Sobredosis - Los Hermanos Lebrón"
    ],
    "Bachata": [
        "Propuesta Indecente - Romeo Santos",
        "Dile al Amor - Aventura",
        "Darte un Beso - Prince Royce",
        "Burbujas de Amor - Juan Luis Guerra",
        "Eres Mía - Romeo Santos",
        "Obsesión - Aventura"
    ]
}

# 3. INICIALIZACIÓN DE ESTADOS DE SESIÓN
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **¡Hola! Soy Síncopa, tu Asistente Coreográfico.**\n\n¿Qué canción o artista te gustaría analizar hoy?"
        }
    ]

if "step" not in st.session_state:
    st.session_state.step = "esperando_cancion"
if "cancion" not in st.session_state:
    st.session_state.cancion = ""
if "analisis_pista" not in st.session_state:
    st.session_state.analisis_pista = None
if "modalidad" not in st.session_state:
    st.session_state.modalidad = ""
if "ultima_evaluacion" not in st.session_state:
    st.session_state.ultima_evaluacion = None
if "ultimo_genero_sugerido" not in st.session_state:
    st.session_state.ultimo_genero_sugerido = "Bachata"
if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. FUNCIONES COREOGRÁFICAS
def analizar_pista(query):
    q = query.lower().strip()
    
    tokens_no_musica = ["podcast", "entrevista", "interview", "vlog", "hablado", "conferencia", "noticias", "discurso", "audiobook"]
    if any(t in q for t in tokens_no_musica):
        return {"es_musica": False, "razon": "Contenido No Musical / Voz Hablada"}

    es_link = any(domain in q for domain in ["spotify.com", "youtube.com", "youtu.be", "drive.google", ".mp3", ".wav"])

    tokens_fuera_de_dominio = [
        "quedate en silencio", "quédate en silencio", "rbd", "rebelde", "pop", "balada", 
        "perla", "rosalia", "rosalía", "flamenco", "rumba", "reggaeton", "reggaetón", 
        "tumbado", "corrido", "rock", "hip hop", "rap", "merengue", "cumbia", "bad bunny", "karol g"
    ]
    
    if (any(t in q for t in tokens_fuera_de_dominio) or "estimar de todos modos" in q) and not es_link:
        return {
            "es_musica": True, 
            "fuera_de_dominio": True, 
            "genero_detectado": "Pista fuera del índice directo de Salsa, Bachata o Quebradita",
            "cancion_formateada": query.title()
        }

    hash_val = int(hashlib.md5(q.encode('utf-8')).hexdigest(), 16)

    tokens_quebradita = ["quebradita", "quebraditas", "banda", "zapateado", "brinco", "fast", "roncona", "culebra", "caballito", "vaquero", "machos", "arkangel"]
    tokens_bachata = ["bachata", "bachatas", "sensual", "bolero", "slow", "suave", "romantica", "romeo", "aventura", "prince royce", "juan luis guerra"]
    tokens_salsa = ["salsa", "salsas", "mambo", "guaguanco", "son", "timba", "marc anthony", "havana d'primera", "maykel blanco", "niche", "lavoe"]

    if any(w in q for w in tokens_quebradita):
        tempo_base = 240.0 + (hash_val % 20)
        secciones_base = 12 + (hash_val % 4)
    elif any(w in q for w in tokens_bachata):
        tempo_base = 120.0 + (hash_val % 15)
        secciones_base = 7 + (hash_val % 3)
    elif any(w in q for w in tokens_salsa):
        tempo_base = 175.0 + (hash_val % 25)
        secciones_base = 9 + (hash_val % 5)
    elif es_link:
        tempo_base = 130.0 + (hash_val % 110)
        secciones_base = 6 + (hash_val % 7)
    else:
        return {
            "es_musica": True, 
            "fuera_de_dominio": True, 
            "genero_detectado": "Pista fuera del índice directo de Salsa, Bachata o Quebradita",
            "cancion_formateada": query.title()
        }

    return {
        "es_musica": True,
        "fuera_de_dominio": False,
        "es_link": es_link,
        "tempo": round(tempo_base, 1),
        "secciones": secciones_base,
        "cancion_formateada": "Pista por Enlace External/Audio" if es_link else query.title()
    }

def calcular_esfuerzo_y_metricas(prediccion_ml, modalidad):
    if prediccion_ml == "Quebradita":
        esfuerzo_base = 8.5
        velocidad_txt = "Muy Alta (Brinco/Zapateado)"
        bailabilidad_txt = "9.5 / 10 (Alta Pliometría)"
        enfasis_txt = "Potencia Pliométrica & Acrobacia"
    elif prediccion_ml == "Salsa":
        esfuerzo_base = 7.0
        velocidad_txt = "Alta (Shines/Giros)"
        bailabilidad_txt = "9.0 / 10 (Ritmo Complejo)"
        enfasis_txt = "Velocidad & Precisión"
    else:
        esfuerzo_base = 5.0
        velocidad_txt = "Moderada / Fluida"
        bailabilidad_txt = "9.8 / 10 (Sensual/Cadencia)"
        enfasis_txt = "Caderas, Disociación & Marco"

    if "Compañía" in modalidad or "Grupo" in modalidad:
        esfuerzo_final = min(10.0, esfuerzo_base + 1.5)
        mod_nota = "(+1.5 por limpieza de bloques y cañones en grupo)"
    elif "Solista" in modalidad:
        esfuerzo_final = min(10.0, esfuerzo_base + 1.0)
        mod_nota = "(+1.0 por dominio escénico continuo)"
    else:
        esfuerzo_final = esfuerzo_base
        mod_nota = "(Estándar para trabajo en pareja)"

    return {
        "esfuerzo": round(esfuerzo_final, 1),
        "mod_nota": mod_nota,
        "velocidad": velocidad_txt,
        "bailabilidad": bailabilidad_txt,
        "enfasis": enfasis_txt
    }

# 5. MANEJO DE ENTRADA DEL CHAT
if prompt := st.chat_input("Escribe tu duda, pide recomendaciones o ingresa una canción..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    p_lower = prompt.lower()

    # Comprobar si el usuario pide explícitamente cambiar de canción
    pide_nueva_cancion = any(w in p_lower for w in ["evaluar otra", "nueva canción", "otra canción", "analizar otra", "cambiar de canción", "reset"])

    # Detección ampliada de consultas de sugerencias y artistas
    palabras_sugerencia = [
        "sugerencia", "sugerencias", "sugieres", "sugiere", 
        "recomienda", "recomiendame", "recomiéndame", "recomendacion", "recomendaciones", 
        "opciones", "canciones", "cancion", "ideas", "que escuchar",
        "bachatas", "salsas", "quebraditas", "temas", "pistas",
        "romeo", "santos", "aventura", "prince", "royce", "guerra", "lavoe", "anthony"
    ]
    
    menciona_genero_plural = any(g in p_lower for g in ["bachatas", "salsas", "quebraditas"])
    es_pregunta_sugerencia = any(w in p_lower for w in palabras_sugerencia) or menciona_genero_plural

    # --- CONSULTA DE RECOMENDACIÓN DIRECTA EN ESTADO INICIAL ---
    if es_pregunta_sugerencia and st.session_state.step == "esperando_cancion":
        pide_mas = any(w in p_lower for w in ["mas", "más", "otras", "otros", "extra"])

        if "quebradita" in p_lower or "quebraditas" in p_lower or "banda" in p_lower:
            gen_buscado = "Quebradita"
        elif "bachata" in p_lower or "bachatas" in p_lower or any(art in p_lower for art in ["romeo", "santos", "aventura", "prince", "royce", "guerra"]):
            gen_buscado = "Bachata"
        elif "salsa" in p_lower or "salsas" in p_lower or any(art in p_lower for art in ["lavoe", "anthony", "colon", "arroyo"]):
            gen_buscado = "Salsa"
        elif pide_mas and st.session_state.ultimo_genero_sugerido:
            gen_buscado = st.session_state.ultimo_genero_sugerido
        else:
            gen_buscado = st.session_state.ultimo_genero_sugerido if st.session_state.ultimo_genero_sugerido else "Bachata"

        # Guardar contexto
        st.session_state.ultimo_genero_sugerido = gen_buscado

        mod_txt = "Pareja / Mixto"
        if "grupo" in p_lower or "compañia" in p_lower or "compañía" in p_lower or "grupos" in p_lower:
            mod_txt = "Grupo / Compañía"
        elif "solista" in p_lower or "solistas" in p_lower or "individual" in p_lower:
            mod_txt = "Solista / Individual"

        sug_base = SUGERENCIAS_GENERO.get(gen_buscado, [])

        # Filtrado por artista si aplica
        artistas_clave = ["romeo", "santos", "aventura", "prince", "royce", "guerra", "lavoe", "anthony", "machos", "arkangel", "tucanes"]
        menciones_artista = [art for art in artistas_clave if art in p_lower]

        if menciones_artista:
            sug_filtradas = [s for s in sug_base if any(art in s.lower() for art in menciones_artista)]
            sug_finales = sug_filtradas if sug_filtradas else sug_base
        else:
            sug_finales = sug_base

        items_txt = "\n".join([f"* 🎶 **{s}**" for s in sug_finales])
        bot_reply = f"🎶 **Recomendaciones de {gen_buscado} (para {mod_txt}):**\n\nAquí tienes las opciones con la métrica y ritmo adecuado:\n\n{items_txt}\n\n*¿Cuál de estas te gustaría evaluar formalmente? Escribe su nombre.*"
        
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

    # --- CASO A: RESPUESTAS SOBRE LA CANCIÓN ACTUAL O HISTORIAL (ESTADO EVALUADO) ---
    elif st.session_state.step == "evaluado" and not pide_nueva_cancion:
        eval_act = st.session_state.ultima_evaluacion
        genero = eval_act["genero"]
        cancion_nombre = eval_act["cancion"]
        tempo_val = eval_act["tempo"]

        # A1: Solicitud de Top / Ranking / Mejores canciones evaluadas
        if any(w in p_lower for w in ["top", "ranking", "mejores", "tabla", "resumen de evaluadas"]):
            if "quebradita" in p_lower or "quebraditas" in p_lower or "banda" in p_lower:
                gen_top = "Quebradita"
            elif "bachata" in p_lower or "bachatas" in p_lower:
                gen_top = "Bachata"
            elif "salsa" in p_lower or "salsas" in p_lower:
                gen_top = "Salsa"
            else:
                gen_top = genero or st.session_state.ultimo_genero_sugerido

            evaluadas_genero = [p for p in st.session_state.historial_evaluaciones if p["genero"] == gen_top]

            if not evaluadas_genero:
                canciones_base = SUGERENCIAS_GENERO.get(gen_top, [])[:5]
                lista_txt = "\n".join([f"{i+1}. 🎶 **{c}**" for i, c in enumerate(canciones_base)])
                respuesta_seguimiento = f"🏆 **Top 5 Sugerido de {gen_top} (Catálogo Base):**\n\nAún no se han evaluado varias pistas de {gen_top} en esta sesión, pero aquí tienes las 5 mejores opciones recomendadas para entrenamiento:\n\n{lista_txt}"
            else:
                top_ordenado = sorted(evaluadas_genero, key=lambda x: x["esfuerzo"], reverse=True)[:5]
                filas = [f"{i+1}. 🎶 **{p['cancion']}** — Exigencia: **{p['esfuerzo']}/10** (~{p['tempo']} BPM)" for i, p in enumerate(top_ordenado)]
                lista_txt = "\n".join(filas)
                respuesta_seguimiento = f"🏆 **Top {len(top_ordenado)} de Canciones Evaluadas de {gen_top}:**\n\nBasado en lo que hemos analizado en esta sesión, aquí tienes el ranking ordenado por nivel de exigencia física:\n\n{lista_txt}"

        # A2: Consultas específicas sobre tiempo/minutos de footwork
        elif any(w in p_lower for w in ["cuantos minutos", "cuánto tiempo", "duracion", "duración", "tiempo de footwork"]):
            if genero == "Salsa":
                respuesta_seguimiento = f"⏱️ **Tiempo de Footwork recomendado para '{cancion_nombre}':**\n\nDado el tempo de **~{tempo_val} BPM**, lo recomendable en una estructura estándar de 3 a 4 minutos es reservar entre **45 segundos y 1.5 minutos** (unos 4 a 8 ochos) para el bloque principal de *Shines/Footwork*. Esto suele hacerse durante las descargas de percusión o solos de mambo."
            elif genero == "Bachata":
                respuesta_seguimiento = f"⏱️ **Tiempo de Footwork recomendado para '{cancion_nombre}':**\n\nEn una bachata (~{tempo_val} BPM), se sugiere dedicar entre **30 y 60 segundos** de footwork sincopado (paso básico avanzado/taps) repartidos en los puentes musicales o solos de requinto."
            else: # Quebradita
                respuesta_seguimiento = f"⏱️ **Tiempo de Zapateado/Footwork para '{cancion_nombre}':**\n\nDebido al ritmo vertiginoso (~{tempo_val} BPM), el zapateado se mantiene casi durante toda la pieza, pero los bloques intensos de footwork solista deben ser de **20 a 40 segundos** para evitar fatiga."

        # A3: Preguntas generales de Footwork / Shines
        elif any(w in p_lower for w in ["footwork", "pasos", "shines", "zapateado", "pies", "brincos"]):
            if genero == "Salsa":
                respuesta_seguimiento = f"👟 **Footwork en '{cancion_nombre}' (Salsa):**\n\n¡Absolutamente! Con un tempo de ~{tempo_val} BPM, esta pista es ideal para incluir **Shines / Footwork rápido** en los cortes de mambo. Te recomiendo incorporar marcaciones sincopadas, susy q, y pasadas de pie a tiempo y contratiempo."
            elif genero == "Bachata":
                respuesta_seguimiento = f"👟 **Footwork en '{cancion_nombre}' (Bachata):**\n\n¡Por supuesto! En Bachata el footwork se luce en los cortes instrumentales. Puedes integrar *sincopados en 1 y 2*, *doble tap*, y desplazamientos laterales con disociación."
            else:
                respuesta_seguimiento = f"👟 **Footwork / Zapateado en '{cancion_nombre}' (Quebradita):**\n\n¡Definitivamente! El footwork aquí es **Zapateado continuo y brincos alternados**, combinados con remates al compás."

        # A4: Recomendaciones de canciones o sugerencias por género/artista
        elif any(w in p_lower for w in ["similar", "parecida", "recomienda", "sugieres", "sugiere", "opciones", "mismo estilo", "canciones", "cancion", "que canciones", "sugerencia", "sugerencias", "bachatas", "salsas", "quebraditas", "romeo", "santos", "aventura", "prince", "royce", "lavoe", "anthony"]):
            gen_buscado = genero
            if "salsa" in p_lower or "salsas" in p_lower:
                gen_buscado = "Salsa"
            elif "bachata" in p_lower or "bachatas" in p_lower or any(art in p_lower for art in ["romeo", "santos", "aventura", "prince", "royce"]):
                gen_buscado = "Bachata"
            elif "quebradita" in p_lower or "quebraditas" in p_lower or "banda" in p_lower:
                gen_buscado = "Quebradita"

            mod_actual = eval_act['modalidad']
            if "grupo" in p_lower or "compañia" in p_lower or "compañía" in p_lower or "mixto" in p_lower:
                mod_actual = "Grupo / Compañía"
            elif "solista" in p_lower or "solistas" in p_lower or "individual" in p_lower:
                mod_actual = "Solista / Individual"
            elif "pareja" in p_lower:
                mod_actual = "Pareja"

            st.session_state.ultimo_genero_sugerido = gen_buscado
            sug_base = SUGERENCIAS_GENERO.get(gen_buscado, [])

            # Filtrado por artista
            artistas_clave = ["romeo", "santos", "aventura", "prince", "royce", "guerra", "lavoe", "anthony", "machos", "arkangel", "tucanes"]
            menciones_artista = [art for art in artistas_clave if art in p_lower]

            if menciones_artista:
                sug_filtradas = [s for s in sug_base if any(art in s.lower() for art in menciones_artista)]
                sug_finales = sug_filtradas if sug_filtradas else sug_base
            else:
                sug_finales = sug_base

            items_txt = "\n".join([f"* 🎶 **{s}**" for s in sug_finales])
            subtitulo_artista = f" de **{prompt.title()}**" if menciones_artista else ""
            respuesta_seguimiento = f"🎶 **Sugerencias de {gen_buscado}{subtitulo_artista} (ideales para {mod_actual}):**\n\nAquí tienes las opciones correspondientes en nuestro catálogo:\n\n{items_txt}"

        # A5: Exigencia física o métricas
        elif any(w in p_lower for w in ["exigencia", "fisica", "física", "esfuerzo", "puntuacion", "puntuación", "métrica"]):
            respuesta_seguimiento = f"📊 **Exigencia Física de '{cancion_nombre}':**\n\n* **Nivel:** **{eval_act['metricas']['esfuerzo']} / 10** {eval_act['metricas']['mod_nota']}\n* **Velocidad:** {eval_act['metricas']['velocidad']}\n* **Formato:** {eval_act['modalidad']}"

        # A6: Calzado o Vestuario
        elif any(w in p_lower for w in ["tacones", "calzado", "zapatos", "vestuario", "ropa", "tenis"]):
            respuesta_seguimiento = f"👠 **Calzado y Vestuario para '{cancion_nombre}' ({genero}):**\n\nPara esta rutina en formato **{eval_act['modalidad']}**, se sugiere utilizar vestuario dinámico con buena movilidad. En calzado: tacones profesionales de flexión (7.5 - 9 cm) para baile latino o botines de cuero con suela de gamuza si es rol masculino/salsa dura."

        # A7: Respuesta conversacional por defecto en estado evaluado
        else:
            respuesta_seguimiento = f"💡 **Síncopa:** Sobre **'{cancion_nombre}'** ({genero}): Puedes preguntarme por minutos de footwork, pedir un **Top 5** de canciones evaluadas, calzado, vestuario o métricas de esfuerzo.\n\n*(Escribe 'analizar otra canción' cuando quieras evaluar una pista nueva)*."

        st.session_state.messages.append({"role": "assistant", "content": respuesta_seguimiento})
        with st.chat_message("assistant"):
            st.markdown(respuesta_seguimiento)

    # --- CASO B: PREGUNTAS TÉCNICAS GENERALES FUERA DE EVALUACIÓN ---
    elif any(w in p_lower for w in ["tiempo", "conteo", "tacones", "calzado"]) and len(prompt.split()) > 3 and st.session_state.step == "esperando_cancion":
        respuesta_directa = "💡 **Respuesta de Síncopa:** La Salsa y Bachata se bailan a 8 tiempos, mientras la Quebradita es en compás rápido de 2/4. En escenario para Salsa/Bachata se sugieren tacones profesionales (7.5 - 9 cm) y para Quebradita tenis de amortiguación."
        st.session_state.messages.append({"role": "assistant", "content": respuesta_directa})
        with st.chat_message("assistant"):
            st.markdown(respuesta_directa)

    # --- CASO C: PASO 1 - INICIO DE EVALUACIÓN DE CANCIÓN ---
    elif st.session_state.step == "esperando_cancion" or pide_nueva_cancion:
        st.session_state.step = "esperando_cancion"
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analizando pista..."):
                time.sleep(0.3)
                analisis = analizar_pista(prompt)

        if not analisis["es_musica"]:
            bot_reply = "⚠️ **Guardrail de Audición Activado:** La pista fue clasificada como *Contenido No Musical / Voz Hablada*. Por favor ingresa una canción o enlace musical."
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.markdown(bot_reply)

        elif analisis.get("fuera_de_dominio", False):
            bot_reply = f"""⚠️ **Pista No Detectada en el Catálogo Base**

🎵 **Búsqueda:** {analisis['cancion_formateada']}
📌 **Diagnóstico:** {analisis['genero_detectado']}

---

💡 **¿Quieres que la evaluemos de todos modos por Audio/Link?**
Pega aquí abajo un **link de Spotify, YouTube** o la referencia de tu archivo. 

> ⚠️ *Nota de Transparencia:* La clasificación se estimará mapeando contra nuestras **métricas acústicas base**:
> 1. **Tempo estimado (BPM)**
> 2. **Número de Secciones Percusivas**
> 3. **Densidad de Golpes Métricos**"""
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.markdown(bot_reply)

        else:
            st.session_state.cancion = prompt
            st.session_state.analisis_pista = analisis
            st.session_state.step = "esperando_modalidad"
            
            disclaimer_txt = "\n\n*(Estimación realizada mediante análisis de parámetros acústicos)*" if analisis.get("es_link", False) else ""
            bot_reply = f"¡Excelente pista! 🎶 **'{analisis['cancion_formateada']}'**.{disclaimer_txt}\n\nPara ajustar el cálculo de exigencia física, dime: **¿La rutina será en Solista, Pareja o Compañía/Grupo?**"
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.markdown(bot_reply)

    # --- CASO D: PASO 2 - SELECCIÓN DE MODALIDAD ---
    elif st.session_state.step == "esperando_modalidad":
        if "solista" in p_lower or "solistas" in p_lower or "individual" in p_lower:
            st.session_state.modalidad = "Solista / Individual"
        elif "grupo" in p_lower or "compañia" in p_lower or "compañía" in p_lower or "grupos" in p_lower:
            st.session_state.modalidad = "Grupo / Compañía"
        else:
            st.session_state.modalidad = "Pareja"

        st.session_state.step = "esperando_rol"
        bot_reply = f"Anotado, formato **{st.session_state.modalidad}**.\n\nPor último: **¿Para qué rol/género va dirigida la rutina?** (Femenino/Bailarina, Masculino/Bailarín o Mixto)"
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

    # --- CASO E: PASO 3 - REPORTE FINAL Y PASO A ESTADO "EVALUADO" ---
    elif st.session_state.step == "esperando_rol":
        rol_user = prompt
        analisis = st.session_state.analisis_pista
        modalidad_txt = st.session_state.modalidad

        with st.chat_message("assistant"):
            with st.spinner("🤖 Generando evaluación coreográfica..."):
                time.sleep(0.4)
                
                tempo_val = analisis["tempo"]
                secciones_val = analisis["secciones"]

                if modelo is not None:
                    df_in = pd.DataFrame({'tempo': [tempo_val], 'num_secciones': [secciones_val]})
                    prediccion_ml = modelo.predict(df_in)[0]
                else:
                    prediccion_ml = "Quebradita" if tempo_val > 220 else ("Bachata" if tempo_val < 140 else "Salsa")

                metricas = calcular_esfuerzo_y_metricas(prediccion_ml, modalidad_txt)

                # Registrar en el historial de la sesión
                st.session_state.historial_evaluaciones.append({
                    "cancion": analisis['cancion_formateada'],
                    "genero": prediccion_ml,
                    "esfuerzo": metricas['esfuerzo'],
                    "tempo": tempo_val,
                    "modalidad": modalidad_txt
                })

                # Guardar evaluación actual
                st.session_state.ultima_evaluacion = {
                    "cancion": analisis['cancion_formateada'],
                    "genero": prediccion_ml,
                    "modalidad": modalidad_txt,
                    "tempo": tempo_val,
                    "metricas": metricas
                }
                st.session_state.step = "evaluado"

                if prediccion_ml == "Bachata":
                    calzado = "Tacones profesionales de baile (7.5 - 9 cm)" if "fem" in rol_user.lower() or "mix" in rol_user.lower() else "Zapatos de baile en piel suave"
                    rutina = "1. Disociación pélvica/torso (3x1 min)\n2. Taps y fortalecimiento de metatarsos\n3. Planchas para estabilidad de marco"
                elif prediccion_ml == "Salsa":
                    calzado = "Tacones profesionales de salsa (7.5 - 9 cm)" if "fem" in rol_user.lower() or "mix" in rol_user.lower() else "Botines/Zapatos de salsa en cuero"
                    rutina = "1. Escalera de agilidad (Ladder Drills)\n2. Cardio HIIT en bloques de 30s\n3. Prensas de hombro para estabilidad"
                else:
                    calzado = "Tenis deportivos de alto impacto con buena amortiguación"
                    rutina = "1. Pliometría (Salto de caja / Box jumps)\n2. Elevación de talones para articulaciones\n3. Sentadillas explosivas para acrobacias"

                sug_lista = SUGERENCIAS_GENERO.get(prediccion_ml, [])
                sug_txt = "\n".join([f"  * 🎶 {s}" for s in sug_lista[:3]])

                respuesta = f"""
🎶 **Pista Evaluada:** {analisis['cancion_formateada']}
📌 **Clasificación ML:** **{prediccion_ml}**

---

### 📊 Evaluación de la Pista
* ⚡ **Velocidad / Tempo:** {metricas['velocidad']} (~{tempo_val} BPM)
* 💃 **Bailabilidad:** {metricas['bailabilidad']}
* 🔥 **Exigencia Física Estimada:** **{metricas['esfuerzo']} / 10** {metricas['mod_nota']}
* 🎯 **Énfasis Coreográfico:** {metricas['enfasis']}

---

### 💡 Diagnóstico y Recomendaciones ({modalidad_txt})
* 👟 **Calzado Técnico:** {calzado}
* 👗 **Vestuario:** Diseñado para acompañar la velocidad y cortes de movimiento.

---

### 🏋️‍♀️ Plan de Entrenamiento Sugerido
{rutina}

---

💡 **¿Quieres explorar opciones similares?**
Aquí tienes algunas pistas recomendadas dentro del mismo género:
{sug_txt}

*(Puedes hacer preguntas de seguimiento sobre esta canción, pedir un **Top 5 de evaluadas**, cuántos minutos de footwork incluir o consultar otra canción)*
"""
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
