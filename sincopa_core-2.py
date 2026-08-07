# -*- coding: utf-8 -*-
"""
sincopa_core.py
================
Módulo único de verdad para Síncopa. Tanto el script de entrenamiento
(entrenar_modelo.py) como la app (app.py) importan de aquí, así que
JAMÁS pueden volver a desincronizarse: usan el mismo dataset real, el
mismo esquema de features y la misma lógica de clasificación.

FUENTE DE DATOS: dataset_bachata_salsa_quebradita.csv (482 canciones
reales, 3 clases: Bachata, Salsa, Quebradita). Las pistas de Timba
cubana (Los Van Van, Alexander Abreu, Issac Delgado...) están
etiquetadas como "Salsa" en este dataset — igual que planteaba el
notebook original ("Timba" como subgénero de Salsa, no como clase
aparte). Por eso el modelo tiene 3 clases, y "Timba" se maneja solo
como una variante estilística de despliegue (catálogo de sugerencias/
entrenamiento/vestuario), nunca como salida del clasificador — sería
inventar una clase que los datos reales no respaldan.

ESQUEMA DE FEATURES: exactamente las `features_candidatas` que ya
proponía el notebook original en `reentrenar_modelo_con_maestro()`,
y que están presentes en el CSV real:

    tempo, danceability, energy, valence, speechiness, acousticness,
    densidad_tatum, num_secciones, num_compases, num_tiempos_beats
"""

import os
import re
import random
import numpy as np
import pandas as pd

FEATURES = [
    "tempo", "danceability", "energy", "valence", "speechiness", "acousticness",
    "densidad_tatum", "num_secciones", "num_compases", "num_tiempos_beats",
]
GENEROS = ["Bachata", "Salsa", "Quebradita"]

RUTA_DATASET_DEFAULT = os.path.join(os.path.dirname(__file__), "dataset_bachata_salsa_quebradita.csv")


def cargar_dataset(ruta=RUTA_DATASET_DEFAULT):
    df = pd.read_csv(ruta)
    faltantes = [f for f in FEATURES if f not in df.columns]
    if faltantes:
        raise ValueError(f"El dataset no tiene las columnas requeridas: {faltantes}")
    return df


# ==========================================================================
# 1. RANGOS POR GÉNERO — CALCULADOS DESDE EL DATASET REAL (percentiles
#    5%-95% para no dejar que un outlier distorsione la estimación).
#    Se usan solo cuando el usuario escribe un título/enlace que NO
#    coincide con ninguna canción del dataset (ver buscar_en_dataset):
#    en ese caso no hay features reales que usar, así que se estima un
#    vector plausible dentro del rango real observado para ese género.
# ==========================================================================
def calcular_rangos_desde_dataset(df):
    rangos = {}
    for genero in df["genero_etiqueta"].unique():
        sub = df[df["genero_etiqueta"] == genero]
        rangos[genero] = {
            f: (float(sub[f].quantile(0.05)), float(sub[f].quantile(0.95))) for f in FEATURES
        }
    return rangos


def calcular_rango_ambiguo(rangos_genero):
    return {
        f: (min(r[f][0] for r in rangos_genero.values()), max(r[f][1] for r in rangos_genero.values()))
        for f in FEATURES
    }


def _muestrear(rangos, rng):
    return {f: round(rng.uniform(*rangos[f]), 4) for f in FEATURES}


def estimar_features(genero_detectado, rangos_genero, rango_ambiguo, rng=None):
    """Genera un vector de features plausible para el género detectado,
    muestreando del rango real (percentiles 5-95) observado en el
    dataset para ese género. Si no hay género (caso ambiguo), muestrea
    del rango combinado de todos los géneros, lo que produce a
    propósito una confianza baja en el modelo."""
    if rng is None:
        rng = np.random.default_rng()
    rangos = rangos_genero.get(genero_detectado, rango_ambiguo)
    return _muestrear(rangos, rng)


# ==========================================================================
# 1b. ESTADÍSTICAS POR GÉNERO (p25/p50/p75) — usadas para comparar una
#     canción concreta contra "lo típico" de su género y así generar
#     comentarios de aprovechamiento basados en datos reales, no en un
#     texto fijo idéntico para todas las canciones del mismo género.
# ==========================================================================
def calcular_estadisticas_genero(df):
    stats = {}
    for genero in df["genero_etiqueta"].unique():
        sub = df[df["genero_etiqueta"] == genero]
        stats[genero] = {
            f: {
                "p25": float(sub[f].quantile(0.25)),
                "p50": float(sub[f].quantile(0.50)),
                "p75": float(sub[f].quantile(0.75)),
            }
            for f in FEATURES
        }
    return stats


def _nivel(valor, p25, p75):
    """Clasifica un valor como 'bajo' / 'típico' / 'alto' frente a su
    género, usando los percentiles 25-75 como banda 'típica'."""
    if valor < p25:
        return "bajo"
    if valor > p75:
        return "alto"
    return "típico"


def comentar_aprovechamiento(genero_catalogo, genero_modelo, features, stats_genero):
    """Genera 3-5 observaciones de aprovechamiento coreográfico
    comparando las features REALES (o estimadas) de esta canción contra
    la distribución real del género en el dataset. Nada de texto fijo:
    cada canción recibe un comentario distinto según sus propios números."""
    stats = stats_genero.get(genero_modelo, {})
    if not stats:
        return "No hay suficientes datos de referencia para este género."

    bullets = []

    # Tempo relativo al género
    t = features["tempo"]
    st = stats["tempo"]
    nivel_t = _nivel(t, st["p25"], st["p75"])
    if nivel_t == "alto":
        bullets.append(f"⏩ **Tempo alto para {genero_catalogo}** ({t:.1f} BPM vs. mediana de "
                        f"{st['p50']:.1f}): exige más resistencia y velocidad de pies; buena para "
                        "mostrar energía y potencia en competencia.")
    elif nivel_t == "bajo":
        bullets.append(f"⏸️ **Tempo más pausado que lo típico** ({t:.1f} BPM vs. mediana de "
                        f"{st['p50']:.1f}): favorece el trabajo de detalle, aislamientos y conexión, "
                        "aunque puede restar impacto visual en tarima si el resto del set es rápido.")
    else:
        bullets.append(f"✅ **Tempo típico de {genero_catalogo}** ({t:.1f} BPM, cerca de la mediana de "
                        f"{st['p50']:.1f}): terreno seguro para una coreografía estándar del género.")

    # Densidad rítmica (tatum) -> complejidad de subdivisión para footwork
    dt = features["densidad_tatum"]
    sdt = stats["densidad_tatum"]
    nivel_dt = _nivel(dt, sdt["p25"], sdt["p75"])
    if nivel_dt == "alto":
        bullets.append(f"🥁 **Alta densidad de subdivisión rítmica** ({dt:.2f} vs. mediana de "
                        f"{sdt['p50']:.2f}): hay espacio para footwork rápido y síncopas; aprovéchalo "
                        "para shines o pasajes de lucimiento individual.")
    elif nivel_dt == "bajo":
        bullets.append(f"🎵 **Densidad rítmica baja** ({dt:.2f} vs. mediana de {sdt['p50']:.2f}): "
                        "conviene priorizar musicalidad y expresión corporal sobre velocidad de pies.")

    # Estructura (num_secciones) -> variedad coreográfica disponible
    ns = features["num_secciones"]
    sns = stats["num_secciones"]
    nivel_ns = _nivel(ns, sns["p25"], sns["p75"])
    if nivel_ns == "alto":
        bullets.append(f"🧩 **Estructura rica** (~{ns:.0f} secciones vs. mediana de {sns['p50']:.0f}): "
                        "suficientes cambios de sección para variar de nivel/energía/formación varias "
                        "veces en la misma pista.")
    elif nivel_ns == "bajo":
        bullets.append(f"🔁 **Estructura simple** (~{ns:.0f} secciones vs. mediana de {sns['p50']:.0f}): "
                        "mejor apostar por una idea coreográfica sólida y repetible que por muchos cambios.")

    # Energy + Valence combinados -> tono emocional de la coreografía
    e, v = features["energy"], features["valence"]
    se, sv = stats["energy"], stats["valence"]
    nivel_e, nivel_v = _nivel(e, se["p25"], se["p75"]), _nivel(v, sv["p25"], sv["p75"])
    if nivel_e == "alto" and nivel_v == "alto":
        bullets.append("🔥 **Energía y ánimo altos**: pista festiva de alto impacto — ideal para el "
                        "cierre de una rutina o un pasaje de máxima expresión grupal.")
    elif nivel_e == "bajo" and nivel_v == "bajo":
        bullets.append("🌙 **Energía y ánimo bajos**: mejor para una apertura o un pasaje romántico/"
                        "sensual, no para el clímax de la coreografía.")

    return "\n".join(f"* {b}" for b in bullets) if bullets else (
        "Los valores de esta pista están dentro de lo esperado en todos los aspectos evaluados."
    )


# ==========================================================================
# 2. BÚSQUEDA DIRECTA EN EL DATASET REAL
#    Si el título/artista que escribe el usuario coincide con una
#    canción real del dataset, usamos SUS features reales en vez de
#    estimar — esto es lo más fiel posible dado que no analizamos audio.
# ==========================================================================
def _normalizar(txt):
    txt = txt.lower().strip()
    txt = re.sub(r"[^\w\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt


def buscar_en_dataset(texto, df):
    """Busca coincidencia por track_name. El nombre de la canción DEBE
    aparecer en el texto de entrada (no basta con que coincida solo el
    artista, o cualquier canción de ese artista "ganaría" por error).
    Entre varios títulos que calcen, el artist_name se usa para
    desempatar. Devuelve la fila (pd.Series) más específica, o None."""
    texto_norm = _normalizar(texto)
    if not texto_norm:
        return None

    mejor_fila, mejor_score = None, 0
    for _, fila in df.iterrows():
        nombre_norm = _normalizar(fila["track_name"])
        artista_norm = _normalizar(fila["artist_name"])

        # Requisito obligatorio: el título de la canción debe estar en el texto.
        if not nombre_norm or nombre_norm not in texto_norm or len(nombre_norm) < 3:
            continue

        score = len(nombre_norm)
        if artista_norm and artista_norm in texto_norm:
            score += len(artista_norm)  # desempate cuando también coincide el artista

        if score > mejor_score:
            mejor_score = score
            mejor_fila = fila

    return mejor_fila


# ==========================================================================
# 3. PALABRAS CLAVE DE GÉNERO — construidas dinámicamente desde los
#    artistas reales del dataset + nombres genéricos de cada estilo.
#    Así el listado de "detección por palabra clave" queda anclado a
#    los datos reales en vez de a una lista manual desactualizada.
# ==========================================================================
PALABRAS_GENERO_BASE = {
    "Bachata": ["bachata"],
    "Salsa": ["salsa"],
    "Quebradita": ["quebradita", "quebradora", "banda", "technobanda", "sinaloense"],
}

# Artistas/keywords conocidos de Timba (subgénero DENTRO de Salsa en el
# dataset real) — solo se usan para elegir catálogo de "sabor" (canción/
# entrenamiento/vestuario), nunca para forzar una clase del modelo.
TIMBA_KEYWORDS = [
    "timba", "van van", "los van van", "alexander abreu", "issac delgado",
    "bamboleo", "manolito", "paulito fg", "havana d'primera", "cubana", "cesar pedroso",
]


def construir_palabras_genero(df):
    palabras = {g: list(v) for g, v in PALABRAS_GENERO_BASE.items()}
    for genero in df["genero_etiqueta"].unique():
        artistas = df[df["genero_etiqueta"] == genero]["artist_name"].dropna().unique()
        for a in artistas:
            a_norm = _normalizar(a)
            if len(a_norm) >= 4:  # evita ruido de nombres de 1-2 letras
                palabras.setdefault(genero, []).append(a_norm)
    return palabras


def detectar_genero_por_palabra_clave(texto_lower, palabras_genero):
    texto_norm = _normalizar(texto_lower)
    for genero, palabras in palabras_genero.items():
        if any(p in texto_norm for p in palabras):
            return genero
    return None


def es_variante_timba(texto):
    texto_norm = _normalizar(texto)
    return any(kw in texto_norm for kw in TIMBA_KEYWORDS)


# ==========================================================================
# 4. ENTRENAMIENTO DEL MODELO SOBRE EL DATASET REAL
# ==========================================================================
def entrenar_random_forest(df=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    if df is None:
        df = cargar_dataset()

    X = df[FEATURES].values
    y = df["genero_etiqueta"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    modelo = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=3,
        random_state=42, class_weight="balanced"
    )
    modelo.fit(X_train, y_train)
    acc = accuracy_score(y_test, modelo.predict(X_test))
    return modelo, acc


# ==========================================================================
# 5. CLASIFICADOR DE TIPO DE CONTENIDO (igual que antes: arregla
#    "cualquier palabra se detecta como canción" y distingue video
#    musical de video con música de fondo). No depende del dataset.
# ==========================================================================
PATRONES_NO_MUSICAL = [
    r"podcast", r"entrevista", r"rese[ñn]a", r"unboxing", r"vlog",
    r"tutorial", r"walkthrough", r"gameplay", r"receta", r"cocinando",
    r"noticiero", r"noticias", r"documental", r"serm[oó]n", r"pr[eé]dica",
    r"conferencia", r"charla\b", r"curso de", r"clase de(?! baile)",
    r"c[oó]mo hacer", r"qu[eé] es\b", r"resumen de\b(?! partido| pel[ií]cula)",
    r"an[aá]lisis de", r"podcast de", r"stream de", r"live reaction",
]
PATRONES_MUSICA_NO_DEDICADA = [
    r"tr[aá]iler", r"pel[ií]cula completa", r"cap[ií]tulo \d", r"episodio \d",
    r"highlights?", r"compilaci[oó]n de goles", r"resumen del? partido",
    r"resumen de pel[ií]cula", r"anuncio\b", r"comercial\b", r"spot publicitario",
    r"reacci[oó]n a", r"making of", r"detr[aá]s de c[aá]maras",
    r"challenge\b", r"meme\b", r"rutina de ejercicio", r"entrenamiento en casa",
    r"workout\b", r"torneo\b", r"esports?\b", r"video juego", r"videojuego",
    r"transmisi[oó]n en vivo(?! de baile| de concierto)",
]
_RE_NO_MUSICAL = re.compile("|".join(PATRONES_NO_MUSICAL), re.IGNORECASE)
_RE_MUSICA_NO_DEDICADA = re.compile("|".join(PATRONES_MUSICA_NO_DEDICADA), re.IGNORECASE)
_RE_TITULO_CANCION = re.compile(r"^\s*[\wÀ-ÿ' .]{2,60}\s*[-–—]\s*[\wÀ-ÿ' .]{2,60}\s*$")
_RE_FRASE_PREGUNTA = re.compile(
    r"^\s*(qu[eé]|c[oó]mo|cu[aá]l|por qu[eé]|cu[aá]ndo|d[oó]nde|qui[eé]n)\b", re.IGNORECASE
)


def clasificar_tipo_contenido(texto, palabras_genero, df=None):
    """Devuelve uno de: 'no_musical', 'musica_no_dedicada', 'cancion_reconocida',
    'cancion_ambigua', 'no_parece_cancion'. Si el texto coincide con una
    canción real del dataset, se prioriza sobre todo lo demás."""
    if df is not None and buscar_en_dataset(texto, df) is not None:
        return "cancion_reconocida"

    texto_lower = texto.lower()
    if _RE_NO_MUSICAL.search(texto_lower):
        return "no_musical"
    if _RE_MUSICA_NO_DEDICADA.search(texto_lower):
        return "musica_no_dedicada"

    genero = detectar_genero_por_palabra_clave(texto_lower, palabras_genero)
    if genero:
        return "cancion_reconocida"

    if _RE_FRASE_PREGUNTA.match(texto_lower):
        return "no_parece_cancion"
    if _RE_TITULO_CANCION.match(texto.strip()):
        return "cancion_ambigua"
    if texto.strip().startswith("http"):
        return "cancion_ambigua"

    return "no_parece_cancion"


# ==========================================================================
# 6. CATÁLOGOS AMPLIADOS
#    Salsa y Timba comparten catálogo base de Salsa, pero Timba tiene
#    contenido propio que se usa cuando es_variante_timba() detecta la
#    variante (ver punto 3) — así conservamos la riqueza de contenido
#    sin inventar una clase de modelo que los datos no tienen.
# ==========================================================================
CATALOGO_SUGERENCIAS = {
    "Salsa": [
        "Vivir Mi Vida - Marc Anthony", "Llorarás - Oscar D'León", "La Rebelión - Joe Arroyo",
        "Hacha y Machete - Héctor Lavoe", "Pedro Navaja - Rubén Blades", "Idilio - Willie Colón",
        "Perdóname - Gilberto Santa Rosa", "Mi Gente - Héctor Lavoe",
        "Todo Tiene Su Final - Héctor Lavoe", "Cali Pachanguero - Grupo Niche",
        "El Cantante - Héctor Lavoe", "Volver a Verte - Grupo Niche",
        "La Boda de Ella - Willie Colón", "Se Me Sigue Notando - Gilberto Santa Rosa",
    ],
    "Bachata": [
        "Propuesta Indecente - Romeo Santos", "Darte un Beso - Prince Royce",
        "Obsesión - Aventura", "Eres Mía - Romeo Santos", "Inmortal - Aventura",
        "Frío Frío - Juan Luis Guerra", "Loco - Zacarías Ferreira",
        "Volví a Nacer - Monchy & Alexandra", "Perdóname - Frank Reyes",
        "Culpables - Monchy & Alexandra", "Cuando Volverás - Hector Acosta",
    ],
    "Quebradita": [
        "La Culebra - Banda Machos", "El Pecador - Mi Banda El Mexicano",
        "No Bailes de Caballito - Mi Banda El Mexicano", "La Quebradora - Banda El Recodo",
        "El Sube y Baja - Banda Maguey", "La Chica Sexy - Banda Toro",
        "El Circo - Banda Machos", "Mi Gusto Es - Banda El Recodo",
        "Nubes Grises - Technobanda",
    ],
    "Timba": [
        "Te Pone la Cabeza Mala - Los Van Van", "La Sandunguita - Issac Delgado",
        "Esto te Pone Cabeza - Manolito Simonet", "Marilú - Havana D'Primera",
        "Soy Bacana - Manolito y su Trabuco", "El Consejo - Bamboleo",
        "La Bruja - Los Van Van", "Y Qué Tú Quieres Que Te Den - Los Van Van",
        "Llegó la Música Cubana - Paulito FG", "Amor Verdadero - Alexander Abreu",
    ],
}

CATALOGO_ENTRENAMIENTOS = {
    "Quebradita": [
        ("Agilidad", "Saltos pliométricos cortos (2x30s) y sentadillas explosivas.",
         "Desarrolla la potencia en el tren inferior para el rebote constante y la estabilidad en acrobacias."),
        ("Resistencia", "Circuito de burpees + jumping jacks, 4 rondas de 40s con 20s descanso.",
         "El tempo de la quebradita es exigente cardiovascularmente; hay que sostener la intensidad toda la canción."),
        ("Fuerza de piernas", "Zancadas con salto (walking lunges + jump), 3x12 por pierna.",
         "Soporta las cargadas y giros rápidos sin perder estabilidad en la rodilla."),
        ("Movilidad de cadera", "Rotaciones de cadera controladas + apertura/cierre en 90/90, 3x10.",
         "Facilita el 'quiebre' característico del género sin lesionar la zona lumbar."),
    ],
    "Bachata": [
        ("Agilidad", "Giros en eje sobre una sola pierna y movilidad pélvica aislada (3x10).",
         "Mejora el control del centro de gravedad y la transición fluida de caderas sin perder el tiempo fuerte."),
        ("Core", "Plancha con rotación de cadera, 3x12 por lado.",
         "El aislamiento de cadera de la bachata depende de un core estable que no comprometa la espalda baja."),
        ("Equilibrio", "Sentadilla a una pierna (pistol asistida) 3x8 por lado.",
         "Los giros y frenos bruscos en bachata sensual exigen buen control unipodal."),
        ("Flexibilidad", "Estiramiento dinámico de flexores de cadera, 2x30s por lado.",
         "Libera rango de movimiento para los aislamientos de cadera y las bajadas."),
    ],
    "Salsa": [
        ("Velocidad de pies", "Coordinación tipo *shines* a alta velocidad sobre metatarsos (4x45s).",
         "Incrementa la reacción de tobillos y la agilidad de los pasos libres."),
        ("Resistencia", "Intervalos de cuerda/saltos 30s on / 15s off x 8 rondas.",
         "El tempo alto de salsa exige base aeróbica para sostener el fraseo de 8 tiempos sin fatigarse."),
        ("Rotación", "Giros controlados sobre eje con freno (spins + spot), 3x8 por lado.",
         "Precisión en los turn patterns y control del mareo en secuencias largas."),
        ("Fuerza de tren superior", "Remo con banda + press ligero, 3x12.",
         "El marco de brazos firme en pareja depende de fuerza controlada, no rigidez."),
    ],
    "Timba": [
        ("Desplazamientos", "Desplazamientos laterales rápidos y quiebres de cintura con cambio de peso (3x1min).",
         "Facilita la adaptación a la polirritmia compleja y los cambios abruptos de dinámica del género."),
        ("Resistencia cardiovascular", "HIIT de 20s máximo esfuerzo / 40s activo x 10 rondas.",
         "La timba mezcla tramos de altísima energía; el cuerpo debe recuperar rápido entre cortes."),
        ("Coordinación", "Ejercicios de disociación torso-cadera frente al espejo, 3x1min.",
         "El despelote y los nudos de casino exigen independencia entre tren superior e inferior."),
        ("Explosividad", "Sentadilla con salto + giro de 180°, 3x10.",
         "Prepara el cuerpo para los cambios bruscos de dirección típicos del casino cubano."),
    ],
}

CATALOGO_VESTUARIO = {
    "Bachata": "👗 **Vestuario para Competencia de Bachata:**\n* **Mujeres:** Vestidos ajustados con flecos o aberturas que acentúen la cadera, pedrería brillante y zapatos con tacón delgado o botines flexibles.\n* **Hombres:** Pantalón entallado de vestir, camisas estilizadas (a veces abiertas o translúcidas) y zapatos de baile con suela flexible.",
    "Salsa": "💃 **Vestuario para Competencia de Salsa:**\n* **Mujeres:** Vestidos cortos con vuelo y capas para lucir los giros en competencia, flecos dinámicos y zapatos profesionales de suela de cuero con tacón cubano o aguja.\n* **Hombres:** Pantalón de corte latino, camisas camiseras o body de competencia, fajín opcional y zapatos de baile profesionales.",
    "Quebradita": "🤠 **Vestuario para Competencia de Quebradita:**\n* **Mujeres:** Vestidos vaqueros estilizados con falda amplia para los giros de rodeo, chalecos con flecos y botas vaqueras de suela corrida.\n* **Hombres:** Camisa vaquera de competencia con bordados, chaleco, pantalón vaquero resistente y botas de rodeo aptas para el impacto y soporte.",
    "Timba": "👟 **Vestuario para Competencia de Timba:**\n* **Mujeres:** Ropa urbana deportiva de alta costura, conjuntos de dos piezas con mallas y tenis flexibles o zapatillas de jazz para quiebres rápidos.\n* **Hombres:** Pantalón jogger de competencia estilizado, camisetas sin mangas o chaquetas deportivas abiertas y tenis de suela lisa.",
}

DETALLES_COREOGRAFICOS = {
    "Bachata": (8, 6, 7, "Compás 4/4. Acento en pulso 4 y 8 con tap/cadera.", "Conexión corporal y marco fluido."),
    "Quebradita": (10, 9, 8, "Compás 2/4. Acento constante en el bote.", "Acrobacias y giros veloces."),
    "Salsa": (9, 8, 9, "Fraseo 8 tiempos. Acentos en campana.", "Shines rápidos y giros en eje."),
    "Timba": (9, 9, 9, "Clave Cubana (2/3 o 3/2). Polirritmia compleja.", "Nudos de Casino y despelote."),
}


def elegir_sin_repetir(genero, catalogo, usados_dict):
    opciones = catalogo.get(genero, [])
    if not opciones:
        return None
    usados = usados_dict.setdefault(genero, [])
    disponibles = [c for c in opciones if c not in usados]
    if not disponibles:
        usados_dict[genero] = []
        disponibles = opciones
    elegido = random.choice(disponibles)
    usados_dict[genero].append(elegido)
    return elegido
