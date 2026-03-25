import streamlit as st
import sys
import os
import json
import re
import time
from pathlib import Path

import httpx
import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.almacenamiento import (
    SupabaseClient,
    guardar_bronze_cv, guardar_bronze_jd,
    guardar_silver_candidato, guardar_silver_jd,
    guardar_gold_evaluacion,
    obtener_candidatos, obtener_jobs, obtener_ranking,
)

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="CV Screening - Tesis DS",
    page_icon="📄",
    layout="wide",
)

OLLAMA_URL = "http://localhost:11434"
CARPETA_TEMP = "temp_cvs"
os.makedirs(CARPETA_TEMP, exist_ok=True)

# ============================================================
# FUNCIONES CORE
# ============================================================

def verificar_ollama():
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False

def listar_modelos():
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except:
        return []

def extraer_texto_pdf(ruta: str) -> dict:
    texto = ""
    paginas = 0
    with pdfplumber.open(ruta) as pdf:
        paginas = len(pdf.pages)
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texto += t + "\n"
    return {"texto": texto.strip(), "paginas": paginas, "caracteres": len(texto.strip())}

def llamar_llm(prompt: str, modelo: str) -> dict:
    resp = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": modelo,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 4096},
        },
        timeout=600,
    )
    return resp.json()

def extraer_json(texto: str) -> dict:
    texto = texto.strip()
    bloque = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', texto, re.DOTALL)
    if bloque:
        texto = bloque.group(1)
    
    ini = texto.find('{')
    fin = texto.rfind('}')
    
    if ini == -1 or fin == -1:
        raise ValueError("No se encontró JSON en la respuesta")
        
    fragmento = texto[ini:fin + 1]
    # Limpiamos posibles comas extra al final de estructuras, causadas por fallos del LLM
    fragmento = re.sub(r',\s*([}\]])', r'\1', fragmento)
    return json.loads(fragmento)


# ============================================================
# PROMPT ESTRUCTURACIÓN (Fase 2)
# ============================================================

PROMPT_ESTRUCTURAR =  """Eres un extractor especializado de información de CVs.
Tu única función es extraer datos y devolverlos en el
formato indicado. No evalúes, no opines, no inferas
más allá de lo explícitamente escrito.

CONTEXTO:
Se te proporciona el texto completo de un CV profesional

TAREA:
Extrae la información del CV y devuélvela en formato JSON validado.
Si un campo no está presente, devuelve null (no uses arrays vacíos para campos de texto).
No inventes ninguna información ausente.

RESTRICCIONES IMPORTANTES:
1. RESPUESTA ESTRICTA: Tu respuesta debe ser SOLO un objeto JSON válido, comenzando con {{ y terminando con }}. Nada de texto introductorio.
2. El JSON DEBE usar EXACTAMENTE las siguientes llaves (no inventes nuevas ni cambies los nombres):
   - nombre_completo (string o null)
   - email (string o null)
   - telefono (string o null)
   - ubicacion (string o null)
   - linkedin (string o null)
   - resumen_profesional (string o null)
   - experiencia_laboral (array de objetos con llaves: empresa, cargo, duracion, descripcion)
   - educacion (array de objetos con llaves: institucion, titulo, anio)
   - habilidades_tecnicas (array de strings)
   - habilidades_blandas (array de strings)
   - idiomas (array de strings)

3. Escapa correctamente las comillas dobles internas (usa \\") o reemplázalas por comillas simples.
4. Reemplaza los saltos de línea dentro de los textos por espacios o \\n.

EJEMPLO PERFECTO:
{{
  "nombre_completo": "María García",
  "email": "maria.garcia@email.com",
  "telefono": "+51 987 654 321",
  "ubicacion": "Lima, Perú",
  "linkedin": "linkedin.com/in/usuario",
  "resumen_profesional": "Profesional en analítica de datos...",
  "experiencia_laboral": [
    {{
      "empresa": "Empresa SAC",
      "cargo": "Data Analyst",
      "duracion": "2021 - 2023",
      "descripcion": "Desarrollo de modelos predictivos..."
    }}
  ],
  "educacion": [
    {{
      "institucion": "UNMSM",
      "titulo": "Computación Científica",
      "anio": "2020"
    }}
  ],
  "habilidades_tecnicas": ["Python", "Machine Learning"],
  "habilidades_blandas": ["Liderazgo"],
  "idiomas": ["Inglés intermedio"]
}}

IMPORTANTE EXTRA: El nombre completo, email y teléfono DEBEN buscarse prioritariamente al INICIO del texto del CV. Esfuérzate al máximo en extraerlos y no los omitas.

CV:
---
{texto_cv}
---
"""



# ============================================================
# PROMPT EVALUACIÓN (Fase 4 - una sola señal LLM)
# ============================================================

PROMPT_EVALUAR = """Eres un evaluador experto en selección de talento con
15 años de experiencia en RRHH técnico. Evalúas candidatos
de forma objetiva, basándote exclusivamente en criterios
profesionales relevantes para el puesto

CONTEXTO:
Se te proporciona:
1. Un perfil de candidato estructurado (extraído de su CV)
2. Una descripción de puesto estructurada (Job Description)

TAREA:
Evalúa el ajuste del candidato al puesto según estas
5 dimensiones. El score total es de 0 a 100:

- skills_tecnicos
  ¿Qué porcentaje de los skills requeridos posee el candidato?
  ¿Tiene skills adicionales valiosos no requeridos?

- experiencia
  ¿Los años y tipo de experiencia coinciden con lo solicitado?
  ¿Las responsabilidades previas son relevantes para el rol?

- educacion 
  ¿La formación académica cumple los requisitos mínimos?
  Considera equivalencia por experiencia si aplica.

- idiomas 
  ¿Cumple los requisitos de idiomas del puesto?

- fit_general 
  Considerando el perfil completo, ¿hay coherencia en
  la trayectoria? ¿El candidato está en el nivel correcto
  para el puesto (ni sobreclasificado ni sub-clasificado)?

Redondea a 1 decimal. Rango resultante: 0-100.

RESTRICCIONES:
- Evalúa solo lo que está en los datos. Si un campo es
  null, asume ausencia, no inventes.
- Las justificaciones deben ser específicas, no genéricas.
  Mal: "Tiene buena experiencia técnica"
  Bien: "3 de 5 skills requeridos presentes; ausencia de
        experiencia en Kubernetes es una brecha crítica"
- La recomendación debe ser consistente con el score:
  0-40   → descartar
  41-60  → considerar
  61-80  → entrevistar
  81-100 → prioridad

FORMATO DE RESPUESTA:
{{
  "score": 75,
  "nivel": "Bueno",
  "motivos_contratacion": "Explicar en 2-3 oraciones por qué DEBERÍA ser contratado. Qué aporta al equipo.",
  "habilidades_faltantes": "Listar las habilidades o experiencias que le FALTAN para el puesto.",
  "justificacion": "Explicar en 3-4 oraciones POR QUÉ se asignó ese puntaje. Ser específico."
}}

Niveles: 0-40 Descartar | 41-60 Considerar | 61-80 Entrevistar | 81-100 Prioridad

PUESTO:
---
{texto_jd}
---

CANDIDATO (datos estructurados):
---
{json_cv}
---

JSON:"""

PROMPT_VERSION = "v1.0"

# ============================================================
# SESSION STATE
# ============================================================

if "candidatos" not in st.session_state:
    st.session_state.candidatos = []
if "job_description" not in st.session_state:
    st.session_state.job_description = None
if "evaluaciones" not in st.session_state:
    st.session_state.evaluaciones = []
if "db" not in st.session_state:
    st.session_state.db = None


# ============================================================
# MAIN UI TITULO
# ============================================================
st.title("CV SCREENING SYSTEM")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("⚙️ Configuración")

    st.divider()
    st.subheader("🤖 Ollama")
    ollama_ok = verificar_ollama()
    if ollama_ok:
        modelos = listar_modelos()
        st.success(f"Conectado ({len(modelos)} modelos)")
        modelo_llm = st.selectbox("Modelo", modelos,
            index=modelos.index("mistral:latest") if "mistral:latest" in modelos else 0)
    else:
        st.error("No disponible — ejecuta `ollama serve`")
        modelo_llm = "mistral:latest"

    # st.divider()
    # st.subheader("🗄️ Supabase")
    # url_sb = st.text_input("URL", value=os.environ.get("SUPABASE_URL", ""))
    # key_sb = st.text_input("Key", value=os.environ.get("SUPABASE_KEY", ""), type="password")

    url_sb = os.environ.get("SUPABASE_URL", "")
    key_sb = os.environ.get("SUPABASE_KEY", "")

    if url_sb and key_sb:
        os.environ["SUPABASE_URL"] = url_sb
        os.environ["SUPABASE_KEY"] = key_sb
        try:
            if st.session_state.db is None:
                st.session_state.db = SupabaseClient()
            # st.success("Conectado")
        except Exception as e:
            # st.error(str(e))
            st.session_state.db = None
    # else:
    #     st.caption("Opcional: para persistir resultados")

    st.divider()
    st.subheader("📈 Estado")
    st.metric("CVs cargados", len(st.session_state.candidatos))
    st.metric("JD cargado", "✅" if st.session_state.job_description else "❌")
    st.metric("Evaluaciones", len(st.session_state.evaluaciones))


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 1. Cargar CVs",
    "📋 2. Job Description",
    "🏆 3. Evaluación",
    "📊 4. Resultados",
])

# ============================================================
# TAB 1: CARGAR CVs
# ============================================================

with tab1:
    st.header("📄 Cargar CVs de Candidatos")

    archivos = st.file_uploader("Sube PDFs", type=["pdf"],
                                 accept_multiple_files=True, key="up_cvs")

    if archivos and st.button("🚀 Procesar CVs", type="primary"):
        if not ollama_ok:
            st.error("Ollama no está disponible. Inicia con `ollama serve`.")
        else:
            barra = st.progress(0)

            for i, archivo in enumerate(archivos):
                nombre = archivo.name
                barra.progress(i / len(archivos), text=f"{nombre} ({i+1}/{len(archivos)})")

                if any(c["archivo"] == nombre for c in st.session_state.candidatos):
                    st.warning(f"⏭️ {nombre} ya cargado")
                    continue

                ruta = os.path.join(CARPETA_TEMP, nombre)
                with open(ruta, "wb") as f:
                    f.write(archivo.getbuffer())

                # FASE 1: pdfplumber
                with st.spinner(f"📄 Extrayendo texto de {nombre}..."):
                    try:
                        ext = extraer_texto_pdf(ruta)
                    except Exception as e:
                        st.error(f"❌ {nombre}: {e}")
                        continue

                if not ext["texto"]:
                    st.error(f"❌ {nombre}: sin texto")
                    continue

                # FASE 2: LLM → JSON
                with st.spinner(f"🤖 Estructurando {nombre} con {modelo_llm}... (~1-3 min)"):
                    inicio = time.time()
                    prompt = PROMPT_ESTRUCTURAR.format(texto_cv=ext["texto"][:8000])
                    raw_response = ""
                    try:
                        resp = llamar_llm(prompt, modelo_llm)
                        raw_response = resp.get("response", "")
                        datos_cv = extraer_json(raw_response)
                        tiempo_llm = round(time.time() - inicio, 2)
                    except Exception as e:
                        st.error(f"❌ {nombre}: LLM falló — {e}")
                        with st.expander("Ver respuesta cruda del LLM (Debugging)"):
                            st.text(raw_response)
                        datos_cv = {"nombre_completo": nombre.replace(".pdf", "")}
                        tiempo_llm = 0

                # Guardar en session
                candidato = {
                    "archivo": nombre,
                    "texto": ext["texto"],
                    "datos": datos_cv,
                    "paginas": ext["paginas"],
                    "caracteres": ext["caracteres"],
                    "tiempo_llm": tiempo_llm,
                    "modelo": modelo_llm,
                }

                # Guardar en Supabase (Bronze + Silver)
                db = st.session_state.db
                if db:
                    try:
                        bid = guardar_bronze_cv(db, nombre, ext["texto"],
                                                ext["paginas"], ext["caracteres"])
                        sid = guardar_silver_candidato(db, bid, datos_cv,
                                                       modelo_llm, tiempo_llm)
                        candidato["bronze_id"] = bid
                        candidato["silver_id"] = sid
                    except Exception as e:
                        st.warning(f"⚠️ Supabase: {e}")

                st.session_state.candidatos.append(candidato)

                nom = datos_cv.get("nombre_completo", nombre)
                n_sk = len(datos_cv.get("habilidades_tecnicas", []))
                n_ex = len(datos_cv.get("experiencia_laboral", []))
                st.success(f"✅ **{nom}** | {n_sk} skills | {n_ex} experiencias | {tiempo_llm}s")

            barra.progress(1.0, text="✅ Completado")

    # Mostrar candidatos
    if st.session_state.candidatos:
        st.divider()
        st.subheader(f"👥 Candidatos ({len(st.session_state.candidatos)})")

        for idx, c in enumerate(st.session_state.candidatos):
            d = c["datos"]
            nom = d.get("nombre_completo", c["archivo"])
            skills = d.get("habilidades_tecnicas", [])

            with st.expander(f"**{nom}** — {len(skills)} skills"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"📧 {d.get('email', 'N/A')}")
                    st.write(f"📍 {d.get('ubicacion', 'N/A')}")
                    if d.get("resumen_profesional"):
                        st.write(f"📝 {d['resumen_profesional'][:200]}")
                with c2:
                    if skills:
                        st.write(f"🔧 **Skills:** {', '.join(skills[:10])}")
                    exp = d.get("experiencia_laboral", [])
                    for e in exp[:3]:
                        st.write(f"💼 {e.get('cargo','?')} @ {e.get('empresa','?')}")
                    edu = d.get("educacion", [])
                    for e in edu[:2]:
                        st.write(f"🎓 {e.get('titulo','?')} — {e.get('institucion','?')}")

                st.caption(f"{c['caracteres']} chars | {c['paginas']} pags | {c['tiempo_llm']}s | {c['modelo']}")

                if st.button("🗑️ Eliminar", key=f"del_{idx}"):
                    st.session_state.candidatos.pop(idx)
                    st.rerun()


# ============================================================
# TAB 2: JOB DESCRIPTION
# ============================================================

with tab2:
    st.header("📋 Job Description")

    texto_jd = st.text_area(
        "Pega el texto del puesto",
        height=300,
        placeholder="Data Scientist Junior - Tech Peru SAC\n\nBuscamos...\n\nRequisitos:\n- Python, SQL...",
        value=st.session_state.job_description["texto"] if st.session_state.job_description else "",
    )

    titulo_jd = st.text_input("Título del puesto (para identificarlo)",
        value=st.session_state.job_description["titulo"] if st.session_state.job_description else "")

    if st.button("💾 Guardar JD", type="primary"):
        if not texto_jd.strip():
            st.error("El texto no puede estar vacío.")
        else:
            jd = {
                "texto": texto_jd.strip(),
                "titulo": titulo_jd or "Puesto",
            }

            db = st.session_state.db
            if db:
                try:
                    bid = guardar_bronze_jd(db, texto_jd.strip())
                    sid = guardar_silver_jd(db, bid, titulo=titulo_jd or "Puesto")
                    jd["bronze_id"] = bid
                    jd["silver_id"] = sid
                except Exception as e:
                    st.warning(f"⚠️ Supabase: {e}")

            st.session_state.job_description = jd
            st.success(f"✅ JD guardado: **{jd['titulo']}**")

    if st.session_state.job_description:
        st.divider()
        jd = st.session_state.job_description
        st.info(f"📌 JD activo: **{jd['titulo']}**")
        st.text(jd["texto"][:400] + ("..." if len(jd["texto"]) > 400 else ""))


# ============================================================
# TAB 3: EVALUACIÓN
# ============================================================

with tab3:
    st.header("🏆 Evaluación de Candidatos")

    n_cands = len(st.session_state.candidatos)
    jd = st.session_state.job_description

    if n_cands == 0:
        st.warning("⬅️ Primero carga CVs en la pestaña 1.")
    elif jd is None:
        st.warning("⬅️ Primero carga un Job Description en la pestaña 2.")
    elif not ollama_ok:
        st.error("Ollama no está disponible.")
    else:
        st.success(f"**{n_cands} candidato(s)** listos para evaluar contra **{jd['titulo']}**")
        st.caption(f"Modelo: {modelo_llm} | ~2-4 min por candidato")

        if st.button("🚀 Evaluar Todos", type="primary"):
            evaluaciones = []
            barra = st.progress(0)

            for i, cand in enumerate(st.session_state.candidatos):
                nom = cand["datos"].get("nombre_completo", cand["archivo"])
                barra.progress(i / n_cands, text=f"Evaluando {nom}... ({i+1}/{n_cands})")

                json_cv_str = json.dumps(cand["datos"], ensure_ascii=False, indent=2)

                prompt = PROMPT_EVALUAR.format(
                    texto_jd=jd["texto"][:2000],
                    json_cv=json_cv_str[:3000],
                )

                with st.spinner(f"📋 Evaluando {nom}..."):
                    inicio = time.time()
                    try:
                        resp = llamar_llm(prompt, modelo_llm)
                        resultado = extraer_json(resp.get("response", ""))
                        tiempo = round(time.time() - inicio, 2)
                    except Exception as e:
                        st.error(f"❌ {nom}: {e}")
                        continue

                score = max(0, min(100, int(resultado.get("score", 0))))
                resultado["score"] = score

                eval_entry = {
                    "candidato": nom,
                    "archivo": cand["archivo"],
                    "score": score,
                    "nivel": resultado.get("nivel", ""),
                    "motivos_contratacion": resultado.get("motivos_contratacion", ""),
                    "habilidades_faltantes": resultado.get("habilidades_faltantes", ""),
                    "justificacion": resultado.get("justificacion", ""),
                    "tiempo": tiempo,
                    "modelo": modelo_llm,
                    "prompt_version": PROMPT_VERSION,
                    "datos_cv": cand["datos"],
                }

                # Gold → Supabase
                db = st.session_state.db
                if db and cand.get("silver_id") and jd.get("silver_id"):
                    try:
                        guardar_gold_evaluacion(
                            db, cand["silver_id"], jd["silver_id"],
                            resultado, modelo_llm, tiempo, PROMPT_VERSION
                        )
                    except Exception as e:
                        st.warning(f"⚠️ Supabase gold: {e}")

                evaluaciones.append(eval_entry)
                st.success(f"✅ **{nom}**: {score}/100 — {resultado.get('nivel', '')}")

            evaluaciones.sort(key=lambda x: x["score"], reverse=True)
            st.session_state.evaluaciones = evaluaciones
            barra.progress(1.0, text="✅ Evaluación completada")
            st.balloons()


# ============================================================
# TAB 4: RESULTADOS
# ============================================================

with tab4:
    st.header("📊 Resultados")

    evaluaciones = st.session_state.evaluaciones

    if not evaluaciones:
        st.info("Aún no hay evaluaciones. Ejecuta la evaluación en la pestaña 3.")
    else:
        jd = st.session_state.job_description
        st.subheader(f"Puesto: {jd['titulo']}")

        # Ranking
        for i, ev in enumerate(evaluaciones):
            pos = i + 1
            if pos == 1: emoji = "🥇"
            elif pos == 2: emoji = "🥈"
            elif pos == 3: emoji = "🥉"
            else: emoji = f"#{pos}"

            score = ev["score"]
            if score >= 81: color = "🟢"
            elif score >= 61: color = "🔵"
            elif score >= 41: color = "🟡"
            else: color = "🔴"

            with st.container(border=True):
                col_r, col_s, col_info = st.columns([1, 1, 5])

                with col_r:
                    st.markdown(f"### {emoji}")
                with col_s:
                    st.metric("Score", f"{score}/100")
                    st.caption(f"{color} {ev['nivel']}")
                with col_info:
                    st.markdown(f"**{ev['candidato']}**")

                    st.markdown(f"**✅ Por qué contratar:**")
                    st.write(ev["motivos_contratacion"])

                    st.markdown(f"**❌ Habilidades faltantes:**")
                    st.write(ev["habilidades_faltantes"])

                    st.markdown(f"**📝 Justificación del puntaje:**")
                    st.write(ev["justificacion"])

                    st.caption(
                        f"⏱️ {ev['tiempo']}s  |  "
                        f"🤖 {ev.get('modelo', 'N/A')}  |  "
                        f"📝 prompt {ev.get('prompt_version', 'N/A')}"
                    )

        # Detalle expandible
        st.divider()
        st.subheader("🔍 Detalle de CV")

        nombres = [ev["candidato"] for ev in evaluaciones]
        sel = st.selectbox("Selecciona candidato", nombres)

        if sel:
            ev = next(e for e in evaluaciones if e["candidato"] == sel)
            d = ev["datos_cv"]

            c1, c2 = st.columns(2)
            with c1:
                st.write(f"📧 {d.get('email', 'N/A')}")
                st.write(f"📱 {d.get('telefono', 'N/A')}")
                st.write(f"📍 {d.get('ubicacion', 'N/A')}")
                st.write(f"📝 {d.get('resumen_profesional', 'N/A')}")
            with c2:
                skills = d.get("habilidades_tecnicas", [])
                if skills:
                    st.write(f"**🔧 Skills:** {', '.join(skills)}")
                blandas = d.get("habilidades_blandas", [])
                if blandas:
                    st.write(f"**🤝 Blandas:** {', '.join(blandas)}")
                idiomas = d.get("idiomas", [])
                if idiomas:
                    st.write(f"**🌐 Idiomas:** {', '.join(idiomas)}")

            with st.expander("📄 Ver JSON Estructurado Completo"):
                st.json(d)

        # Exportar
        st.divider()
        export = []
        for i, ev in enumerate(evaluaciones):
            export.append({
                "posicion": i + 1,
                "candidato": ev["candidato"],
                "score": ev["score"],
                "nivel": ev["nivel"],
                "motivos_contratacion": ev["motivos_contratacion"],
                "habilidades_faltantes": ev["habilidades_faltantes"],
                "justificacion": ev["justificacion"],
                "modelo": ev.get("modelo", ""),
                "prompt_version": ev.get("prompt_version", ""),
            })

        st.download_button(
            "📥 Descargar ranking (JSON)",
            data=json.dumps(export, ensure_ascii=False, indent=2),
            file_name="ranking_candidatos.json",
            mime="application/json",
        )
