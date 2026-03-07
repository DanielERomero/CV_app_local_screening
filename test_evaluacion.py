"""
Test 3: Evaluación de CV contra Job Description con LLM
Uso: uv run python test_evaluacion.py
     uv run python test_evaluacion.py --cv resultado.json --jd mi_jd.txt
     uv run python test_evaluacion.py --modelo phi4-mini:3.8b
"""

import sys
import json
import re
import time
import httpx

OLLAMA_URL = "http://localhost:11434"

PROMPT_EVALUAR = """Eres un reclutador senior de tecnología con 10 años de experiencia.

TAREA: Evalúa al candidato para el puesto descrito. Sé riguroso y objetivo.

REGLAS:
- Responde SOLAMENTE con el JSON.
- El score va de 0 a 100.
- No discrimines por género, edad, nacionalidad o raza.

FORMATO DE RESPUESTA:
{{
  "score": 75,
  "nivel": "Bueno",
  "motivos_contratacion": "Explicar en 2-3 oraciones por qué DEBERÍA ser contratado. Qué aporta al equipo.",
  "habilidades_faltantes": "Listar las habilidades o experiencias que le FALTAN para el puesto.",
  "justificacion": "Explicar en 3-4 oraciones POR QUÉ se asignó ese puntaje. Ser específico."
}}

Niveles: 0-20 No apto | 21-40 Bajo | 41-60 Regular | 61-80 Bueno | 81-100 Excelente

PUESTO:
---
{texto_jd}
---

CANDIDATO (datos estructurados):
---
{json_cv}
---

JSON:"""

JD_DEFAULT = """
Data Scientist Junior - Tech Peru SAC

Buscamos un Data Scientist Junior para nuestro equipo de analytics.

Responsabilidades:
- Análisis exploratorio de datos con Python (Pandas, NumPy, Matplotlib)
- Desarrollo de modelos de machine learning para predicción y clasificación
- Creación de dashboards interactivos en Power BI
- Consultas y manejo de bases de datos con SQL
- Despliegue de modelos en la nube (AWS o GCP)

Requisitos:
- Bachiller o Licenciado en Ingeniería, Estadística, Matemáticas o afines
- Mínimo 1 año de experiencia en análisis de datos o data science
- Python (Pandas, scikit-learn), SQL, Power BI
- Conocimientos de Machine Learning y estadística
- Deseable: experiencia con AWS o GCP
- Deseable: inglés intermedio
"""


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
    fragmento = re.sub(r',\s*([}\]])', r'\1', fragmento)
    return json.loads(fragmento)


def evaluar(json_cv: dict, texto_jd: str, modelo: str = "mistral:latest") -> dict:
    json_cv_str = json.dumps(json_cv, ensure_ascii=False, indent=2)

    prompt = PROMPT_EVALUAR.format(
        texto_jd=texto_jd[:2000],
        json_cv=json_cv_str[:3000],
    )

    print(f"📋 Evaluando con {modelo}...")
    print(f"   Prompt: {len(prompt)} chars")
    inicio = time.time()

    resp = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": modelo,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1024},
        },
        timeout=600,
    )

    datos_resp = resp.json()
    tiempo = round(time.time() - inicio, 2)
    texto_resp = datos_resp.get("response", "")

    print(f"   Tiempo: {tiempo}s")
    print(f"   Tokens prompt: {datos_resp.get('prompt_eval_count', '?')}")
    print(f"   Tokens respuesta: {datos_resp.get('eval_count', '?')}")
    print()

    try:
        resultado = extraer_json(texto_resp)
        resultado["score"] = max(0, min(100, int(resultado.get("score", 0))))
        return {"exito": True, "resultado": resultado, "tiempo": tiempo, "raw": texto_resp}
    except Exception as e:
        return {"exito": False, "error": str(e), "tiempo": tiempo, "raw": texto_resp}


if __name__ == "__main__":
    modelo = "mistral:latest"
    ruta_cv_json = "test_estructuracion_resultado.json"
    ruta_jd = None
    texto_jd = JD_DEFAULT

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--modelo" and i + 1 < len(args):
            modelo = args[i + 1]
            i += 2
        elif args[i] == "--cv" and i + 1 < len(args):
            ruta_cv_json = args[i + 1]
            i += 2
        elif args[i] == "--jd" and i + 1 < len(args):
            ruta_jd = args[i + 1]
            i += 2
        else:
            i += 1

    # Cargar CV JSON
    print(f"📂 Cargando CV de: {ruta_cv_json}")
    try:
        with open(ruta_cv_json, "r", encoding="utf-8") as f:
            json_cv = json.load(f)
        print(f"   Candidato: {json_cv.get('nombre_completo', '?')}")
    except FileNotFoundError:
        print(f"❌ No se encontró {ruta_cv_json}")
        print(f"   Primero ejecuta: uv run python test_estructuracion.py cvs/tu_cv.pdf")
        sys.exit(1)

    # Cargar JD
    if ruta_jd:
        print(f"📂 Cargando JD de: {ruta_jd}")
        with open(ruta_jd, "r", encoding="utf-8") as f:
            texto_jd = f.read()
    else:
        print(f"📋 Usando JD de prueba: Data Scientist Junior")

    print()

    # Evaluar
    resultado = evaluar(json_cv, texto_jd, modelo)

    print("=" * 55)

    if resultado["exito"]:
        r = resultado["resultado"]
        score = r["score"]

        if score >= 81: barra = "🟢"
        elif score >= 61: barra = "🔵"
        elif score >= 41: barra = "🟡"
        else: barra = "🔴"

        print(f"  {barra} SCORE: {score}/100 — {r.get('nivel', '?')}")
        print()
        print(f"  ✅ MOTIVOS PARA CONTRATAR:")
        print(f"  {r.get('motivos_contratacion', 'N/A')}")
        print()
        print(f"  ❌ HABILIDADES FALTANTES:")
        print(f"  {r.get('habilidades_faltantes', 'N/A')}")
        print()
        print(f"  📝 JUSTIFICACIÓN DEL PUNTAJE:")
        print(f"  {r.get('justificacion', 'N/A')}")
        print()
        print(f"  ⏱️  Tiempo: {resultado['tiempo']}s")

        salida = "test_evaluacion_resultado.json"
        with open(salida, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"  💾 Guardado en: {salida}")

    else:
        print("❌ FALLÓ")
        print(f"Error: {resultado['error']}")
        print(f"Respuesta raw:")
        print(resultado["raw"][:500])
