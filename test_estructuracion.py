"""
Test 2: Estructuración de CV con LLM
Uso: uv run python test_estructuracion.py cvs/mi_cv.pdf
     uv run python test_estructuracion.py cvs/mi_cv.pdf --modelo phi4-mini:3.8b
     uv run python test_estructuracion.py --texto "Joel Taquía..."  (texto directo)
"""

import sys
import json
import re
import time
import pdfplumber
import httpx

OLLAMA_URL = "http://localhost:11434"

PROMPT = """Eres un extractor especializado de información de CVs.
Tu única función es extraer datos y devolverlos en el
formato indicado. No evalúes, no opines, no inferas
más allá de lo explícitamente escrito.

CONTEXTO:
Se te proporciona el texto completo de un CV profesional.

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

CV:
---
{texto_cv}
---
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
    # Limpiamos posibles comas extra generadas por LLMs al final de diccionarios
    fragmento = re.sub(r',\s*([}\]])', r'\1', fragmento)
    return json.loads(fragmento)


def estructurar(texto_cv: str, modelo: str = "mistral:latest") -> dict:
    prompt = PROMPT.format(texto_cv=texto_cv[:4000])

    print(f"🤖 Enviando a {modelo}...")
    print(f"   Prompt: {len(prompt)} chars")
    inicio = time.time()

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

    datos_resp = resp.json()
    tiempo = round(time.time() - inicio, 2)
    texto_resp = datos_resp.get("response", "")

    print(f"   Tiempo: {tiempo}s")
    print(f"   Tokens prompt: {datos_resp.get('prompt_eval_count', '?')}")
    print(f"   Tokens respuesta: {datos_resp.get('eval_count', '?')}")
    print()

    try:
        datos = extraer_json(texto_resp)
        return {"exito": True, "datos": datos, "tiempo": tiempo, "raw": texto_resp}
    except Exception as e:
        return {"exito": False, "error": str(e), "tiempo": tiempo, "raw": texto_resp}


import os

def listar_modelos():
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        print(f"❌ Error al conectar con Ollama: {e}")
        return []

if __name__ == "__main__":
    modelo = None
    texto = None
    ruta_pdf = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--modelo" and i + 1 < len(args):
            modelo = args[i + 1]
            i += 2
        elif args[i] == "--texto" and i + 1 < len(args):
            texto = args[i + 1]
            i += 2
        elif args[i].endswith(".pdf"):
            ruta_pdf = args[i]
            i += 1
        else:
            i += 1

    if not modelo:
        print("Obteniendo modelos disponibles en Ollama...")
        modelos_disp = listar_modelos()
        if not modelos_disp:
            print("⚠️ No hay modelos disponibles o no se pudo conectar. Se usará 'mistral:latest'.")
            modelo = "mistral:latest"
        else:
            print("\n🤖 Modelos disponibles:")
            for idx, m in enumerate(modelos_disp):
                print(f"  {idx + 1}. {m}")
            
            while True:
                seleccion = input(f"\nSelecciona el modelo (1-{len(modelos_disp)}) o presiona Enter para usar '{modelos_disp[0]}': ")
                if not seleccion.strip():
                    modelo = modelos_disp[0]
                    break
                elif seleccion.isdigit():
                    idx_sel = int(seleccion) - 1
                    if 0 <= idx_sel < len(modelos_disp):
                        modelo = modelos_disp[idx_sel]
                        break
                print("❌ Selección inválida. Intenta de nuevo.")
        print(f"✅ Modelo seleccionado: {modelo}\n")

    if not texto and not ruta_pdf:
        print("Buscando PDF automáticamente en la carpeta 'cvs'...")
        ruta_carpeta = "cvs"
        if os.path.isdir(ruta_carpeta):
            archivos_pdf = [f for f in os.listdir(ruta_carpeta) if f.endswith('.pdf')]
            if archivos_pdf:
                ruta_pdf = os.path.join(ruta_carpeta, archivos_pdf[0])
                print(f"✅ Se encontró: {ruta_pdf}")
            else:
                print(f"❌ No se encontró ningún archivo PDF en la carpeta '{ruta_carpeta}'")
                sys.exit(1)
        else:
            print(f"❌ La carpeta '{ruta_carpeta}' no existe.")
            sys.exit(1)

    if ruta_pdf:
        print(f"📄 Extrayendo texto de: {ruta_pdf}")
        with pdfplumber.open(ruta_pdf) as pdf:
            texto = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texto += t + "\n"
        texto = texto.strip()
        print(f"   {len(texto)} caracteres extraídos")
        print()

    resultado = estructurar(texto, modelo)

    print("=" * 55)

    if resultado["exito"]:
        datos = resultado["datos"]
        print("✅ ESTRUCTURACIÓN EXITOSA")
        print("-" * 55)
        print(f"Nombre:    {datos.get('nombre_completo', '?')}")
        print(f"Email:     {datos.get('email', '?')}")
        print(f"Teléfono:  {datos.get('telefono', '?')}")
        print(f"Ubicación: {datos.get('ubicacion', '?')}")
        print()

        skills = datos.get("habilidades_tecnicas", [])
        print(f"Skills ({len(skills)}): {', '.join(skills[:10])}")

        exp = datos.get("experiencia_laboral", [])
        print(f"Experiencia ({len(exp)}):")
        for e in exp[:4]:
            print(f"  - {e.get('cargo','?')} @ {e.get('empresa','?')} ({e.get('duracion','?')})")

        edu = datos.get("educacion", [])
        print(f"Educación ({len(edu)}):")
        for e in edu:
            print(f"  - {e.get('titulo','?')} — {e.get('institucion','?')}")

        print()
        nombre_modelo_seguro = modelo.replace(':', '-')
        salida = f"test_estructuracion_resultado_{nombre_modelo_seguro}.json"
        with open(salida, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON guardado en: {salida}")

    else:
        print("❌ FALLÓ")
        print(f"Error: {resultado['error']}")
        print(f"Respuesta raw (primeros 500 chars):")
        print(resultado["raw"][:500])
