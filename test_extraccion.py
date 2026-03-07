"""
Test 1: Extracción de texto con pdfplumber
Uso: uv run python test_extraccion.py <nombre_archivo_en_cvs>
"""

import sys
import os
import time
import pdfplumber

CVS_DIR = "cvs"

def extraer_texto(ruta: str) -> dict:
    inicio = time.time()
    texto = ""
    paginas = 0

    with pdfplumber.open(ruta) as pdf:
        paginas = len(pdf.pages)
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texto += t + "\n"

    texto = texto.strip()
    tiempo = round(time.time() - inicio, 3)

    return {
        "texto": texto,
        "paginas": paginas,
        "caracteres": len(texto),
        "tiempo": tiempo,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Uso: uv run python test_extraccion.py <nombre_archivo_en_{CVS_DIR}>")
        print(f"Ejemplo: uv run python test_extraccion.py cv_prueba.pdf")
        sys.exit(1)

    nombre_archivo = sys.argv[1]
    
    # Intentar ruta directa o dentro de la carpeta CVS_DIR
    if os.path.exists(nombre_archivo):
        ruta = nombre_archivo
    else:
        ruta = os.path.join(CVS_DIR, nombre_archivo)
    
    if not os.path.exists(ruta):
        print(f"❌ Error: El archivo '{ruta}' no existe.")
        sys.exit(1)

    print(f"📄 Extrayendo: {ruta}")
    print("-" * 50)

    resultado = extraer_texto(ruta)

    print(f"Páginas:    {resultado['paginas']}")
    print(f"Caracteres: {resultado['caracteres']}")
    print(f"Tiempo:     {resultado['tiempo']}s")
    print()
    print("=" * 50)
    print("TEXTO EXTRAÍDO (primeros 1000 chars):")
    print("=" * 50)
    print(resultado["texto"][:1000])
    print()

    if resultado["caracteres"] < 100:
        print("⚠️  Poco texto extraído. El PDF puede ser escaneado (imagen).")
    else:
        print(f"✅ Extracción exitosa")
