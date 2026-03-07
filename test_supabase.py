"""
Test 4: Guardado en Supabase (Arquitectura Medallion)
Uso: uv run python test_supabase.py
     uv run python test_supabase.py --leer  (solo consultar datos existentes)

Requiere .env con SUPABASE_URL y SUPABASE_KEY
Y haber ejecutado crear_tablas_medallion.sql en el SQL Editor
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.almacenamiento import (
    SupabaseClient,
    guardar_bronze_cv, guardar_bronze_jd,
    guardar_silver_candidato, guardar_silver_jd,
    guardar_gold_evaluacion,
    obtener_candidatos, obtener_jobs, obtener_ranking,
)


def test_conexion():
    print("🔍 Conectando a Supabase...")
    try:
        db = SupabaseClient()
        db.consultar("bronze_cvs", limite=1)
        print("✅ Conexión exitosa")
        return db
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Verifica .env y que las tablas estén creadas")
        return None


def test_escritura(db: SupabaseClient):
    print()
    print("=" * 50)
    print("🥉 BRONZE - Guardando datos crudos")
    print("-" * 50)

    bronze_cv_id = guardar_bronze_cv(
        db, "test_cv.pdf", "Joel Taquía Ayllón... texto de prueba", 2, 500
    )
    print(f"   bronze_cvs → {bronze_cv_id}")

    bronze_jd_id = guardar_bronze_jd(
        db, "Data Scientist Junior... requisitos...", "test"
    )
    print(f"   bronze_job_descriptions → {bronze_jd_id}")

    print()
    print("🥈 SILVER - Guardando datos estructurados")
    print("-" * 50)

    datos_cv = {
        "nombre_completo": "Joel Taquía Ayllón (TEST)",
        "email": "test@test.com",
        "habilidades_tecnicas": ["Python", "SQL", "AWS"],
        "experiencia_laboral": [
            {"empresa": "BWIT", "cargo": "Data Scientist", "duracion": "2019-2020"}
        ],
        "educacion": [
            {"institucion": "UNMSM", "titulo": "Lic. Computación Científica"}
        ],
    }

    silver_cv_id = guardar_silver_candidato(
        db, bronze_cv_id, datos_cv, "mistral:latest", 196.5
    )
    print(f"   silver_candidatos → {silver_cv_id}")

    silver_jd_id = guardar_silver_jd(
        db, bronze_jd_id, "Data Scientist Junior (TEST)", "Tech Peru"
    )
    print(f"   silver_job_descriptions → {silver_jd_id}")

    print()
    print("🥇 GOLD - Guardando evaluación")
    print("-" * 50)

    resultado_eval = {
        "score": 72,
        "nivel": "Bueno",
        "motivos_contratacion": "Tiene experiencia en data science y Python.",
        "habilidades_faltantes": "Power BI, experiencia con ML en producción.",
        "justificacion": "Score de 72 porque tiene buena base técnica pero falta experiencia.",
    }

    gold_id = guardar_gold_evaluacion(
        db, silver_cv_id, silver_jd_id, resultado_eval, "mistral:latest", 180.5
    )
    print(f"   gold_evaluaciones → {gold_id}")

    print()
    print("✅ Datos de prueba guardados en las 3 capas")
    return silver_jd_id


def test_lectura(db: SupabaseClient, job_id: str = None):
    print()
    print("=" * 50)
    print("📋 LECTURA DE DATOS")
    print("=" * 50)

    print()
    print("🥈 Candidatos (Silver):")
    candidatos = obtener_candidatos(db, limite=5)
    if not candidatos:
        print("   (vacío)")
    for c in candidatos:
        print(f"   {c.get('nombre', '?')} | {c.get('email', '?')} | modelo: {c.get('modelo_llm', '?')}")

    print()
    print("🥈 Job Descriptions (Silver):")
    jobs = obtener_jobs(db)
    if not jobs:
        print("   (vacío)")
    for j in jobs:
        print(f"   {j.get('titulo', '?')} @ {j.get('empresa', '?')}")
        if not job_id:
            job_id = j["id"]

    if job_id:
        print()
        print(f"🥇 Ranking para JD: {job_id}")
        ranking = obtener_ranking(db, job_id)
        if not ranking:
            print("   (sin evaluaciones)")
        for i, r in enumerate(ranking, 1):
            print(f"   #{i} Score: {r.get('score', '?')} | {r.get('nivel', '?')} | {r.get('justificacion', '')[:60]}")


if __name__ == "__main__":
    db = test_conexion()
    if not db:
        sys.exit(1)

    if "--leer" in sys.argv:
        test_lectura(db)
    else:
        jd_id = test_escritura(db)
        test_lectura(db, jd_id)

    print()
    print("💡 Abre Supabase Dashboard > Table Editor para ver los datos")
