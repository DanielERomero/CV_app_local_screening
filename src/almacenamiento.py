import os
import json
from typing import Optional
import httpx


def cargar_env(ruta: str = ".env"):
    if not os.path.exists(ruta):
        return
    with open(ruta, "r") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            if "=" in linea:
                clave, valor = linea.split("=", 1)
                os.environ[clave.strip()] = valor.strip().strip('"').strip("'")

cargar_env()


def obtener_config() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise ValueError("Faltan SUPABASE_URL y SUPABASE_KEY en .env")
    return url, key


class SupabaseClient:
    def __init__(self):
        url, key = obtener_config()
        self.base_url = f"{url}/rest/v1"
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def insertar(self, tabla: str, datos: dict | list[dict]) -> list[dict]:
        if isinstance(datos, dict):
            datos = [datos]
        resp = httpx.post(
            f"{self.base_url}/{tabla}",
            headers=self.headers, json=datos, timeout=30
        )
        if resp.status_code not in (200, 201):
            raise Exception(f"Error INSERT {tabla}: {resp.status_code} - {resp.text}")
        return resp.json()

    def consultar(self, tabla: str, filtros: dict = None,
                  seleccionar: str = "*", limite: int = 100,
                  orden: str = None) -> list[dict]:
        params = {"select": seleccionar, "limit": str(limite)}
        if filtros:
            params.update(filtros)
        if orden:
            params["order"] = orden
        resp = httpx.get(
            f"{self.base_url}/{tabla}",
            headers=self.headers, params=params, timeout=30
        )
        if resp.status_code != 200:
            raise Exception(f"Error SELECT {tabla}: {resp.status_code} - {resp.text}")
        return resp.json()


# ============================================================
# BRONZE - Datos crudos
# ============================================================

def guardar_bronze_cv(db: SupabaseClient, archivo: str, texto: str,
                       paginas: int = 0, caracteres: int = 0) -> str:
    resultado = db.insertar("bronze_cvs", {
        "archivo_nombre": archivo,
        "texto_crudo": texto,
        "num_paginas": paginas,
        "num_caracteres": caracteres,
    })
    return resultado[0]["id"]


def guardar_bronze_jd(db: SupabaseClient, texto: str, fuente: str = "") -> str:
    resultado = db.insertar("bronze_job_descriptions", {
        "texto_crudo": texto,
        "fuente": fuente,
    })
    return resultado[0]["id"]


# ============================================================
# SILVER - Datos estructurados
# ============================================================

def guardar_silver_candidato(db: SupabaseClient, bronze_id: str,
                              datos_cv: dict, modelo: str, tiempo: float) -> str:
    resultado = db.insertar("silver_candidatos", {
        "bronze_cv_id": bronze_id,
        "nombre": datos_cv.get("nombre_completo", ""),
        "email": datos_cv.get("email", ""),
        "telefono": datos_cv.get("telefono", ""),
        "ubicacion": datos_cv.get("ubicacion", ""),
        "resumen": datos_cv.get("resumen_profesional", ""),
        "experiencia": datos_cv.get("experiencia_laboral", []),
        "educacion": datos_cv.get("educacion", []),
        "habilidades_tec": datos_cv.get("habilidades_tecnicas", []),
        "habilidades_bla": datos_cv.get("habilidades_blandas", []),
        "idiomas": datos_cv.get("idiomas", []),
        "certificaciones": datos_cv.get("certificaciones", []),
        "json_completo": datos_cv,
        "modelo_llm": modelo,
        "tiempo_llm": tiempo,
    })
    return resultado[0]["id"]


def guardar_silver_jd(db: SupabaseClient, bronze_id: str,
                       titulo: str, empresa: str = "",
                       requisitos: list = None, experiencia_min: int = 0,
                       educacion_min: str = "", modalidad: str = "") -> str:
    resultado = db.insertar("silver_job_descriptions", {
        "bronze_jd_id": bronze_id,
        "titulo": titulo,
        "empresa": empresa,
        "requisitos": requisitos or [],
        "experiencia_min": experiencia_min,
        "educacion_min": educacion_min,
        "modalidad": modalidad,
    })
    return resultado[0]["id"]


# ============================================================
# GOLD - Evaluaciones
# ============================================================

def guardar_gold_evaluacion(db: SupabaseClient, candidato_id: str,
                             job_id: str, resultado_eval: dict,
                             modelo: str, tiempo: float) -> str:
    resultado = db.insertar("gold_evaluaciones", {
        "candidato_id": candidato_id,
        "job_id": job_id,
        "score": resultado_eval.get("score", 0),
        "nivel": resultado_eval.get("nivel", ""),
        "motivos_contratacion": resultado_eval.get("motivos_contratacion", ""),
        "habilidades_faltantes": resultado_eval.get("habilidades_faltantes", ""),
        "justificacion": resultado_eval.get("justificacion", ""),
        "respuesta_llm": resultado_eval,
        "modelo_eval": modelo,
        "tiempo_eval": tiempo,
    })
    return resultado[0]["id"]


# ============================================================
# CONSULTAS
# ============================================================

def obtener_candidatos(db: SupabaseClient, limite: int = 50) -> list[dict]:
    return db.consultar("silver_candidatos", limite=limite, orden="created_at.desc")

def obtener_jobs(db: SupabaseClient) -> list[dict]:
    return db.consultar("silver_job_descriptions", orden="created_at.desc")

def obtener_ranking(db: SupabaseClient, job_id: str) -> list[dict]:
    return db.consultar(
        "gold_evaluaciones",
        filtros={"job_id": f"eq.{job_id}"},
        orden="score.desc"
    )
