# CV Screening System

Aplicación local basada en Streamlit y LLMs (Ollama) para analizar, estructurar y evaluar currículums vitae en PDF contra una descripción de puesto (Job Description).

## 🚀 Requisitos Previos

- **Python** 3.10+
- **[uv](https://docs.astral.sh/uv/)**: Gestor de paquetes ultrarrápido para Python.
- **[Ollama](https://ollama.com/)**: Para correr modelos de lenguaje localmente (ej: `mistral`, `llama3`).

## 📥 Instalación (usando `uv`)

1. **Clona el repositorio** en tu computadora:
   ```bash
   git clone https://github.com/DanielERomero/CV_app_local_screening.git
   cd CV_app_local_screening
   ```

2. **Crea el entorno virtual e instala las dependencias**:
   Como el proyecto utiliza `uv`, puedes sincronizar todo el entorno virtual a partir del archivo `pyproject.toml` / `uv.lock` con este simple comando:
   ```bash
   uv sync
   ```
   *(Esto creará automáticamente la carpeta `.venv` y descargará lo necesario de forma muy rápida).*

3. **Configura las variables de entorno**:
   Crea un archivo llamado `.env` en la raíz del proyecto (basado en el `.env.example` en caso de existir) y coloca allí tus credenciales de Supabase (si usas persistencia en la nube):
   ```env
   SUPABASE_URL=tu_url_aqui
   SUPABASE_KEY=tu_key_aqui
   ```

## ⚙️ Ejecución

1. **Inicia el servidor local de modelos (Ollama)**:
   Asegúrate de que Ollama se esté ejecutando en segundo plano, y de contar con los modelos que prefieras descargados:
   ```bash
   # Iniciar el servicio (si no se inicia solo con la app de escritorio)
   ollama serve

   # Descargar el modelo por defecto (si no lo tienes)
   ollama pull mistral:latest 
   ```

2. **Levanta la interfaz web con Streamlit**:
   Utiliza `uv` para correr el script dentro del entorno virtual:
   ```bash
   uv run streamlit run app.py
   ```
   *(El navegador se abrirá automáticamente en `http://localhost:8501`)*

## 💡 Uso de la Aplicación
- **Pestaña 1**: Sube tus CVs en formato PDF. El sistema extraerá el texto y lo convertirá en un JSON estructurado.
- **Pestaña 2**: Pega el texto del Job Description (la oferta de trabajo).
- **Pestaña 3**: Ejecuta la evaluación. El modelo calificará a cada candidato del 0 al 100 de acuerdo al JD.
- **Pestaña 4**: Visualiza el ranking final, los puntos fuertes y lo que le falta a cada perfil, y descarga los resultados!
