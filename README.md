# App UPZ Bogota - Isocronas, Clusterizacion y Optimizacion

App en Streamlit para visualizar variables por UPZ, calcular isocronas dinamicas y correr una optimizacion tipo MCLP.

## Ejecutar local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Subir a Streamlit Community Cloud

1. Crear un repositorio en GitHub.
2. Subir `app.py`, `requirements.txt` y la carpeta `data/`.
3. Entrar a Streamlit Community Cloud.
4. Crear una app nueva desde el repo.
5. Seleccionar `app.py` como archivo principal.

Nota: la app usa OSMnx si esta disponible. Si OSMnx falla por limites de recursos o red, activa el modo aproximado de isocronas.
