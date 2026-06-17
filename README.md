# App UPZ Bogota - Pinturas BLER

Aplicacion Streamlit con tres vistas funcionales:

1. Variables por UPZ con mapa de calor y panel de detalle.
2. Isocronas dinamicas por UPZ con cobertura.
3. Resultados con clusterizacion K4 y optimizacion MCLP.

## Archivos principales

- `app.py`: codigo de la aplicacion.
- `requirements.txt`: dependencias para Streamlit Cloud.
- `data/dataset_final.csv`: variables base por UPZ.
- `data/upz_clusterizada.csv`: clusters oficiales, usando `cluster_k4`.
- `data/pensionadosupz_042023.*`: shapefile de UPZ.
- `data/logo_bler.png`: logo mostrado en el encabezado.
- `data/ventas_clientes_asesor_zona.xlsx`: ventas reales por asesor y localidad/zona.

## Despliegue

1. Subir esta carpeta a GitHub.
2. Entrar a Streamlit Community Cloud.
3. Crear una nueva app.
4. Seleccionar el repositorio y `app.py` como archivo principal.
5. Deploy.

Nota: la vista de ventas reales esta por localidad/zona. Para pintarla directamente sobre UPZ se necesita una tabla de correspondencia UPZ-localidad o un shapefile de localidades.
