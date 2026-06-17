# App UPZ Bogota - Pinturas BLER

Aplicacion Streamlit con tres vistas funcionales:

1. Variables por UPZ o por localidad con mapa de calor y panel de detalle.
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

## Nuevas funcionalidades comerciales

- La vista 1 permite cambiar entre nivel `UPZ` y nivel `Localidad`.
- En nivel `Localidad` se pueden visualizar ventas reales 2025, ventas reales 2024, crecimiento de ventas y participacion de zona.
- La correspondencia UPZ-localidad esta incluida dentro de `app.py`, basada en la lista suministrada.
- Las zonas comerciales del Excel que no corresponden a localidades de Bogota se muestran aparte en un desplegable.

## Despliegue

1. Subir esta carpeta a GitHub.
2. Entrar a Streamlit Community Cloud.
3. Crear una nueva app.
4. Seleccionar el repositorio y `app.py` como archivo principal.
5. Deploy.

Cuando actualices los archivos de datos en GitHub, Streamlit redeploya la app y vuelve a leer los datos.
