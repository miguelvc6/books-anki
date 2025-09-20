# books-anki

Pipeline para extraer vocabulario alemán de libros y generar decks de Anki con ejemplos y metadatos útiles.

## Requisitos previos
- Python 3.10 o superior
- spaCy con algún modelo alemán (`de_core_news_lg`, `de_core_news_md` o `de_core_news_sm`)
- Dependencias listadas en `requirements.txt`
- (Opcional) Un endpoint de LibreTranslate para traducciones automáticas

## Instalación rápida
1. Crear y activar un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   python -m spacy download de_core_news_lg
   ```

## Wiktextract (Kaikki.org)
El script se apoya en el volcado de Wiktextract para enriquecer los términos con género, plurales y traducciones. Descarga el archivo alemán desde Kaikki.org y colócalo en `data/de-extract.jsonl.gz`:

- URL directa: https://kaikki.org/dictionary/downloads/de/de-extract.jsonl.gz
- No es necesario descomprimirlo; el pipeline puede leer archivos `.jsonl` o `.jsonl.gz`.
- Si prefieres guardarlo en otro lugar, indica la ruta con `--wiktextract` al momento de ejecutar.

## Preparación de datos
- Coloca los libros (texto plano codificado en UTF-8) dentro de `data/raw/<nombre_del_proyecto>/*.txt`.
- Cada archivo `.txt` se procesa como un libro independiente. El nombre de la carpeta `nombre_del_proyecto` se utiliza para etiquetar los resultados.

## Uso
Ejecuta el pipeline con Python:
```bash
python -m src.main --book-name harry-potter
```
Parámetros principales:
- `--book-name`: nombre de la carpeta en `data/raw` con los libros a procesar (por defecto `harry-potter`).
- `--input-dir` y `--output-dir`: rutas completas si quieres anular las carpetas por defecto.
- `--wiktextract`: ruta al volcado de Wiktextract, si no está en `data/de-extract.jsonl.gz`.
- `--force-model`: nombre exacto del modelo de spaCy a cargar.
- `--limit-books`: procesa únicamente los primeros *N* archivos (útil para pruebas).
- `--mt-url` y `--mt-api-key`: configuración del endpoint de LibreTranslate si utilizas traducción automática.
- `--log-level`: nivel de detalle en los logs (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

## Resultados
- Los CSV finales se guardan en `data/processed/<nombre_del_proyecto>/`.
- Cada archivo contiene columnas listas para importarse en Anki (`front`, `back`, ejemplo en cloze, formas verbales, género, plural, frecuencia, etc.).

## Variables de entorno útiles
- `LIBRETRANSLATE_URL`: URL base del servicio LibreTranslate (sin la ruta final `/translate`).
- `LIBRETRANSLATE_API_KEY`: clave de API si tu instancia lo requiere.

Con esto deberías poder preparar tus datos, descargar el recurso de Kaikki y generar decks de Anki personalizados.
