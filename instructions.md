Quiero crear unos decks de Anki con el vocabulario de los libros de Harry Potter en alemán - español.

# Plan técnico

## 0) Decisiones de diseño (antes de tocar código)

* **Unidad léxica**: usa **lema+POS** como clave (p.ej., `aufstehen`/VERB, `Hund`/NOUN). Para separables, el lema debe ser **prefijo+verbo**.
* **Normalización de sustantivos**: **nominativo singular** con su **género inherente**. Formato de Anki: `der/die/das Lemma, Plural`.
* **Filtrado**: por defecto **NOUN/VERB/ADJ/ADV**; excluir **funcionales** (ART/PRON/CCONJ/SCONJ/PART), **PROPN** (nombres propios) y **onomatopeyas**.
* **Traducciones**: prioridad a **diccionario estructurado** (Wiktionary/wiktextract) y *fallback* a un **traductor MT** (DeepL/Google) sólo si falta entrada léxica.
* **Contexto**: añade **una frase de ejemplo** del libro por entrada (mejor tipo *Cloze* en un campo extra).

---

## 1) Ingesta y limpieza

* **Lectura**: asegura `utf-8` y **normalización Unicode NFC**.
* **Deshyphenation**: repara guiones de final de línea (`-\n` → “”).
* **Segmentación de oraciones**: para ejemplo de contexto y detección de verbos separables por dependencia.
* **Tokenización**: usa un **pipeline UD** para alemán:

  * Opción A (más morfosintaxis): **Stanza** (`de`), mejor en *features* pero más lento.
  * Opción B (simple/rápida): **spaCy** `de_core_news_lg`.
* **Arreglos de texto**: `ftfy` para rarezas Unicode; normaliza comillas, guiones, etc.

**Herramientas**: `stanza`; `ftfy`; `regex` o `spacy` (+ `de_core_news_lg`)

---

## 2) Etiquetado, lematización y separables

* **POS/DEP**: del pipeline elegido.
* **Lemas**:

  * spaCy da lemas decentes; para alemán puro, considera **HanTa** o **DEMorphy** si quieres contraste de lemas.
* **Verbos separables**:

  * Detecta partícula separable por dependencia **UD `compound:prt`** (spaCy) o relación equivalente (Stanza).
  * **Regla**: si `token.pos_ == VERB` y tiene hijo `PART` con dep `compound:prt`, construye lema `prefijo + lemma_verbo_base` (p.ej., `auf` + `stehen` → `aufstehen`).
  * Lista de prefijos separables útiles para validación: `ab, an, auf, aus, bei, ein, fort, her, hin, los, mit, nach, vor, weg, zu, zurück, zusammen`.
  * Maneja también participios con `ge-` intercalado: si lema base es `anfangen` y forma es `angefangen`, el lema ya será `anfangen` (OK), pero si hay partícula separada **y** participio en la frase, prioriza reconstruir con partícula.


**Herramientas**: `spacy`, `stanza`, **HanTa** (`HanTa`), **DEMorphy** (`demorphy`), **char\_split**.

---

## 3) Filtrado y deduplicación

* **Filtra** por POS deseados.
* **NER**: excluye `PROPN` (p.ej., *Hogwarts*, *Dumbledore*).
* **Stopwords**: puedes excluir lista estándar de spaCy, pero considera **no eliminar** adverbios frecuentes útiles (*eben, doch, halt*), o mantenlos con etiqueta ADV.
* **Clave única**: `(lemma, POS)` dentro de cada libro.
* **“Novedad” entre libros**: elimina del libro *n* aquellos `(lemma, POS)` ya vistos en `1..(n-1)` para evitar estudiar de forma repetida.

**Herramientas**: `spacy` (stopwords/NER), `pandas`.

---

## 4) Género y plural de sustantivos

* Necesitas **léxico** que dé **género** y **plural(es)** por **lema**.
* Rutas:

  1. **Wiktionary (wiktextract)** *offline*: JSONL con entradas estructuradas (POS, género, plurales, separabilidad, traducciones). **Pros**: libre, reproducible; **Contras**: cobertura no perfecta, hay ruido.
  2. **DEMorphy** / **HanTa**: buenos para lemas y algunas *features*, pero **no siempre** dan género/plural fiable léxico.
  3. **Duden/LEO**: scraping rompe ToS → **evitar**.
* **Plural “sufijo breve”** (estilo `-e`, `-er`, `-en`, `-s`, `⸚`):

  * Calcula el **diferencial** entre plural y lema para derivar el sufijo (detecta umlaut: `Mann → Männer` → `-¨er`).
  * Si hay múltiples plurales con sentidos distintos, muestra el **más frecuente** (puedes ponderar por **wordfreq** para alemán) y añade el alternativo en un campo extra.

**Herramientas**: **wiktextract** (dataset), `wordfreq`.

---

## 5) Traducciones DE→ES (prioridad a diccionario, *fallback* MT)

* **Paso 1 (diccionario)**: busca en **wiktextract** traducciones al español del **lema+POS**.

  * Si hay varias acepciones, elige **1–2** más generales; conserva todas en campo extra.
* **Paso 2 (MT)**: si no hay traducción diccionario, usa:

  * **DeepL API** (calidad alta; coste; necesitas clave).
  * o **Google Cloud Translate** (cobertura alta; coste).
  * o **LibreTranslate** local (gratis; menor calidad).
* **Desambiguación por contexto** (opcional y potente): si un lema es polisémico, usa la **oración fuente** y ejecuta **traducción de oración + alineación léxica aproximada** (simples heurísticas con similitud de subcadenas o atención vía un modelo pequeño), o pregunta al traductor **“glossario de una palabra”** si la API lo permite. Si no, acepta polisemia y limita a glosa general.

**Herramientas**: `requests`/`aiohttp`, **DeepL** SDK o `google-cloud-translate`, **wiktextract**.

---

## 6) Formato de tarjeta y exportación

* **Front (alemán)**:

  * NOUN: `der/die/das Hund, -e` (con artículo correcto).
  * VERB: infinitivo (`aufstehen`), opcional **formas base**: `steht auf – stand auf – ist aufgestanden` si las tienes (wiktionary suele tener “Stammformen”).
  * ADJ/ADV: lema (`mäßig`).
* **Back (español)**: 1–2 traducciones principales; si hay polisemia, “; ” separadas.
* **Campos extra** (muy recomendables, aunque no los uses ahora):

  * **Ejemplo** (oración del libro, con *Cloze*).
  * **Frecuencia** en el libro.
  * **Libro/Capítulo**.
  * **Notas** (p.ej., separable/inseparable, prefijos relacionados).
* **CSV**: separador `,` o `\t` (más seguro). Codifica como **`utf-8-sig`** para Excel/Anki. Escapa comas y comillas.
* **1 CSV por libro**.

**Herramientas**: `pandas` (to\_csv).

---

## 7) Métricas, QA

* **Validación**:

  * Muestreo aleatorio de 100 entradas/libro → tasa de **género correcto**, **plural correcto**, **traducción plausible**.
  * **Cobertura**: % de entradas con traducción por diccionario vs MT.
  * **Errores comunes**: separables no reconstruidos; `der See/die See`; plurales alternativos; adverbios pronominales (`davon`, `darauf`); compuestos raros.
* **Registro**: `logging` con contadores por categoría de fallo para iterar reglas.

---

## 8) Fallos previsibles y mitigaciones

* **Verbos separables no detectados** si el parser falla → añade **heurística de proximidad**: si verbo finito y a ≤3 tokens hay `PART` en una lista de prefijos separables y no hay dependencia marcada, **combina**.
* **Género ambiguo** (p.ej., `Joghurt` der/das) → elige **forma dominante** del léxico; si en contexto hay determinante (`der/die/das/ein/kein/mein…`) con `Gender` en *features*, respétalo.
* **Plurales múltiples** → muestra **principal**; para Anki “front” mantén el principal.
* **Traducciones muy específicas** del mundo HP → si no hay ES en diccionario, usa **MT** pero marca con etiqueta `proper/fiction` si POS=PROPN (mejor excluir si no quieres nombres propios).
* **Compuestos muy largos** → conserva compuesto.

---

## 10) Herramientas recomendadas (resumen)

* NLP: **spaCy `de_core_news_lg`** *(o Stanza)*, **HanTa**, **DEMorphy** (opcional contraste).
* Diccionario: **wiktextract** (dump German de Wiktionary).
* Compounds: **char\_split** (opcional).
* Frecuencia: **wordfreq** (alemán).
* Utilidades: `ftfy`, `regex`, `pandas`, `duckdb`/`sqlite3`, `aiohttp`.
* MT (fallback): **DeepL API** o **Google Cloud Translate**; alternativa **LibreTranslate** local.

---

## 11) Flujo por libro (pseudopasos)

1. Cargar texto → normalizar Unicode → reparar guiones.
2. Segmentar oraciones → tokenizar → POS/DEP.
3. Reconstruir **separables** (regla UD + heurística).
4. Extraer candidatas por POS → **lemmatizar** → descartar PROPN/stopwords si aplica.
5. Contar frecuencias y guardar **una oración de ejemplo** por lema.
6. **Restar** lemmas ya cubiertos en libros previos.
7. Para NOUN: consultar **género/plural** (wiktextract).
8. Para todos: obtener **traducciones ES** (wiktextract → MT fallback).
9. Formatear “front/back” y **exportar CSV**.
10. Actualizar **set acumulado** de `(lemma, POS)`.