"""Command-line pipeline to extract German vocabulary from Harry Potter books and build Anki decks."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import ftfy
import pandas as pd
import requests

try:
    import spacy
    from spacy.language import Language
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "spaCy is required for this project. Install dependencies with pip install -r requirements.txt."
    ) from exc

LOGGER = logging.getLogger(__name__)
ALLOWED_POS = {"NOUN", "VERB", "ADJ", "ADV"}
SEPARABLE_PREFIXES = {
    "ab",
    "an",
    "auf",
    "aus",
    "bei",
    "ein",
    "fort",
    "her",
    "hin",
    "los",
    "mit",
    "nach",
    "vor",
    "weg",
    "zu",
    "zur\u00fcck",
    "zusammen",
}
MT_DEFAULT_URL = os.environ.get("LIBRETRANSLATE_URL")
MT_API_KEY = os.environ.get("LIBRETRANSLATE_API_KEY")
DEFAULT_WIKTEXTRACT_PATH = Path("data/de-extract.jsonl.gz") if Path("data/de-extract.jsonl.gz").exists() else None


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    wiktextract_path: Optional[Path]
    mt_url: Optional[str] = None
    mt_api_key: Optional[str] = None
    limit_books: Optional[int] = None
    force_model: Optional[str] = None


@dataclass
class LexemeData:
    lemma: str
    pos: str
    genders: List[str] = field(default_factory=list)
    plurals: List[str] = field(default_factory=list)
    translations: List[str] = field(default_factory=list)
    verb_forms: Dict[str, str] = field(default_factory=dict)


class WiktextractLexicon:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._index: Dict[Tuple[str, str], LexemeData] = {}
        if self.path and self.path.exists():
            self._build_index()
        else:
            LOGGER.warning(
                "Wiktextract dataset not provided; gender/plural and dictionary translations may be missing."
            )

    def _build_index(self) -> None:
        LOGGER.info("Loading wiktextract dataset from %s", self.path)
        if self.path.suffix == ".gz":
            import gzip

            def open_stream(file_path: Path):
                return gzip.open(file_path, "rt", encoding="utf-8")
        else:
            def open_stream(file_path: Path):
                return open(file_path, "rt", encoding="utf-8")

        with open_stream(self.path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    blob = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.debug("Skipping malformed json line")
                    continue
                if blob.get("lang_code") != "de":
                    continue
                lemma = blob.get("word")
                if not lemma:
                    continue
                pos = self._map_pos(blob.get("pos"))
                if pos not in ALLOWED_POS:
                    continue
                key = (lemma.lower(), pos)
                entry = self._index.setdefault(key, LexemeData(lemma=lemma, pos=pos))
                genders = blob.get("genders")
                if genders and not entry.genders:
                    entry.genders = [g for g in genders if g]
                for form in blob.get("forms", []) or []:
                    tags = {t.lower() for t in form.get("tags", []) or []}
                    form_text = form.get("form")
                    if not form_text:
                        continue
                    if "plural" in tags and form_text not in entry.plurals:
                        entry.plurals.append(form_text)
                    for tag in (
                        "infinitive",
                        "present 3rd",
                        "present 3rd singular",
                        "preterite",
                        "past participle",
                    ):
                        if tag in tags and tag not in entry.verb_forms:
                            entry.verb_forms[tag] = form_text
                for sense in blob.get("senses", []) or []:
                    for translation in sense.get("translations", []) or []:
                        lang_code = (
                            translation.get("lang_code")
                            or translation.get("language_code")
                            or translation.get("lang")
                        )
                        if lang_code not in {
                            "es",
                            "spa",
                            "spanish",
                            "espanol",
                            "espa\u00f1ol",
                        }:
                            continue
                        word = translation.get("word")
                        if not word:
                            continue
                        note_bits: List[str] = []
                        for key_name in ("sense", "usage", "note", "tags"):
                            value = translation.get(key_name)
                            if isinstance(value, str):
                                note_bits.append(value)
                            elif isinstance(value, Sequence):
                                note_bits.extend(str(x) for x in value)
                        if note_bits:
                            formatted = f"{word} ({'; '.join(note_bits)})"
                        else:
                            formatted = word
                        if formatted not in entry.translations:
                            entry.translations.append(formatted)
        LOGGER.info("Loaded %s lexemes from wiktextract", len(self._index))

    @staticmethod
    def _map_pos(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        raw = raw.lower()
        mapping = {
            "noun": "NOUN",
            "verb": "VERB",
            "adjective": "ADJ",
            "adverb": "ADV",
        }
        return mapping.get(raw)

    def lookup(self, lemma: str, pos: str) -> Optional[LexemeData]:
        return self._index.get((lemma.lower(), pos))


class LibreTranslateClient:
    def __init__(self, url: Optional[str], api_key: Optional[str] = None) -> None:
        self.url = url.rstrip("/") if url else None
        self.api_key = api_key
        self.cache: Dict[str, List[str]] = {}
        if not self.url:
            LOGGER.info("MT fallback disabled (no LibreTranslate URL provided)")

    def translate(self, text: str) -> Optional[List[str]]:
        if not self.url:
            return None
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        payload = {"q": text, "source": "de", "target": "es", "format": "text"}
        if self.api_key:
            payload["api_key"] = self.api_key
        try:
            response = requests.post(f"{self.url}/translate", data=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:  # pragma: no cover - network failure
            LOGGER.warning("MT request failed for %s: %s", text, exc)
            return None
        translation = data.get("translatedText") or data.get("translated_text")
        if translation:
            variants = [
                variant.strip()
                for variant in re.split(r"[,;/]", translation)
                if variant.strip()
            ]
            self.cache[text] = variants
            return variants
        return None


@dataclass
class TermEntry:
    lemma: str
    pos: str
    surface: str
    frequency: int = 0
    example_sentence: Optional[str] = None
    example_cloze: Optional[str] = None
    gender: Optional[str] = None
    plural: Optional[str] = None
    plural_display: Optional[str] = None
    translations: List[str] = field(default_factory=list)
    translation_source: Optional[str] = None
    verb_forms: Optional[str] = None


def normalize_text(raw_text: str) -> str:
    text = ftfy.fix_text(raw_text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"-\s*\n", "", text)
    text = unicodedata.normalize("NFC", text)
    return text


def split_paragraphs(text: str) -> List[str]:
    chunks = re.split(r"\n{2,}", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def load_model(force_name: Optional[str] = None) -> Language:
    candidates = (
        [force_name]
        if force_name
        else ["de_core_news_lg", "de_core_news_md", "de_core_news_sm"]
    )
    for name in candidates:
        if not name:
            continue
        try:
            LOGGER.info("Loading spaCy model %s", name)
            return spacy.load(name)
        except OSError:
            LOGGER.info("spaCy model %s not found", name)
            continue
    raise SystemExit(
        "No German spaCy model available. Install one with python -m spacy download de_core_news_lg."
    )


def reconstruct_lemma(token) -> str:
    lemma = token.lemma_
    prefix = None
    for child in token.children:
        dep = child.dep_.lower()
        if child.pos_ == "PART" and dep in {"prt", "compound:prt"}:
            prefix = child.lemma_.lower()
            break
    if not prefix:
        prefix = infer_particle_from_context(token)
    if prefix:
        if lemma.startswith(prefix):
            return lemma
        return f"{prefix}{lemma}"
    return lemma


def infer_particle_from_context(token) -> Optional[str]:
    if token.pos_ != "VERB":
        return None
    sent = token.sent
    idx = token.i - sent.start
    window = sent[max(0, idx - 3) : idx + 4]
    for part in window:
        if part is token:
            continue
        text = part.lemma_.lower()
        if part.pos_ == "PART" and text in SEPARABLE_PREFIXES:
            return text
    return None


def is_candidate(token) -> bool:
    if token.is_space or not token.text.strip():
        return False
    if token.pos_ not in ALLOWED_POS:
        return False
    if token.pos_ == "PROPN":
        return False
    if token.lemma_ == "":
        return False
    if token.like_num:
        return False
    if token.pos_ == "ADV" and token.is_stop:
        return True
    return True


def make_cloze(sentence: str, surface: str) -> str:
    pattern = re.compile(re.escape(surface), flags=re.IGNORECASE)
    return pattern.sub(f"{{{{c1::{surface}}}}}", sentence, count=1)


def derive_plural_display(lemma: str, plural: Optional[str]) -> Optional[str]:
    if not plural:
        return None
    if plural == lemma:
        return "-"
    umlaut_pairs = {
        ("a", "\u00e4"),
        ("o", "\u00f6"),
        ("u", "\u00fc"),
    }
    umlaut_present = any(src in lemma and tgt in plural for src, tgt in umlaut_pairs)
    if plural.startswith(lemma):
        suffix = plural[len(lemma) :]
    else:
        suffix = plural
    suffix = suffix or "-"
    if not suffix.startswith("-"):
        suffix = "-" + suffix
    if umlaut_present and suffix == "-er":
        return "-\u00a8er"
    return suffix


def article_from_gender(gender: Optional[str]) -> Optional[str]:
    mapping = {
        "masculine": "der",
        "feminine": "die",
        "neuter": "das",
        "m": "der",
        "f": "die",
        "n": "das",
    }
    if not gender:
        return None
    return mapping.get(gender.lower())


def format_front(entry: TermEntry) -> str:
    if entry.pos == "NOUN":
        article = article_from_gender(entry.gender) or ""
        lemma = entry.lemma
        plural = entry.plural_display or entry.plural or "-"
        core = " ".join(part for part in (article, lemma) if part)
        return f"{core}, {plural}"
    if entry.pos == "VERB":
        if entry.verb_forms:
            return f"{entry.lemma} ({entry.verb_forms})"
        return entry.lemma
    return entry.lemma


def format_back(entry: TermEntry) -> str:
    if entry.translations:
        return "; ".join(entry.translations[:2])
    return ""


def sanitize_book_name(path: Path) -> str:
    stem = path.stem.lower()
    normalized = unicodedata.normalize("NFKD", stem)
    ascii_friendly = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
    base = re.sub(r"[^a-z0-9]+", "_", ascii_friendly)
    return base.strip("_")


def process_book(
    path: Path,
    nlp: Language,
    lexicon: WiktextractLexicon,
    mt_client: LibreTranslateClient,
    seen_global: Set[Tuple[str, str]],
) -> Tuple[List[TermEntry], Set[Tuple[str, str]]]:
    LOGGER.info("Processing book %s", path.name)
    raw_text = path.read_text(encoding="utf-8")
    text = normalize_text(raw_text)
    paragraphs = split_paragraphs(text)
    entries: Dict[Tuple[str, str], TermEntry] = {}
    for doc in nlp.pipe(paragraphs, batch_size=8):
        for sent in doc.sents:
            for token in sent:
                if not is_candidate(token):
                    continue
                lemma = (
                    reconstruct_lemma(token) if token.pos_ == "VERB" else token.lemma_
                )
                pos = token.pos_
                key = (lemma.lower(), pos)
                if key in seen_global:
                    continue
                entry = entries.get(key)
                if not entry:
                    entry = TermEntry(lemma=lemma, pos=pos, surface=token.text)
                    entry.example_sentence = sent.text.strip()
                    entry.example_cloze = make_cloze(entry.example_sentence, token.text)
                    entries[key] = entry
                entry.frequency += 1
    for key, entry in entries.items():
        lexeme = lexicon.lookup(entry.lemma, entry.pos)
        translations: List[str] = []
        if lexeme:
            if lexeme.translations:
                translations = lexeme.translations
            if entry.pos == "NOUN":
                if lexeme.genders:
                    entry.gender = lexeme.genders[0]
                if lexeme.plurals:
                    entry.plural = lexeme.plurals[0]
                    entry.plural_display = derive_plural_display(
                        entry.lemma, entry.plural
                    )
            if entry.pos == "VERB" and lexeme.verb_forms:
                parts = []
                for tag in (
                    "present 3rd",
                    "present 3rd singular",
                    "preterite",
                    "past participle",
                ):
                    form = lexeme.verb_forms.get(tag)
                    if form:
                        parts.append(form)
                if parts:
                    entry.verb_forms = " - ".join(parts)
        if not translations:
            mt = mt_client.translate(entry.lemma)
            if mt:
                translations = mt
                entry.translation_source = "mt"
        else:
            entry.translation_source = "dictionary"
        if translations:
            entry.translations = translations
    seen_keys = set(entries.keys())
    return list(entries.values()), seen_keys


def to_records(entries: List[TermEntry], book_label: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for entry in entries:
        records.append(
            {
                "front": format_front(entry),
                "back": format_back(entry),
                "lemma": entry.lemma,
                "pos": entry.pos,
                "gender": entry.gender,
                "plural": entry.plural,
                "plural_display": entry.plural_display,
                "translations": "; ".join(entry.translations),
                "translation_source": entry.translation_source,
                "example": entry.example_sentence,
                "example_cloze": entry.example_cloze,
                "frequency": entry.frequency,
                "book": book_label,
            }
        )
    return records


def run_pipeline(config: PipelineConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    nlp = load_model(config.force_model)
    lexicon = WiktextractLexicon(config.wiktextract_path)
    mt_client = LibreTranslateClient(
        config.mt_url or MT_DEFAULT_URL, config.mt_api_key or MT_API_KEY
    )
    book_paths = sorted(config.input_dir.glob("*.txt"))
    if config.limit_books is not None:
        book_paths = book_paths[: config.limit_books]
    if not book_paths:
        raise SystemExit(f"No .txt files found in {config.input_dir}")
    seen_global: Set[Tuple[str, str]] = set()
    for idx, book_path in enumerate(book_paths, start=1):
        entries, seen = process_book(book_path, nlp, lexicon, mt_client, seen_global)
        seen_global.update(seen)
        records = to_records(entries, book_path.stem)
        frame = pd.DataFrame(records)
        output_name = f"{sanitize_book_name(book_path)}.csv"
        output_path = config.output_dir / output_name
        LOGGER.info("Writing %s entries to %s", len(frame), output_path)
        frame.sort_values(by=["pos", "lemma"], inplace=True)
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")


def parse_args(argv: Optional[Sequence[str]] = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--book-name",
        type=str,
        default="harry-potter",
        help="Name of the folder inside data/raw to process",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory with raw German book .txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write deck CSV files",
    )
    parser.add_argument(
        "--wiktextract",
        type=Path,
        default=DEFAULT_WIKTEXTRACT_PATH,
        help="Path to wiktextract German JSONL(.gz) dump",
    )
    parser.add_argument(
        "--mt-url",
        type=str,
        default=MT_DEFAULT_URL,
        help="LibreTranslate endpoint for MT fallback",
    )
    parser.add_argument(
        "--mt-api-key",
        type=str,
        default=MT_API_KEY,
        help="API key for LibreTranslate (if required)",
    )
    parser.add_argument(
        "--limit-books",
        type=int,
        default=None,
        help="Process only the first N books (for debugging)",
    )
    parser.add_argument(
        "--force-model",
        type=str,
        default=None,
        help="Force a specific spaCy model name",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args(argv)
    if args.input_dir is None:
        args.input_dir = Path("data/raw") / args.book_name
    if args.output_dir is None:
        args.output_dir = Path("data/processed") / args.book_name
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s %(message)s"
    )
    return PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        wiktextract_path=args.wiktextract,
        mt_url=args.mt_url,
        mt_api_key=args.mt_api_key,
        limit_books=args.limit_books,
        force_model=args.force_model,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    run_pipeline(config)


if __name__ == "__main__":
    main()
