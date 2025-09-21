"""Command-line pipeline to extract German vocabulary from Harry Potter books and build Anki decks."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import ftfy
import pandas as pd
import requests
from wordfreq import zipf_frequency

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
POS_PRIORITY = {
    "NOUN": 4,
    "VERB": 3,
    "ADJ": 2,
    "ADV": 1,
}
ARTICLE_PATTERN = re.compile(r"^(?:der|die|das|den|dem|des|ein|eine|einen|einem|eines)\s+", re.IGNORECASE)
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
    translations_en: List[str] = field(default_factory=list)
    definitions_de: List[str] = field(default_factory=list)
    verb_forms: Dict[str, str] = field(default_factory=dict)


class WiktextractLexicon:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._index: Dict[Tuple[str, str], LexemeData] = {}
        self._form_index: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
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

        def _dedup_keep_order(xs: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for x in xs:
                k = x.lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(x)
            return out

        debug_targets = {("Hund", "NOUN"), ("gehen", "VERB"), ("aufstehen", "VERB")}
        is_german_dump = bool(
            self.path and self.path.name.lower().startswith(("de-", "dewiktionary"))
        )
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
                if not entry.genders:
                    tag_genders = [t for t in (blob.get("tags") or []) if t in {"masculine", "feminine", "neuter", "common"}]
                    if tag_genders:
                        entry.genders = tag_genders
                for form in blob.get("forms", []) or []:
                    tags = {t.lower() for t in form.get("tags", []) or []}
                    form_text = form.get("form")
                    if not form_text:
                        continue
                    if "plural" in tags and form_text not in entry.plurals:
                        entry.plurals.append(form_text)
                    form_key = (form_text.lower(), pos)
                    if key not in self._form_index[form_key]:
                        self._form_index[form_key].append(key)
                    normalized_form = normalize_plural_form(form_text)
                    if normalized_form:
                        normalized_key = (normalized_form.lower(), pos)
                        if key not in self._form_index[normalized_key]:
                            self._form_index[normalized_key].append(key)
                    for tag in (
                        "infinitive",
                        "present 3rd",
                        "present 3rd singular",
                        "preterite",
                        "past participle",
                    ):
                        if tag in tags and tag not in entry.verb_forms:
                            entry.verb_forms[tag] = form_text
                def _append_translation(data_obj: dict) -> None:
                    lang_code = (
                        data_obj.get("lang_code")
                        or data_obj.get("language_code")
                        or data_obj.get("lang")
                    )
                    lang = str(lang_code).lower() if isinstance(lang_code, str) else None
                    if lang not in {"en", "eng", "english"}:
                        return
                    word = data_obj.get("word")
                    if not word:
                        return
                    note_bits: List[str] = []
                    for key_name in ("sense", "usage", "note", "tags"):
                        value = data_obj.get(key_name)
                        if isinstance(value, str):
                            note_bits.append(value)
                        elif isinstance(value, Sequence):
                            note_bits.extend(str(x) for x in value)
                    formatted = f"{word} ({'; '.join(note_bits)})" if note_bits else word
                    formatted = formatted.strip()
                    if formatted:
                        entry.translations_en.append(formatted)

                for sense in blob.get("senses", []) or []:
                    for translation in sense.get("translations", []) or []:
                        _append_translation(translation)
                    if is_german_dump:
                        for gloss in sense.get("glosses") or []:
                            if not isinstance(gloss, str):
                                continue
                            g = re.sub(r"\s+", " ", gloss).strip()
                            if not g:
                                continue
                            if g not in entry.definitions_de:
                                entry.definitions_de.append(g)
                for translation in blob.get("translations") or []:
                    _append_translation(translation)
                entry.translations_en = _dedup_keep_order(entry.translations_en)
                if (
                    LOGGER.isEnabledFor(logging.DEBUG)
                    and (entry.lemma, entry.pos) in debug_targets
                ):
                    LOGGER.debug(
                        "Lexeme %s/%s: EN=%d",
                        entry.lemma,
                        entry.pos,
                        len(entry.translations_en),
                    )
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

    def get_dictionary_glosses(
        self, lemma: str, pos: str, *, max_n: int = 2
    ) -> tuple[list[str], str | None]:
        """Return (glosses, source) with English dictionary glosses if available."""
        lexeme = self.lookup(lemma, pos)
        if not lexeme:
            _, lexeme = self.resolve(lemma, pos)
        if not lexeme:
            return [], None

        def pick(values: Sequence[str] | None) -> list[str]:
            seen: Set[str] = set()
            result: list[str] = []
            if not values:
                return result
            for value in values:
                normalized = re.sub(r"\s+", " ", value).strip()
                if not normalized:
                    continue
                key = normalized.lower()
                if key in seen:
                    continue
                seen.add(key)
                result.append(normalized)
                if len(result) >= max_n:
                    break
            return result

        cleaned = pick(getattr(lexeme, 'translations_en', None))
        if cleaned:
            return cleaned, "dictionary-en"
        return [], None


    def resolve(
        self, lemma: str, pos: str, surface: Optional[str] = None
    ) -> Tuple[str, Optional[LexemeData]]:
        for candidate in self._iter_candidates(lemma, surface):
            key = (candidate.lower(), pos)
            direct = self._index.get(key)
            via_forms = self._resolve_from_forms(key)
            chosen: Optional[LexemeData] = None
            if direct and via_forms:
                if self._prefer_lexeme(via_forms, direct, pos):
                    chosen = via_forms
                else:
                    chosen = direct
            elif direct:
                chosen = direct
            elif via_forms:
                chosen = via_forms
            if chosen:
                return chosen.lemma, chosen
        return lemma, None

    @staticmethod
    def _iter_candidates(lemma: Optional[str], surface: Optional[str]) -> Iterable[str]:
        seen: Set[str] = set()
        for raw in (lemma, surface):
            if not raw:
                continue
            text_value = raw.strip()
            if not text_value:
                continue
            for variant in (text_value, text_value.lower()):
                if variant not in seen:
                    seen.add(variant)
                    yield variant
        if not seen and lemma:
            yield lemma

    def _resolve_from_forms(self, key: Tuple[str, str]) -> Optional[LexemeData]:
        form_keys = self._form_index.get(key)
        if not form_keys:
            return None
        best: Optional[LexemeData] = None
        best_score = -1
        for target_key in form_keys:
            lexeme = self._index.get(target_key)
            if not lexeme:
                continue
            score = self._lexeme_score(lexeme, key[1])
            if score > best_score:
                best = lexeme
                best_score = score
        return best

    def _lexeme_score(self, lexeme: LexemeData, pos: str) -> int:
        score = 0
        if pos == "NOUN":
            if lexeme.genders:
                score += 4
            if lexeme.plurals:
                score += 4
        if pos == "VERB" and lexeme.verb_forms:
            score += 3
        if lexeme.translations_en:
            score += 2
        if lexeme.lemma:
            score += max(0, 3 - lexeme.lemma.count(" "))
        return score

    def _prefer_lexeme(self, candidate: LexemeData, current: LexemeData, pos: str) -> bool:
        if candidate is current:
            return False
        cand_score = self._lexeme_score(candidate, pos)
        curr_score = self._lexeme_score(current, pos)
        if cand_score != curr_score:
            return cand_score > curr_score
        cand_len = len(candidate.lemma or "")
        curr_len = len(current.lemma or "")
        if cand_len != curr_len:
            return cand_len < curr_len
        return (candidate.lemma or "") < (current.lemma or "")

    @staticmethod
    def _normalize_for_frequency(text: str) -> str:
        normalized = re.sub(r"[^a-zA-Z\u00e4\u00f6\u00fc\u00df]", "", text.lower())
        return normalized

    @staticmethod
    def _base_variants(text: str) -> List[str]:
        variants: List[str] = []
        if not text:
            return variants
        stripped = text.strip()
        if not stripped:
            return variants
        lowered = stripped.lower()
        for value in (stripped, lowered):
            if value and value not in variants:
                variants.append(value)
        san_apostrophe = re.sub(r"[\u2019']", "", lowered)
        if san_apostrophe and san_apostrophe not in variants:
            variants.append(san_apostrophe)
        apostrophe_suffix = re.sub(r"(?:[\u2019']s)$", "", lowered)
        if apostrophe_suffix and apostrophe_suffix not in variants:
            variants.append(apostrophe_suffix)
        return variants

    @staticmethod
    def _expand_verb_forms(stem: str) -> List[str]:
        variants: List[str] = []
        if not stem:
            return variants
        stem = stem.lower()
        variants.append(stem)
        if stem.endswith("end") and len(stem) > 3:
            variants.append(stem[:-1])
        if stem.endswith("st") and len(stem) > 2:
            root = stem[:-2]
            if root:
                variants.append(root)
                variants.append(root + "en")
        if stem.endswith("t") and len(stem) > 1:
            root = stem[:-1]
            variants.append(root)
            variants.append(root + "en")
        if stem.endswith("e") and len(stem) > 1:
            root = stem[:-1]
            variants.append(root)
            variants.append(root + "en")
        if not stem.endswith("en"):
            variants.append(stem + "en")
        return variants

    def _iter_infinitive_candidates(self, lemma: str, surface: Optional[str]) -> Iterable[str]:
        seen: Set[str] = set()
        sources: List[str] = [lemma]
        if surface:
            sources.append(surface)
        for raw in sources:
            for base in self._base_variants(raw):
                for candidate in self._expand_verb_forms(base):
                    cleaned = re.sub(r"[^a-zA-Z\u00e4\u00f6\u00fc\u00df]", "", candidate)
                    if len(cleaned) < 3:
                        continue
                    if cleaned not in seen:
                        seen.add(cleaned)
                        yield cleaned

    def suggest_verb_infinitive(
        self, lemma: str, surface: Optional[str]
    ) -> Optional[Tuple[str, LexemeData]]:
        current_freq = zipf_frequency(self._normalize_for_frequency(lemma), "de")
        for candidate in self._iter_infinitive_candidates(lemma, surface):
            key = (candidate, "VERB")
            lexeme = self._index.get(key)
            if not lexeme:
                form_keys = self._form_index.get(key)
                if form_keys:
                    for target_key in form_keys:
                        lexeme = self._index.get(target_key)
                        if lexeme:
                            break
            if not lexeme:
                continue
            base = lexeme.verb_forms.get("infinitive")
            if not base and lexeme.lemma.lower().endswith("en"):
                base = lexeme.lemma
            if not base:
                continue
            base_clean = self._normalize_for_frequency(base)
            if not base_clean or base_clean == self._normalize_for_frequency(lemma):
                continue
            base_freq = zipf_frequency(base_clean, "de")
            if base_freq <= current_freq + 0.15:
                continue
            return base, lexeme
        return None


class LibreTranslateClient:
    def __init__(self, url: Optional[str], api_key: Optional[str] = None) -> None:
        self.url = url.rstrip("/") if url else None
        self.api_key = api_key

    def translate(
        self, text: str, source: str = "de", target: str = "en"
    ) -> Optional[List[str]]:
        if not self.url:
            return None
        payload = {"q": text, "source": source, "target": target, "format": "text"}
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
    plural_alternatives: List[str] = field(default_factory=list)
    context_gender: Optional[str] = None
    context_number: Optional[str] = None
    plural_candidates: Counter[str] = field(default_factory=Counter)


class MultiSourceTranslator:
    def __init__(
        self,
        lexicon: WiktextractLexicon,
        mt_client: Optional[LibreTranslateClient] = None,
        *,
        max_variants: int = 2,
    ) -> None:
        self.lexicon = lexicon
        self.mt = mt_client
        self.max_variants = max_variants

    def translate_entry(self, entry: TermEntry) -> None:
        glosses_en, source = self.lexicon.get_dictionary_glosses(
            entry.lemma,
            entry.pos,
            max_n=self.max_variants,
        )
        if glosses_en:
            entry.translations = glosses_en[: self.max_variants]
            entry.translation_source = source or "dictionary-en"
            return

        if self.mt:
            lemma_translations = self._mt_text(entry.lemma, source="de", target="en")
            if lemma_translations:
                entry.translations = lemma_translations[: self.max_variants]
                entry.translation_source = "mt-de-en"
                return

        entry.translations = []
        entry.translation_source = None

    def _mt_text(self, text: str, *, source: str, target: str) -> Optional[List[str]]:
        if not self.mt:
            return None
        result = self.mt.translate(text, source=source, target=target)
        if not result:
            return None
        cleaned: List[str] = []
        seen: Set[str] = set()
        for item in result:
            if not isinstance(item, str):
                continue
            trimmed = item.strip()
            if not trimmed:
                continue
            key = trimmed.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(trimmed)
            if len(cleaned) >= self.max_variants:
                break
        return cleaned or None




def normalize_text(raw_text: str) -> str:
    text = ftfy.fix_text(raw_text)
    text = text.replace("\r\n", "\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"-\s*\n", "", text)
    text = re.sub(r"(?<=\s)[\-\u2010\u2011\u2012\u2013\u2212](?=\w)", "- ", text)
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
    text = token.text.strip()
    dash_chars = ('-', chr(0x2013), chr(0x2014))
    if text.startswith(dash_chars):
        return False
    stripped = text.strip(''.join(dash_chars))
    if any(char.isdigit() for char in text):
        return False
    if text.isupper() and len(text) > 1:
        return False
    if '-' in text:
        first_segment, _, rest_segment = text.partition('-')
        if len(first_segment) == 1 and rest_segment and rest_segment[0].isalpha() and first_segment.lower() == rest_segment[0].lower():
            return False
    if len(stripped) <= 4 and len({ch.lower() for ch in stripped if ch.isalpha()}) <= 2:
        return False
    if re.search(r'[A-Za-zÄÖÜäöüß]-[A-Za-zÄÖÜäöüß]-', text):
        return False
    if not stripped or not any(char.isalpha() for char in stripped):
        return False
    if token.is_punct:
        return False
    if token.pos_ == "ADV" and token.is_stop:
        return True
    return True



def normalize_gender_tag(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    mapping = {
        "masculine": "masculine",
        "feminine": "feminine",
        "neuter": "neuter",
        "m": "masculine",
        "f": "feminine",
        "n": "neuter",
        "masc": "masculine",
        "fem": "feminine",
        "neut": "neuter",
        "common": "common",
    }
    return mapping.get(raw.lower())


def normalize_number_tag(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    mapping = {
        "sing": "singular",
        "plur": "plural",
        "dual": "dual",
        "singular": "singular",
        "plural": "plural",
    }
    return mapping.get(raw.lower())


def infer_context_features(token) -> Tuple[Optional[str], Optional[str]]:
    gender = None
    number = None
    gender_tags = token.morph.get("Gender")
    if gender_tags:
        gender = normalize_gender_tag(gender_tags[0])
    number_tags = token.morph.get("Number")
    if number_tags:
        number = normalize_number_tag(number_tags[0])
    determiner_candidates = [child for child in token.children if child.pos_ == "DET"]
    neighbors = []
    if token.i > 0:
        try:
            neighbors.append(token.doc[token.i - 1])
        except IndexError:
            pass
    if token.i + 1 < len(token.doc):
        try:
            neighbors.append(token.doc[token.i + 1])
        except IndexError:
            pass
    for neighbor in neighbors:
        if neighbor.pos_ == "DET" and neighbor not in determiner_candidates:
            determiner_candidates.append(neighbor)
    for det in determiner_candidates:
        det_gender = det.morph.get("Gender")
        if det_gender:
            normalized = normalize_gender_tag(det_gender[0])
            if normalized:
                gender = normalized
        det_number = det.morph.get("Number")
        if det_number:
            normalized_number = normalize_number_tag(det_number[0])
            if normalized_number:
                number = normalized_number
        if gender and number:
            break
    if number == "plural":
        gender = None
    return gender, number


def canonicalize_lemma_casing(lemma: str, pos: str) -> str:
    if not lemma:
        return lemma
    if pos == "NOUN":
        return lemma[0].upper() + lemma[1:]
    return lemma.lower()


def prune_pos_variants(entries: List[TermEntry]) -> List[TermEntry]:
    best: Dict[str, TermEntry] = {}
    for entry in entries:
        key = entry.lemma.lower()
        priority = POS_PRIORITY.get(entry.pos, 0)
        existing = best.get(key)
        if not existing:
            best[key] = entry
            continue
        existing_priority = POS_PRIORITY.get(existing.pos, 0)
        if priority > existing_priority:
            best[key] = entry
            continue
        if priority == existing_priority and entry.frequency > existing.frequency:
            best[key] = entry
    return list(best.values())


def normalize_plural_form(form: str) -> str:
    cleaned = unicodedata.normalize("NFC", form.strip())
    cleaned = ARTICLE_PATTERN.sub("", cleaned)
    cleaned = cleaned.strip(" ,.;:'\"()")
    return cleaned


def select_plural(plurals: Sequence[str], lemma: str) -> Tuple[str, List[str]]:
    seen = []
    for plural in plurals:
        normalized = normalize_plural_form(plural)
        if not normalized:
            continue
        if normalized not in seen:
            seen.append(normalized)
    if not seen:
        fallback = normalize_plural_form(plurals[0]) if plurals else ""
        return fallback, []
    scored = []
    priority_map = {
        "-e": 5,
        "-en": 4,
        "-er": 4,
        "-¨er": 4,
        "-n": 3,
        "-": 2,
        "-s": 1,
    }
    for plural in seen:
        freq = zipf_frequency(plural, "de")
        display = derive_plural_display(lemma, plural) or ""
        priority = priority_map.get(display, 2)
        scored.append((priority, freq, -len(plural), plural))
    scored.sort(reverse=True)
    preferred = scored[0][3]
    alternatives = [plural for plural in seen if plural != preferred]
    return preferred, alternatives


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
    gender_norm = normalize_gender_tag(gender)
    mapping = {
        "masculine": "der",
        "feminine": "die",
        "neuter": "das",
        "common": "die",
    }
    if not gender_norm:
        return None
    return mapping.get(gender_norm)


def format_front(entry: TermEntry, *, annotate_pos: bool = False) -> str:
    if entry.pos == "NOUN":
        article = article_from_gender(entry.gender) or ""
        lemma = entry.lemma
        plural = entry.plural_display or entry.plural or "-"
        core = " ".join(part for part in (article, lemma) if part)
        base = f"{core}, {plural}"
    elif entry.pos == "VERB":
        if entry.verb_forms:
            base = f"{entry.lemma} ({entry.verb_forms})"
        else:
            base = entry.lemma
    else:
        base = entry.lemma
    if annotate_pos:
        return f"{base} [{entry.pos.lower()}]"
    return base


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
    translator: MultiSourceTranslator,
    seen_global: Set[str],
) -> Tuple[List[TermEntry], Set[str]]:
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
                raw_lemma = (
                    reconstruct_lemma(token) if token.pos_ == "VERB" else token.lemma_
                )
                pos = token.pos_
                lemma_candidate = raw_lemma.strip()
                lemma_candidate, lexeme = lexicon.resolve(lemma_candidate, pos, token.text)
                resolved_pos = lexeme.pos if lexeme else pos
                if resolved_pos != "VERB" or not lexeme or not lexeme.verb_forms.get("infinitive"):
                    suggestion = lexicon.suggest_verb_infinitive(lemma_candidate, token.text)
                    if suggestion:
                        lemma_candidate, lexeme = suggestion
                        resolved_pos = "VERB"
                lemma = canonicalize_lemma_casing(lemma_candidate, resolved_pos)
                lemma_key = lemma.lower()
                if lemma_key in seen_global:
                    continue
                entry_key = (lemma_key, resolved_pos)
                entry = entries.get(entry_key)
                if not entry:
                    entry = TermEntry(lemma=lemma, pos=resolved_pos, surface=token.text)
                    entry.example_sentence = sent.text.strip()
                    entry.example_cloze = make_cloze(entry.example_sentence, token.text)
                    entries[entry_key] = entry
                if pos == "NOUN":
                    context_gender, context_number = infer_context_features(token)
                    if context_gender and not entry.context_gender:
                        entry.context_gender = context_gender
                    if context_number and not entry.context_number:
                        entry.context_number = context_number
                    number_tags = token.morph.get("Number")
                    if number_tags and "Plur" in number_tags:
                        observed_plural = normalize_plural_form(token.text)
                        if observed_plural:
                            entry.plural_candidates[observed_plural] += 1
                entry.frequency += 1
    entry_list = prune_pos_variants(list(entries.values()))
    for entry in entry_list:
        lexeme = lexicon.lookup(entry.lemma, entry.pos)
        if lexeme:
            if entry.pos == "NOUN":
                if lexeme.genders:
                    entry.gender = normalize_gender_tag(lexeme.genders[0])
                if lexeme.plurals:
                    plural, alternatives = select_plural(lexeme.plurals, entry.lemma)
                    entry.plural = plural
                    entry.plural_alternatives = alternatives
                    entry.plural_display = derive_plural_display(entry.lemma, entry.plural)
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
        if entry.pos == "NOUN" and not entry.plural and entry.plural_candidates:
            ordered: List[str] = []
            for form, _ in entry.plural_candidates.most_common():
                if not form:
                    continue
                if form not in ordered:
                    ordered.append(form)
            if ordered:
                entry.plural = ordered[0]
                entry.plural_display = derive_plural_display(entry.lemma, entry.plural)
                entry.plural_alternatives = [alt for alt in ordered[1:] if alt != entry.plural]

        if entry.pos == "NOUN" and not entry.gender and entry.context_gender:
            entry.gender = entry.context_gender

        translator.translate_entry(entry)

    total_entries = len(entry_list)
    dict_en = sum(1 for entry in entry_list if entry.translation_source == "dictionary-en")
    mt_de_en = sum(1 for entry in entry_list if entry.translation_source == "mt-de-en")
    dict_none = max(total_entries - (dict_en + mt_de_en), 0)
    if total_entries:
        pct_en = dict_en / total_entries * 100
        pct_mt = mt_de_en / total_entries * 100
        pct_none = dict_none / total_entries * 100
    else:
        pct_en = pct_mt = pct_none = 0.0
    LOGGER.info(
        "Coverage %s: EN %.1f%% | MT(DE\u2192EN) %.1f%% | none %.1f%%",
        path.name,
        pct_en,
        pct_mt,
        pct_none,
    )
    if LOGGER.isEnabledFor(logging.DEBUG):
        examples = [
            e
            for e in entry_list
            if e.translation_source in {"dictionary-en", "mt-de-en"}
        ][:5]
        for ex in examples:
            LOGGER.debug("Example translation: %s [%s] -> %s (%s)", ex.lemma, ex.pos, ex.translations, ex.translation_source)

    seen_keys = {entry.lemma.lower() for entry in entry_list}
    return entry_list, seen_keys


def to_records(entries: List[TermEntry], book_label: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    base_fronts: Dict[str, Set[str]] = {}
    cache: Dict[int, str] = {}
    for idx, entry in enumerate(entries):
        base = format_front(entry)
        cache[idx] = base
        base_fronts.setdefault(base, set()).add(entry.pos)
    for idx, entry in enumerate(entries):
        base = cache[idx]
        annotate = len(base_fronts.get(base, set())) > 1
        front_value = format_front(entry, annotate_pos=annotate)
        records.append(
            {
                "front": front_value,
                "back": format_back(entry),
                "lemma": entry.lemma,
                "pos": entry.pos,
                "gender": entry.gender,
                "plural": entry.plural,
                "plural_display": entry.plural_display,
                "plural_alternatives": "; ".join(entry.plural_alternatives),
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
    if not (config.wiktextract_path and config.wiktextract_path.exists()):
        if config.wiktextract_path:
            LOGGER.warning(
                "Wiktextract dataset not found at %s; dictionary glosses will be unavailable.",
                config.wiktextract_path,
            )
        else:
            LOGGER.warning("Wiktextract dataset not configured; dictionary glosses will be unavailable.")
    nlp = load_model(config.force_model)
    lexicon = WiktextractLexicon(config.wiktextract_path)
    mt_client = LibreTranslateClient(
        config.mt_url or MT_DEFAULT_URL, config.mt_api_key or MT_API_KEY
    )
    translator = MultiSourceTranslator(
        lexicon,
        mt_client if getattr(mt_client, "url", None) else None,
        max_variants=2,
    )
    if not translator.mt:
        LOGGER.info("Machine translation disabled (no URL); using dictionary glosses only.")
    book_paths = sorted(config.input_dir.glob("*.txt"))
    if config.limit_books is not None:
        book_paths = book_paths[: config.limit_books]
    if not book_paths:
        raise SystemExit(f"No .txt files found in {config.input_dir}")
    seen_global: Set[str] = set()
    for idx, book_path in enumerate(book_paths, start=1):
        entries, seen = process_book(book_path, nlp, lexicon, translator, seen_global)
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
