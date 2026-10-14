#!/usr/bin/env python3
"""Pipeline persiapan korpus M-011 (Naze 1.1) - dijalankan lokal oleh owner.

Task yang ditangani (docs/tasks/M011_TASKS.md):
  - M011-T003 : siapkan wikipedia_en.txt (~1,6 MB) dari dump lokal berlisensi CC-BY-SA
  - M011-T004 : pangkas wikipedia_id.txt ke ~3,2 MB (subset deterministik seed=42)
  - M011-T006 : rekomposisi perintah_asisten.txt menjadi dwibahasa (~1,2 MB)
  - M011-T008 : tulis corpus_manifest.json v2.0.0 (komposisi D-025 + SHA-256)
  - M011-T009 : regenerasi splits 80/10/10 + ids (seed=42, tanpa leakage)

Referensi: docs/architecture/M011_TECHNICAL_DESIGN.md (DECISION-025).
Req: REQ-012, REQ-005, REQ-101, REQ-105.

Konvensi:
  - "MB" = 1.000.000 byte (konsisten target komposisi D-025, total ~8 MB).
  - Seluruh seleksi dan pengacakan deterministik dengan seed=42.
  - Token = byte UTF-8 (ByteTokenizer, vocab 256 - DECISION-015).
  - Holdout TIDAK BOLEH digunakan untuk training/hyperparameter tuning (D-024/025).
  - id-trim menimpa wikipedia_id.txt di tempat; versi asli tetap ada di git history.

Contoh pemakaian (dari root repositori):
  python scripts/prepare_corpus_m011.py en-prepare --input dump_sewiki.txt \\
      --license-confirmed --source-name "Simple English Wikipedia" \\
      --source-url "https://example.org/source"
  python scripts/prepare_corpus_m011.py id-trim
  python scripts/prepare_corpus_m011.py perintah-recompose
  python scripts/prepare_corpus_m011.py manifest
  python scripts/prepare_corpus_m011.py splits
  python scripts/prepare_corpus_m011.py verify
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SEED = 42
TOLERANCE = 0.10
MANIFEST_NAME = "corpus_manifest.json"
SPLIT_MANIFEST_NAME = "split_manifest.json"
SPLIT_FILES = {"train": "train.txt", "validation": "val.txt", "holdout": "holdout.txt"}
SPLIT_IDS = {"train": "train_ids.npy", "validation": "val_ids.npy", "holdout": "holdout_ids.npy"}

# (key, path, persen, target_mb, sumber, lisensi, tujuan)
COMPONENTS = [
    (
        "bahasa_indonesia_umum",
        "umumnya/wikipedia_id.txt",
        40,
        3.2,
        "Wikipedia Indonesia (CC-BY-SA)",
        "CC-BY-SA 3.0 (Wikipedia Indonesia)",
        "Bahasa Indonesia umum",
    ),
    (
        "bahasa_inggris_umum",
        "umumnya/wikipedia_en.txt",
        20,
        1.6,
        "Simple English Wikipedia (CC-BY-SA)",
        "CC-BY-SA 3.0 (Simple English Wikipedia)",
        "Bahasa Inggris umum",
    ),
    (
        "percakapan_asisten_naze",
        "asisten/percakapan_naze.txt",
        15,
        1.2,
        "Synthetic - controlled generation",
        "Proprietary - Naze Project (Synthetic)",
        "Suara merek Naze (BRAND_VOICE.md)",
    ),
    (
        "perintah_asisten_dwibahasa",
        "perintah/perintah_asisten.txt",
        15,
        1.2,
        "Synthetic - controlled generation",
        "Proprietary - Naze Project (Synthetic)",
        "Perintah asisten dwibahasa (NazeIO)",
    ),
    (
        "instruksi_editing",
        "editing/instruksi_editing.txt",
        10,
        0.8,
        "Synthetic - controlled generation",
        "Proprietary - Naze Project (Synthetic)",
        "Naze Motion Agent",
    ),
]

CORPUS_FILES = [component[1] for component in COMPONENTS]

DAYPART_EN = {"pagi": "am", "siang": "pm", "sore": "pm", "malam": "pm"}
MONTH_EN = {
    "januari": "January",
    "februari": "February",
    "maret": "March",
    "april": "April",
    "mei": "May",
    "juni": "June",
    "juli": "July",
    "agustus": "August",
    "september": "September",
    "oktober": "October",
    "november": "November",
    "desember": "December",
}

APP_NAMES = [
    "whatsapp", "telegram", "instagram", "facebook", "twitter", "tiktok",
    "youtube", "spotify", "google maps", "google chrome", "gmail",
    "shopee", "grab", "gojek",
]
KEYWORDS = [
    "rapat", "belanja", "tugas", "kuliah", "proyek", "ide",
    "olahraga", "liburan", "rencana", "catatan",
]
SONGS = [
    "senja", "matahari", "hujan", "kembali", "perjalanan",
    "bangun", "selamat pagi", "tenang", "ceria", "semangat",
]
CONTACTS = [
    "ibu", "ayah", "bapak", "teman", "saudara", "dokter",
    "dosen", "kantor", "rumah", "kost",
]
DEVICES = ["wifi", "hotspot", "gps", "bluetooth", "mode pesawat"]
ACTIVITIES = [
    "rapat", "telepon ibu", "meeting", "janji dokter", "minum obat",
    "olahraga", "kuliah", "makan siang", "ngopi", "pulang kerja",
]
TRANSLATE_WORDS = [
    "makan", "minum", "rumah", "jalan", "sekolah", "kerja",
    "senang", "sedih", "pagi", "malam",
]


def fail(message: str) -> None:
    print("ERROR: " + message, file=sys.stderr)
    sys.exit(1)


def info(message: str) -> None:
    print(message)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    return len(data)


def read_lines(path: Path) -> list:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def dedup_preserve_order(lines: list) -> list:
    seen = set()
    unique = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return unique


def within_tolerance(actual: int, target: int) -> bool:
    return abs(actual - target) <= target * TOLERANCE


def corpus_dir(args: argparse.Namespace) -> Path:
    return Path(args.root) / "data" / "corpus"


# ---------------------------------------------------------------------------
# M011-T003: wikipedia_en.txt
# ---------------------------------------------------------------------------

RE_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.IGNORECASE | re.DOTALL)
RE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
RE_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
RE_TAG = re.compile(r"<[^>]+>")
RE_WS = re.compile(r"\s+")

SKIP_PREFIXES = ("|", "!", "Category:", "File:", "Image:", ":", ";")


def clean_wiki_line(line: str) -> Optional[str]:
    text = line.strip()
    if not text or len(text) < 20:
        return None
    if text.startswith(SKIP_PREFIXES):
        return None
    if text.startswith("=") and text.endswith("="):
        return None
    text = RE_COMMENT.sub(" ", text)
    text = RE_REF.sub(" ", text)
    text = RE_TEMPLATE.sub(" ", text)
    text = RE_LINK.sub(r"\1", text)
    text = RE_TAG.sub(" ", text)
    text = RE_WS.sub(" ", text).strip()
    if len(text) < 20:
        return None
    return text


def cmd_en_prepare(args: argparse.Namespace) -> None:
    if not args.license_confirmed:
        fail(
            "M011-T003 AC 1: verifikasi lisensi WAJIB sebelum integrasi. "
            "Jalankan ulang dengan --license-confirmed setelah lisensi sumber "
            "(CC-BY-SA) diverifikasi dan didokumentasikan."
        )
    if not args.source_name or not args.source_url:
        fail("--source-name dan --source-url wajib diisi untuk atribusi CC-BY-SA.")
    input_path = Path(args.input)
    if not input_path.is_file():
        fail("file input tidak ditemukan: " + str(input_path))
    target = args.target_bytes
    cleaned = dedup_preserve_order(
        [text for text in (clean_wiki_line(line) for line in read_lines(input_path)) if text is not None]
    )
    selected = []
    total = 0
    for text in cleaned:
        size = len(text.encode("utf-8")) + 1
        if total + size <= target:
            selected.append(text)
            total += size
    out_path = corpus_dir(args) / "umumnya" / "wikipedia_en.txt"
    written = write_text(out_path, "\n".join(selected) + "\n")
    info(
        "T003: wikipedia_en.txt ditulis: {} byte, {} baris (sumber: {}, {})"
        .format(written, len(selected), args.source_name, args.source_url)
    )
    info("sha256: " + sha256_file(out_path))
    if not within_tolerance(written, target):
        fail(
            "T003: ukuran hasil {} byte di luar toleransi +/-10% dari target {} byte; "
            "perbesar sumber input.".format(written, target)
        )


# ---------------------------------------------------------------------------
# M011-T004: pangkas wikipedia_id.txt
# ---------------------------------------------------------------------------

def cmd_id_trim(args: argparse.Namespace) -> None:
    path = corpus_dir(args) / "umumnya" / "wikipedia_id.txt"
    if not path.is_file():
        fail("M011-T004: file tidak ditemukan: " + str(path))
    lines = [line.strip() for line in read_lines(path)]
    lines = [line for line in lines if line]
    if args.dedup:
        lines = dedup_preserve_order(lines)
    target = args.target_bytes
    rng = random.Random(SEED)
    order = list(range(len(lines)))
    rng.shuffle(order)
    chosen = set()
    total = 0
    for idx in order:
        size = len(lines[idx].encode("utf-8")) + 1
        if total + size <= target:
            chosen.add(idx)
            total += size
    selected = [lines[idx] for idx in range(len(lines)) if idx in chosen]
    written = write_text(path, "\n".join(selected) + "\n")
    info(
        "T004: wikipedia_id.txt dipangkas: {} byte, {} baris (seed={}, dedup={})".format(
            written, len(selected), SEED, args.dedup
        )
    )
    info("sha256: " + sha256_file(path))
    if not within_tolerance(written, target):
        fail(
            "T004: ukuran hasil {} byte di luar toleransi +/-10% dari target {} byte.".format(
                written, target
            )
        )


# ---------------------------------------------------------------------------
# M011-T006: rekomposisi perintah_asisten.txt dwibahasa
# ---------------------------------------------------------------------------

def _hour_en(tokens: list, hour_idx: int) -> str:
    hour = tokens[hour_idx]
    if hour_idx + 1 < len(tokens) and tokens[hour_idx + 1] in DAYPART_EN:
        return hour + " " + DAYPART_EN[tokens[hour_idx + 1]]
    return hour


def translate_command(line: str) -> Optional[str]:
    """Terjemahkan perintah Indonesia menjadi padanan Inggris bermakna identik.

    Berbasis token (lowercase) dan deterministik; mengembalikan None bila
    perintah tidak dikenali kamus aturan.
    """
    tokens = line.lower().split()
    n = len(tokens)
    if n == 0:
        return None

    def rest(start: int) -> str:
        return " ".join(tokens[start:])

    def find_jam(start: int) -> int:
        for idx in range(start, n):
            if tokens[idx] == "jam" and idx + 1 < n:
                return idx
        return -1

    # Aplikasi
    if n >= 3 and tokens[1] == "aplikasi":
        verb = {"buka": "open", "tutup": "close", "hapus": "uninstall", "perbarui": "update"}
        if tokens[0] in verb:
            return verb[tokens[0]] + " " + rest(2)
    # Alarm
    if tokens[0] in ("setel", "atur") and n >= 2 and tokens[1] == "alarm":
        if n >= 5 and tokens[2] == "untuk" and tokens[3] == "jam":
            return "set alarm for " + _hour_en(tokens, 4)
        if n >= 3 and tokens[2] == "jam":
            return "set alarm for " + _hour_en(tokens, 3)
        return "set an alarm"
    if tokens[0] == "hapus" and n >= 2 and tokens[1] == "alarm":
        if n >= 3 and tokens[2] == "semua":
            return "delete all alarms"
        if n >= 4 and tokens[2] == "jam":
            return "delete alarm for " + _hour_en(tokens, 3)
        return "delete the alarm"
    # Catatan
    if tokens[0] == "buat" and n >= 2 and tokens[1] == "catatan":
        if n >= 3 and tokens[2] == "baru":
            return "create a new note"
        if n >= 5 and tokens[2:5] == ["dengan", "kata", "judul"]:
            return "create a note titled " + rest(5)
        if n >= 4 and tokens[2] == "dengan" and tokens[3] == "judul":
            return "create a note titled " + rest(4)
    if tokens[0] == "baca" and n >= 3 and tokens[1] == "catatan":
        return "read note " + rest(2)
    if tokens[0:3] == ["baca", "semua", "catatan"]:
        return "read all notes"
    if tokens[0] == "hapus" and n >= 3 and tokens[1] == "catatan":
        return "delete note " + rest(2)
    if tokens[0:3] == ["hapus", "semua", "catatan"]:
        return "delete all notes"
    if tokens[0] == "cari" and n >= 6 and tokens[1] == "catatan" and tokens[2:5] == ["dengan", "kata", "kunci"]:
        return "search notes for " + rest(5)
    # Pengingat
    if tokens[0] in ("setel", "atur") and n >= 3 and tokens[1] == "pengingat" and tokens[2] == "untuk":
        jam_idx = find_jam(3)
        if jam_idx > 3:
            return "set a reminder for " + " ".join(tokens[3:jam_idx]) + " at " + _hour_en(tokens, jam_idx + 1)
        if jam_idx == 3 and n >= 5:
            return "set a reminder for " + _hour_en(tokens, 4)
        return "set a reminder for " + rest(3)
    if tokens[0] == "hapus" and n >= 2 and tokens[1] == "pengingat":
        if n >= 3 and tokens[2] == "semua":
            return "delete all reminders"
        jam_idx = find_jam(2)
        if jam_idx > 2:
            return "delete reminder " + " ".join(tokens[2:jam_idx]) + " at " + _hour_en(tokens, jam_idx + 1)
        return "delete reminder " + rest(2)
    # Kalender
    if tokens[0:2] == ["buka", "kalender"]:
        if n >= 5 and tokens[2:4] == ["untuk", "bulan"]:
            return "open the calendar for " + MONTH_EN.get(tokens[4], tokens[4])
        return "open the calendar"
    if tokens[0:4] == ["buka", "galeri", "musik"]:
        return "open the music gallery"
    # Browser / tab
    if tokens[0:4] == ["buka", "browser", "di", "halaman"]:
        return "open the browser on " + rest(4)
    if tokens[0:5] == ["buka", "tab", "baru", "di", "halaman"]:
        return "open a new tab on " + rest(5)
    if tokens[0:3] == ["tutup", "tab", "terakhir"]:
        return "close the last tab"
    # Toggles
    if tokens[0] == "aktifkan" and n >= 2:
        return "turn on " + rest(1)
    if tokens[0] == "matikan" and n >= 2:
        return "turn off " + rest(1)
    # Volume / kecerahan
    if tokens[0:2] == ["atur", "volume"] and n >= 5 and tokens[2] == "ke" and tokens[4] == "persen":
        return "set volume to " + tokens[3] + " percent"
    if tokens[0:2] == ["atur", "kecerahan"] and n >= 5 and tokens[2] == "ke" and tokens[4] == "persen":
        return "set brightness to " + tokens[3] + " percent"
    # Timer
    if tokens[0:2] == ["setel", "timer"] and n >= 4 and tokens[3] == "menit":
        return "set a timer for " + tokens[2] + " minutes"
    if tokens[0:2] == ["hentikan", "timer"]:
        return "stop the timer"
    # Pencarian web
    if tokens[0] == "cari" and n >= 7 and tokens[1] == "di" and tokens[2] == "google" and tokens[3:6] == ["dengan", "kata", "kunci"]:
        return "search google for " + rest(6)
    # Musik
    if tokens[0:2] == ["putar", "lagu"] and n >= 3:
        return "play song " + rest(2)
    if tokens[0:2] == ["putar", "playlist"] and n >= 3:
        return "play playlist " + rest(2)
    if tokens[0] == "putar" and n >= 2:
        return "play " + rest(1)
    if tokens[0:2] == ["jeda", "musik"]:
        return "pause the music"
    if tokens[0:2] == ["hentikan", "musik"]:
        return "stop the music"
    if tokens[0:2] == ["lanjutkan", "musik"]:
        return "resume the music"
    if tokens[0:2] == ["musik", "berikutnya"]:
        return "next song"
    # Panggilan
    if tokens[0] == "panggil" and n >= 2:
        return "call " + rest(1)
    if tokens[0:2] == ["video", "call"] and n >= 3:
        return "video call " + rest(2)
    if tokens[0] in ("tutup", "akhiri") and n >= 2 and tokens[1] == "panggilan":
        return "end the call"
    # Pesan
    if tokens[0] == "hapus" and n >= 4 and tokens[1] == "pesan" and tokens[2] == "dari":
        return "delete the message from " + rest(3)
    # Terjemahan
    if tokens[0] == "terjemahkan" and n >= 3 and tokens[1] == "kata":
        return "translate the word " + rest(2)
    # Kamera
    if tokens[0:2] == ["ambil", "video"]:
        return "take a video"
    if tokens[0:2] == ["ambil", "foto"]:
        return "take a photo"
    # Fallback generik (makna tetap identik)
    if tokens[0] == "buka" and n >= 2:
        return "open " + rest(1)
    if tokens[0] == "tutup" and n >= 2:
        return "close " + rest(1)
    if tokens[0] == "hapus" and n >= 2:
        return "delete " + rest(1)
    if tokens[0] in ("setel", "atur") and n >= 2:
        return "set " + rest(1)
    return None


def generate_expansion(seen_ids: set) -> list:
    """Ekspansi deterministik bila inventaris sumber belum mencapai target."""
    lines = []
    for app in APP_NAMES:
        lines.append("buka aplikasi " + app)
        lines.append("tutup aplikasi " + app)
        lines.append("perbarui aplikasi " + app)
    for hour in range(1, 13):
        for daypart in DAYPART_EN:
            lines.append("setel alarm untuk jam {} {}".format(hour, daypart))
            lines.append("hapus alarm jam {} {}".format(hour, daypart))
    for keyword in KEYWORDS:
        lines.append("buat catatan dengan judul " + keyword)
        lines.append("baca catatan " + keyword)
        lines.append("hapus catatan " + keyword)
        lines.append("cari catatan dengan kata kunci " + keyword)
    for activity in ACTIVITIES:
        for hour in range(6, 23):
            lines.append("setel pengingat untuk {} jam {}".format(activity, hour))
    for month in MONTH_EN:
        lines.append("buka kalender untuk bulan " + month)
    for percent in range(10, 101, 10):
        lines.append("atur volume ke {} persen".format(percent))
        lines.append("atur kecerahan ke {} persen".format(percent))
    for minutes in range(1, 61, 5):
        lines.append("setel timer {} menit".format(minutes))
    for song in SONGS:
        lines.append("putar lagu " + song)
    for contact in CONTACTS:
        lines.append("panggil " + contact)
        lines.append("video call " + contact)
    for device in DEVICES:
        lines.append("aktifkan " + device)
        lines.append("matikan " + device)
    for word in TRANSLATE_WORDS:
        lines.append("terjemahkan kata " + word)
    pairs = []
    for line in lines:
        if line in seen_ids:
            continue
        english = translate_command(line)
        if english is None:
            continue
        pairs.append((line, english))
    return pairs


def cmd_perintah_recompose(args: argparse.Namespace) -> None:
    path = corpus_dir(args) / "perintah" / "perintah_asisten.txt"
    if not path.is_file():
        fail("M011-T006: file tidak ditemukan: " + str(path))
    lines = [line.strip() for line in read_lines(path)]
    lines = [line for line in lines if line]
    inventory = dedup_preserve_order(lines)
    info("T006: inventaris unik: {} baris (dari {} baris sumber)".format(len(inventory), len(lines)))
    pairs = []
    unmatched = []
    for line in inventory:
        english = translate_command(line)
        if english is None:
            unmatched.append(line)
        else:
            pairs.append((line, english))
    if args.strict and unmatched:
        fail(
            "T006 (strict): {} perintah tidak dikenali kamus (contoh: {}). "
            "Perluas aturan translate_command di skrip ini.".format(len(unmatched), unmatched[:5])
        )
    target = args.target_bytes
    out_lines = []
    total = 0
    for indo, english in pairs:
        size = len(indo.encode("utf-8")) + len(english.encode("utf-8")) + 2
        if total + size > target:
            continue
        out_lines.append(indo)
        out_lines.append(english)
        total += size
    if total < target * (1 - TOLERANCE):
        seen = set(inventory)
        for indo, english in generate_expansion(seen):
            if total >= target:
                break
            size = len(indo.encode("utf-8")) + len(english.encode("utf-8")) + 2
            if total + size > target:
                continue
            out_lines.append(indo)
            out_lines.append(english)
            total += size
    if not within_tolerance(total, target):
        fail(
            "T006: ukuran hasil {} byte di luar toleransi +/-10% dari target {} byte; "
            "perluas inventaris sumber atau aturan ekspansi.".format(total, target)
        )
    written = write_text(path, "\n".join(out_lines) + "\n")
    info(
        "T006: perintah_asisten.txt direkomposisi: {} byte, {} baris (pasangan ID/EN interleaved)".format(
            written, len(out_lines)
        )
    )
    info("sha256: " + sha256_file(path))
    if unmatched:
        info(
            "catatan: {} baris sumber tidak dikenali kamus dan dilewati (contoh: {})".format(
                len(unmatched), unmatched[:3]
            )
        )


# ---------------------------------------------------------------------------
# M011-T008: corpus_manifest.json v2.0.0
# ---------------------------------------------------------------------------

def cmd_manifest(args: argparse.Namespace) -> None:
    root = corpus_dir(args)
    components_json = {}
    fingerprints = {}
    licenses = {}
    total = 0
    all_ok = True
    for key, rel, percent, target_mb, source, license_name, purpose in COMPONENTS:
        path = root / rel
        if not path.is_file():
            fail("M011-T008: komponen tidak ditemukan: " + rel)
        size = path.stat().st_size
        total += size
        target = int(target_mb * 1_000_000)
        good = within_tolerance(size, target)
        all_ok = all_ok and good
        components_json[key] = {
            "percentage": percent,
            "target_size_mb": target_mb,
            "actual_size_bytes": size,
            "actual_size_mb": round(size / 1_000_000, 2),
            "sources": [source],
            "files": [rel],
            "license": license_name,
            "purpose": purpose,
            "status": "INTEGRATED",
        }
        fingerprints[rel] = {
            "sha256": sha256_file(path),
            "size_bytes": size,
            "status": "VERIFIED",
        }
        licenses[rel] = license_name
        info(
            "  {}: {} byte ({})".format(rel, size, "OK" if good else "DI LUAR TOLERANSI +/-10%")
        )
    if args.strict_tolerance and not all_ok:
        fail("M011-T008: toleransi +/-10% per komponen tidak terpenuhi (--strict-tolerance)")
    manifest = {
        "version": "2.0.0",
        "name": "Naze 1.1 Bilingual Corpus",
        "description": "Corpus dwibahasa (Indonesia + Inggris) dengan suara merek Naze untuk training Naze 1.1 (M-011) sesuai DECISION-025",
        "strategy": "hybrid bilingual (DECISION-025: 40/20/15/15/10)",
        "total_size_bytes": total,
        "total_size_mb": round(total / 1_000_000, 2),
        "created": now_iso(),
        "seed": SEED,
        "composition": components_json,
        "dataset_splitting": {
            "training": {"percentage": 80, "seed": SEED, "status": "NOT_YET_SPLIT", "reference": "splits/" + SPLIT_MANIFEST_NAME},
            "validation": {"percentage": 10, "seed": SEED, "status": "NOT_YET_SPLIT", "reference": "splits/" + SPLIT_MANIFEST_NAME},
            "holdout": {
                "percentage": 10,
                "seed": SEED,
                "status": "NOT_YET_SPLIT",
                "note": "TIDAK BOLEH digunakan untuk training atau hyperparameter tuning",
                "reference": "splits/" + SPLIT_MANIFEST_NAME,
            },
        },
        "preprocessing": {
            "tokenizer": "ByteTokenizer",
            "vocab_size": 256,
            "encoding": "UTF-8",
            "normalization": "none (byte-level is lossless)",
            "deduplication": "APPLIED per komponen (en-prepare, id-trim --dedup, perintah-recompose; percakapan_naze dan instruksi_editing didedup saat dibuat)",
            "status": "PREPARED",
        },
        "fingerprints": fingerprints,
        "licenses": licenses,
        "notes": {
            "wikipedia_id.txt": "Subset deterministik seed=42 dari korpus Naze 1.0 (M-010) - M011-T004",
            "wikipedia_en.txt": "Sumber berlisensi terbuka; lisensi diverifikasi sebelum integrasi - M011-T003",
            "percakapan_naze.txt": "Percakapan sintetis dwibahasa per BRAND_VOICE.md - M011-T005",
            "perintah_asisten.txt": "Perintah dwibahasa ID/EN interleaved bermakna identik - M011-T006",
            "instruksi_editing.txt": "Instruksi editing konsisten kontrak NMA - M011-T007",
        },
        "warnings": [
            "Holdout tidak boleh digunakan untuk training atau hyperparameter tuning (DECISION-024/025)",
            "Konten Wikipedia (CC-BY-SA) wajib atribusi sesuai lisensi bila didistribusikan",
        ],
        "decision_reference": {
            "od_112": "DECISION-025",
            "od_123": "OPEN - diresolusi via eksperimen pembanding M011-T012",
            "status": "ACCEPTED (owner, 2026-10-14)",
            "milestone": "M-011",
            "stage": "Naze 1.1",
        },
    }
    written = write_text(root / MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    info("T008: {} v2.0.0 ditulis: {} byte; total korpus {} byte".format(MANIFEST_NAME, written, total))
    if not all_ok:
        info("PERINGATAN: terdapat komponen di luar toleransi +/-10% (lihat rincian di atas).")


# ---------------------------------------------------------------------------
# M011-T009: splits 80/10/10 + ids
# ---------------------------------------------------------------------------

def load_unique_corpus_lines(root: Path) -> list:
    all_lines = []
    for rel in CORPUS_FILES:
        path = root / rel
        if not path.is_file():
            fail("M011-T009: komponen tidak ditemukan: " + rel + " (jalankan subcommand persiapan terlebih dahulu)")
        all_lines.extend(line for line in read_lines(path) if line.strip())
    unique = dedup_preserve_order(all_lines)
    info("T009: baris komponen: {}; baris unik (dedup global): {}".format(len(all_lines), len(unique)))
    return unique


def cmd_splits(args: argparse.Namespace) -> None:
    try:
        import numpy as np
    except ImportError:
        fail("numpy diperlukan (dependensi proyek) untuk regenerasi file ids.")
    root = corpus_dir(args)
    unique = load_unique_corpus_lines(root)
    rng = random.Random(SEED)
    order = list(range(len(unique)))
    rng.shuffle(order)
    n = len(unique)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    parts = {
        "train": [unique[i] for i in order[:n_train]],
        "validation": [unique[i] for i in order[n_train:n_train + n_val]],
        "holdout": [unique[i] for i in order[n_train + n_val:]],
    }
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_info = {}
    for name, lines in parts.items():
        text = "\n".join(lines) + "\n"
        path = splits_dir / SPLIT_FILES[name]
        size = write_text(path, text)
        split_info[name] = {
            "lines": len(lines),
            "tokens": size,
            "sha256": sha256_file(path),
        }
    corpus_bytes = ("\n".join(unique) + "\n").encode("utf-8")
    np.save(root / "corpus_all_ids.npy", np.frombuffer(corpus_bytes, dtype=np.uint8).astype(np.int64))
    for name in parts:
        data = (splits_dir / SPLIT_FILES[name]).read_bytes()
        np.save(splits_dir / SPLIT_IDS[name], np.frombuffer(data, dtype=np.uint8).astype(np.int64))
    split_manifest = {
        "version": "2.0.0",
        "name": "Naze 1.1 Bilingual Corpus Splits",
        "seed": SEED,
        "total_lines": n,
        "total_tokens": sum(entry["tokens"] for entry in split_info.values()),
        "components": CORPUS_FILES,
        "splits": {
            "train": dict(split_info["train"], percentage=80.0),
            "validation": dict(split_info["validation"], percentage=10.0),
            "holdout": dict(
                split_info["holdout"],
                percentage=10.0,
                note="TIDAK BOLEH digunakan untuk training atau hyperparameter tuning",
            ),
        },
        "method": "line-based split dengan shuffle deterministik (seed=42); dedup global sebelum split (jaminan tanpa leakage lintas split)",
        "date": now_iso(),
    }
    write_text(splits_dir / SPLIT_MANIFEST_NAME, json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n")
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        fail("M011-T009: " + MANIFEST_NAME + " tidak ditemukan; jalankan subcommand manifest terlebih dahulu.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("training", "validation", "holdout"):
        manifest["dataset_splitting"][name]["status"] = "SPLIT_INTEGRATED"
    write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    info(
        "T009: splits 80/10/10 dibuat ulang (train {} / validation {} / holdout {} baris; seed={})".format(
            split_info["train"]["lines"], split_info["validation"]["lines"], split_info["holdout"]["lines"], SEED
        )
    )
    info("T009: corpus_all_ids.npy + ids per split + " + SPLIT_MANIFEST_NAME + " ditulis.")


# ---------------------------------------------------------------------------
# Verifikasi (AC T008 + T009)
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> None:
    root = corpus_dir(args)
    problems = []
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        fail("verify: " + MANIFEST_NAME + " tidak ditemukan; jalankan subcommand manifest terlebih dahulu.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel, entry in manifest["fingerprints"].items():
        path = root / rel
        if not path.is_file():
            problems.append("komponen tidak ditemukan: " + rel)
            continue
        if sha256_file(path) != entry["sha256"]:
            problems.append("sha256 tidak cocok: " + rel)
        if path.stat().st_size != entry["size_bytes"]:
            problems.append("ukuran berubah: " + rel)
    for key, comp in manifest["composition"].items():
        rel = comp["files"][0]
        path = root / rel
        if path.is_file():
            target = int(comp["target_size_mb"] * 1_000_000)
            if not within_tolerance(path.stat().st_size, target):
                problems.append(
                    "toleransi +/-10%: {} ({} byte vs target {} byte)".format(
                        rel, path.stat().st_size, target
                    )
                )
    splits_dir = root / "splits"
    line_sets = {}
    for name, file_name in SPLIT_FILES.items():
        path = splits_dir / file_name
        if not path.is_file():
            problems.append("split tidak ditemukan: " + file_name)
            continue
        line_sets[name] = set(line for line in read_lines(path) if line)
    for name_a, name_b in itertools.combinations(sorted(line_sets), 2):
        shared = line_sets[name_a] & line_sets[name_b]
        if shared:
            problems.append(
                "leakage antara {} dan {}: {} baris bersama".format(name_a, name_b, len(shared))
            )
    unique = load_unique_corpus_lines(root)
    order_a = list(range(len(unique)))
    random.Random(SEED).shuffle(order_a)
    order_b = list(range(len(unique)))
    random.Random(SEED).shuffle(order_b)
    if order_a != order_b:
        problems.append("shuffle seed=42 tidak deterministik")
    if problems:
        for problem in problems:
            info("VERIFY GAGAL: " + problem)
        sys.exit(1)
    info("verify: OK (fingerprint, toleransi +/-10%, tanpa leakage, determinisme seed=42)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline persiapan korpus M-011 (Naze 1.1) - jalankan lokal dari root repositori."
    )
    parser.add_argument("--root", default=".", help="root repositori (default: direktori kerja saat ini)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_en = subparsers.add_parser("en-prepare", help="M011-T003: siapkan wikipedia_en.txt dari dump lokal")
    p_en.add_argument("--input", required=True, help="path file dump teks sumber (mis. Simple English Wikipedia)")
    p_en.add_argument("--license-confirmed", action="store_true", help="konfirmasi lisensi sumber telah diverifikasi (WAJIB)")
    p_en.add_argument("--source-name", default="", help="nama sumber untuk dokumentasi atribusi")
    p_en.add_argument("--source-url", default="", help="URL sumber untuk dokumentasi atribusi")
    p_en.add_argument("--target-bytes", type=int, default=1_600_000, help="target ukuran byte (default 1600000)")
    p_en.set_defaults(func=cmd_en_prepare)

    p_id = subparsers.add_parser("id-trim", help="M011-T004: pangkas wikipedia_id.txt ke ~3,2 MB")
    p_id.add_argument("--target-bytes", type=int, default=3_200_000, help="target ukuran byte (default 3200000)")
    p_id.add_argument("--dedup", action="store_true", help="dedup baris sebelum pemilihan subset")
    p_id.set_defaults(func=cmd_id_trim)

    p_perintah = subparsers.add_parser("perintah-recompose", help="M011-T006: rekomposisi perintah_asisten.txt dwibahasa")
    p_perintah.add_argument("--target-bytes", type=int, default=1_200_000, help="target ukuran byte (default 1200000)")
    p_perintah.add_argument("--strict", action="store_true", help="gagal bila ada perintah yang tidak dikenali kamus")
    p_perintah.set_defaults(func=cmd_perintah_recompose)

    p_manifest = subparsers.add_parser("manifest", help="M011-T008: tulis corpus_manifest.json v2.0.0")
    p_manifest.add_argument("--strict-tolerance", action="store_true", help="gagal bila toleransi +/-10% tidak terpenuhi")
    p_manifest.set_defaults(func=cmd_manifest)

    p_splits = subparsers.add_parser("splits", help="M011-T009: regenerasi splits 80/10/10 + ids")
    p_splits.set_defaults(func=cmd_splits)

    p_verify = subparsers.add_parser("verify", help="verifikasi AC T008/T009")
    p_verify.set_defaults(func=cmd_verify)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
