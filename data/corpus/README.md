# Corpus Dataset — Naze 1.0 (M-010)

**Status:** INTEGRATED; seluruh gate DECISION-023 terpenuhi pada model final (2026-10-13).
**Source of Truth:** DECISION-024 (OD-112) | **Manifest:** corpus_manifest.json v2.0.0 (diselaraskan dengan keadaan aktual pada audit ID-BASIC-001, 2026-10-13)

## Isi Aktual (terverifikasi listing repo 2026-10-13)

| Berkas | Ukuran (byte) | Peran | Lisensi |
|---|---|---|---|
| umumnya/wikipedia_id.txt | 5.282.032 | 71,5% — bahasa Indonesia umum | CC-BY-SA 3.0 (Wikipedia Indonesia) |
| perintah/perintah_asisten.txt | 2.101.229 | 28,4% — perintah asisten format kalimat | Proprietary — Naze Project (sintetis) |
| editing/instruksi_editing.txt | 4.442 | 0,06% — instruksi editing | Proprietary — Naze Project (sintetis) |
| data/korpus/perintah_tingkat_1.txt | 2.774 | Korpus aux: pemetaan perintah->intent (benchmark NazeIO + rehearsal M-010) | Proprietary — Naze Project (sintetis) |

Catatan: berkas `berita_id.txt` yang dahulu tercantum pada struktur di dokumen lama tidak pernah ada di repo. Komposisi aktual menyimpang dari target D-024 (perintah 28,4% vs 20%; editing 0,06% vs 10%) — dilaporkan apa adanya; model akhir tetap memenuhi seluruh gate.

## Splits (DONE)
train/val/holdout = 80/10/10, line-based deterministic shuffle seed 42: `splits/train.txt` (5.859.468 token), `splits/val.txt` (749.894), `splits/holdout.txt` (773.864; TIDAK BOLEH untuk training/hyperparameter tuning). Detail + sha256: `splits/split_manifest.json`.

## Tokenisasi
ByteTokenizer (DECISION-010/015), vocab 256 fixed, UTF-8 lossless, tanpa normalisasi.

## Fingerprint
Tercatat di corpus_manifest.json. Status per berkas: wikipedia VERIFIED; instruksi_editing VERIFIED; perintah_asisten — hash versi pra-ekspansi tercatat, hash berkas aktual (2.101.229 byte) dihitung validator ID-BASIC-001 (IDB-T003).

## Catatan Pemeliharaan
Cache token ID (`corpus_all_ids.npy`, `splits/*_ids.npy`) dihapus dari working tree (2026-10-13) karena tidak direferensikan kode mana pun (search_code: 0 hasil); pipeline meng-encode teks saat runtime. Blob lama tetap ada di riwayat git; pembersihan riwayat memerlukan force-push dan tidak dilakukan.
