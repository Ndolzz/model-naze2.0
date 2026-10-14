# Arsip Korpus Warisan (data/korpus/)

**Task:** M011-T002 (M-011 pekerjaan rumah)
**Keputusan owner:** 2026-10-14 — konten diarsipkan verbatim ke direktori ini, kemudian direktori warisan data/korpus/ dihapus.
**Req:** REQ-105, REQ-304

## Asal

Direktori data/korpus/ adalah sisa pra-penataan struktur data/corpus/ (M-010). Berisi satu file:

- perintah_tingkat_1.txt (2.774 byte) — 50 pasangan format "perintah: {teks}" / "intent: {intent}" untuk Naze Engine (nazeio): pengenalan perintah lokal tanpa internet.

## Analisis Duplikasi (AC T002)

- **Format berbeda dari korpus aktif.** data/corpus/perintah/perintah_asisten.txt (2,0 MB) memakai format baris perintah tanpa label (mis. "buka aplikasi whatsapp"), bukan format perintah:/intent:.
- **Cakupan semantik tergantikan.** Seluruh intent warisan (buka_aplikasi, buka_pengaturan, pasang_pengingat, nyalakan_bluetooth, matikan_bluetooth, status_bluetooth, tangkap_layar, buka_maps, cari_maps, navigasi_maps, telepon_kontak, telepon_nomor, jam, hari, tanggal, baterai, hitung_hari, hapus_riwayat) tercakup secara semantik dalam korpus perintah aktif dan/atau benchmark NazeIO.
- **Benchmark mempertahankan turunannya.** data/benchmark/commands.json berisi 20 entri benchmark yang diturunkan dari file ini; field source_format telah diperbarui mengarah ke arsip ini.
- **Tidak ada data unik yang hilang.** Seluruh konten file diarsipkan verbatim di direktori ini; pemeriksaan cakupan dilakukan sebelum penghapusan (commit arsip mendahului commit hapus).

## Dampak

Tidak ada. File ini tidak direferensikan pipeline training/data aktif (korpus aktif: data/corpus/). Referensi satu-satunya (commands.json field source_format) telah diperbarui pada commit arsip ini.
