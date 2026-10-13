# PROVENANCE — Dataset ID-BASIC-001

Tanggal investigasi: 2026-10-13. Seluruh sumber eksternal diakses dan diverifikasi langsung pada tanggal tersebut.

## Keputusan

Dataset ID-BASIC-001 menggunakan 100% contoh buatan sendiri (fallback resmi sesuai DECISION-025). Tidak ada data pihak ketiga yang diimpor. Tiga kandidat sumber eksternal diinvestigasi dan dikecualikan dengan alasan berikut.

## 1. IndoNLU — DIKECUALIKAN

- Sumber resmi: https://github.com/indobenchmark/indonlu
- Bukti lisensi: berkas LICENSE pada repositori resmi berlisensi Apache-2.0 (dibaca langsung dari repositori). Klaim lisensi MIT pada halaman Hugging Face tidak konsisten dengan repositori resmi.
- Alasan eksklusi: lisensi yang benar-benar berlaku untuk berkas data tidak dapat diverifikasi secara tegas (lisensi kode tidak otomatis berlaku untuk data), dan berkas data tidak tersedia langsung di repositori resmi.
- Bagian yang digunakan: tidak ada.

## 2. Kateglo — DIKECUALIKAN

- Sumber resmi: https://github.com/ivanlanin/kateglo
- Bukti lisensi: README resmi menyatakan isi Kateglo berlisensi CC-BY-NC-SA 3.0 (nonkomersial, share-alike); data Pusba memerlukan izin sublisensi.
- Alasan eksklusi: ketentuan NC-SA tidak kompatibel dengan kebutuhan distribusi dataset pada repositori ini tanpa penelusuran atribusi sublisensi yang memadai.
- Bagian yang digunakan: tidak ada.

## 3. Wordnet Bahasa / Open Multilingual Wordnet — DIKECUALIKAN

- Sumber resmi: https://github.com/limaginaire/Bahasa-Wordnet (commit eece9d9fb1e2c34aa10234dad2d33c261abf9658).
- Bukti lisensi: berkas LICENSE pada commit tersebut berlisensi MIT (terverifikasi).
- Alasan eksklusi:
  - Berkas lema wn-msa-all.tab (17.003.012 byte) melebihi batas pengambilan berkas 1 MB; hasil unduhan melalui raw.githubusercontent terpotong pada 32.793 karakter (sekitar 0,2% isi berkas).
  - Sumber mirror wordnet-extras memakai offset wn31 yang tidak kompatibel dengan wn30 sehingga penggabungan definisi akan menghasilkan data salah.
  - Mirror open-language/id-wordnet menyertakan berkas berlabel MIT dengan atribusi tidak sahih (copyright Google/AngularJS).
  - Berkas wn-ind-def.tab (880.989 karakter, 12.668 definisi offset ke definisi) berhasil diambil penuh, tetapi tidak memuat lema sehingga tidak dapat diberi label kata.
- Bagian yang digunakan: tidak ada.

## Contoh buatan sendiri

- Seluruh contoh pada data/idbasic/*.txt dan data/idbasic/benchmark/*.json ditulis khusus untuk repositori ini.
- Metode penyusunan: kurasi manual per kategori (vocabulary, grammar, sentence_structure, punctuation, contextual_meaning) dengan templat prompt tetap dan jawaban referensi eksplisit.
- Transformasi: tidak ada; contoh dibuat langsung pada format blok dua baris pipeline.
- Distribusi: seluruh contoh mengikuti lisensi repositori; tidak ada batasan atribusi pihak ketiga.
- Redistribusi: aman karena tidak menautkan data pihak ketiga.
