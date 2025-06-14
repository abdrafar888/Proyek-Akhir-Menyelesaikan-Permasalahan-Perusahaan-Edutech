## Business Understanding

### Latar Belakang
Jaya Jaya Institut merupakan salah satu institusi pendidikan tinggi yang telah berdiri sejak tahun 2000. Selama bertahun-tahun, institusi ini berhasil meluluskan ribuan mahasiswa dengan reputasi yang baik di dunia kerja. Namun, di balik keberhasilan tersebut, terdapat permasalahan serius yang harus dihadapi, yaitu tingginya jumlah mahasiswa yang tidak menyelesaikan studi alias dropout.

Tingginya angka dropout ini tentu menjadi indikator performa akademik yang kurang ideal dan berdampak pada reputasi serta kualitas institusi. Untuk mengatasi hal ini, pihak manajemen Jaya Jaya Institut ingin mendeteksi lebih awal mahasiswa yang berpotensi mengalami dropout, sehingga mereka dapat diberikan intervensi atau bimbingan yang sesuai sejak dini.

Sebagai calon data scientist masa depan, Anda diminta untuk menganalisis data performa siswa yang telah disediakan oleh pihak kampus guna mengidentifikasi pola-pola umum yang berkaitan dengan dropout. Selain itu, Anda juga diminta untuk membuat dashboard visualisasi yang memudahkan pihak institusi dalam memonitor dan memahami kondisi akademik mahasiswanya secara berkala.

### Permasalahan (Problem Statement)
- Bagaimana distribusi dropout berdasarkan jenis kelamin, tingkat pendidikan orang tua, dan Berdasarkan Beasiswa
- Apakah terdapat karakteristik umum dari siswa yang dropout dibandingkan dengan yang menyelesaikan studi?
- Apakah terdapat pola-pola tertentu yang dapat dikenali secara visual maupun statistik dari data performa siswa?

### Tujuan
- Mengidentifikasi atribut-atribut siswa yang memiliki korelasi signifikan dengan kemungkinan dropout, seperti gender, Berdasarkan Beasiswa, dan pendidikan orang tua.
- Menyajikan visualisasi data yang informatif untuk membantu pihak kampus memahami faktor-faktor risiko dropout.
- Menyediakan insight berbasis data sebagai dasar pengambilan keputusan strategis untuk menekan angka dropout di masa depan.

## Cakupan Proyek

### Pekerjaan yang Dilakukan

Proyek ini bertujuan untuk memahami pola dan faktor yang memengaruhi tingginya tingkat dropout mahasiswa di Jaya Jaya Institut. Analisis dilakukan terhadap data historis mahasiswa yang mencakup atribut demografis, akademik, dan sosial-ekonomi. Adapun proses kerja proyek ini meliputi:

- Eksplorasi awal dan pemeriksaan kualitas data
- Pra-pemrosesan data (encoding, scaling, dll.)
- Pembangunan model klasifikasi dengan beberapa algoritma machine learning
- Evaluasi performa model dengan metrik akurasi, confusion matrix, dan classification report
- Penyusunan insight analitis dan rekomendasi strategis

### Batasan

- Analisis dilakukan pada dataset statis (snapshot), tanpa mempertimbangkan dinamika waktu (longitudinal).
- Dataset terbatas pada data internal mahasiswa tanpa melibatkan faktor eksternal (misal: kondisi ekonomi nasional, situasi keluarga, dll.).
- Fokus analisis adalah klasifikasi biner: dropout (1) vs. non-dropout (0), di mana label 'Dropout' menjadi fokus utama prediksi.

### Tujuan Proyek

- Mengidentifikasi faktor-faktor utama yang berkontribusi terhadap risiko dropout mahasiswa.
- Menyediakan insight visual dan numerik yang dapat digunakan manajemen untuk pengambilan keputusan berbasis data.
- Membangun model klasifikasi yang andal untuk membantu institusi mendeteksi mahasiswa berisiko tinggi dropout secara otomatis.
- Memberikan saran strategis yang dapat diterapkan oleh pihak kampus untuk menurunkan angka dropout.

### Output Proyek

- Dataset yang telah dibersihkan (data_bersih.csv)
- Visualisasi evaluasi model
- Laporan klasifikasi dari berbagai algoritma (accuracy, precision, recall, f1-score)
- Insight dan rekomendasi strategi penurunan angka dropout berdasarkan hasil model klasifikasi

## Persiapan Proyek

### 1. Sumber Data

Dataset yang digunakan berisi informasi demografis dan pekerjaan karyawan PT Jaya Jaya Maju. File dapat ditemukan pada lokasi berikut:

- **File lokal:** `data.csv` 
- **Google Drive:** [Link ke spreadsheet](https://docs.google.com/spreadsheets/d/16Rp-Pr4b_3LHorY5dnO7oXjcb2nlKdfdV5-BLbtTFyw/edit?usp=sharing)

### 2. Pembuatan dan Aktivasi Virtual Environment

Untuk memastikan kestabilan dan konsistensi lingkungan kerja, ikuti langkah-langkah berikut:

#### a. Buat virtual environment

```bash
python -m venv venv
```

#### b. Aktifkan virtual environment

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

