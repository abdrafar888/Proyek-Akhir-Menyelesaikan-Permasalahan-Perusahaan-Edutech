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

- **File utama:** `data.csv` — data mentah asli dari institusi
- **File bersih (hasil preprocessing):** `data_bersih.csv`
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

#### c. Instalasi library yang dibutuhkan

```bash
pip install -r requirements.txt
```
## Business Dashboard

Dashboard ini dibuat untuk membantu manajemen **Jaya Jaya Institut** dalam memahami lebih dalam pola dan karakteristik mahasiswa yang mengalami dropout. Visualisasi interaktif disusun menggunakan **Looker Studio**, dan menampilkan metrik kunci yang relevan dengan isu retensi mahasiswa.

### Tujuan Dashboard

- Menyajikan data dropout mahasiswa secara ringkas dan mudah dipahami.
- Mengidentifikasi pola dropout berdasarkan demografi dan latar belakang pendidikan.
- Menjadi dasar pengambilan keputusan dalam merancang strategi pencegahan dropout.

### Tampilan Dashboard

![Dashboard Jaya Jaya Institut](https://github.com/abdrafar888/Proyek-Akhir-Menyelesaikan-Permasalahan-Perusahaan-Edutech/blob/main/abdul_rafar_1oFX_Dashboard.png)

### Komponen Visualisasi

- **Total Mahasiswa, Mahasiswa, dan Mahasiswi**  
  Menampilkan distribusi total secara keseluruhan dan berdasarkan jenis kelamin.

- **Persentase Mahasiswa Dropout vs Tidak Dropout**  
  Diagram lingkaran yang memperlihatkan rasio proporsi antara keduanya.

- **Dropout Berdasarkan Gender**  
  Bar chart yang menunjukkan jumlah mahasiswa yang dropout atau tidak berdasarkan jenis kelamin.

- **Dropout Berdasarkan Pendidikan Ibu**  
  Menunjukkan hubungan antara tingkat pendidikan ibu dan status dropout mahasiswa.

- **Dropout Berdasarkan Beasiswa**  
  Menggambarkan bagaimana distribusi dropout dipengaruhi oleh status penerimaan beasiswa.

### Akses Dashboard

Saat ini dashboard dapat diakses secara internal melalui Looker Studio. Jika tersedia akses publik, tautan akan disertakan di sini:

**[Link menuju dashboard](https://lookerstudio.google.com/reporting/d08e64c2-6959-44c2-8462-07ca13371c43)**

## Menjalankan Sistem Machine Learning

Untuk menguji sistem machine learning yang telah dikembangkan, tersedia sebuah **aplikasi web interaktif** berbasis Streamlit yang dapat langsung digunakan oleh pengguna.

### Akses Aplikasi

**[Klik di sini untuk membuka aplikasi prototype](https://proyek-akhir-menyelesaikan-permasalahan-perusahaan-edutech-azc.streamlit.app/)**

### Fitur Aplikasi

- Menampilkan data mahasiswa yang telah diproses.
- Memungkinkan pemilihan salah satu dari 6 model machine learning:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Support Vector Machine (SVM)
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
- Menampilkan hasil evaluasi model:
  - Akurasi
  - Confusion Matrix (visualisasi)
  - Classification Report

### Cara Menggunakan

1. Buka link aplikasi:  
    [https://proyek-akhir-menyelesaikan-permasalahan-perusahaan-edutech-azc.streamlit.app/](https://proyek-akhir-menyelesaikan-permasalahan-perusahaan-edutech-azc.streamlit.app/)

2. Di sidebar sebelah kiri:
   - Pilih salah satu model machine learning.
   - Klik tombol **" Jalankan Model"**.

3. Aplikasi akan menampilkan:
   - Data mahasiswa (beberapa baris awal).
   - Akurasi model terhadap data uji.
   - Confusion Matrix sebagai heatmap.
   - Classification Report dalam bentuk tabel.

### Tujuan

Aplikasi ini bertujuan untuk membantu pihak kampus dalam:

- Mendeteksi mahasiswa yang berisiko tinggi mengalami dropout.
- Menyediakan sistem yang **mudah digunakan**, bahkan untuk pengguna non-teknis.
- Memberikan insight cepat berbasis data akademik dan demografis.

## Kesimpulan

Hasil analisis eksploratif terhadap data mahasiswa Jaya Jaya Institut mengungkap sejumlah temuan penting terkait hubungan antara gender dan kemungkinan dropout.

1. **Tingkat Dropout Lebih Tinggi pada Mahasiswa Perempuan** 
 - Data menunjukkan bahwa mahasiswa perempuan memiliki tingkat dropout sebesar 45,05%, jauh lebih tinggi dibandingkan mahasiswa laki-laki yang berada di angka 25,10%. Perbedaan ini mengindikasikan bahwa gender merupakan faktor signifikan dalam risiko putus kuliah.

2. **Tingkat Pendidikan Ibu yang Lebih Rendah Berkorelasi dengan Dropout yang Lebih Tinggi** 

 - Mahasiswa yang berasal dari ibu dengan pendidikan dasar dan menengah menunjukkan proporsi dropout yang lebih tinggi dibandingkan dengan yang ibunya memiliki pendidikan tinggi. Ini mengindikasikan bahwa latar belakang pendidikan ibu berpengaruh terhadap keberlangsungan studi anaknya, dan dapat menjadi salah satu indikator risiko potensial untuk dropout.

3. **Beasiswa Berperan Signifikan dalam Menurunkan Risiko Dropout** 

 - Mahasiswa yang menerima beasiswa hanya memiliki tingkat dropout sebesar 12,19%, dibandingkan dengan 38,71% pada mahasiswa tanpa beasiswa. Ini menegaskan bahwa dukungan finansial memiliki dampak besar dalam meningkatkan ketahanan studi mahasiswa.

## Rekomendasi Action Items

Berdasarkan hasil temuan dari analisis data mahasiswa Jaya Jaya Institut, berikut beberapa rekomendasi strategis yang dapat diterapkan untuk menurunkan angka dropout dan meningkatkan retensi mahasiswa:

1. **Implementasi Program Pendampingan Khusus untuk Mahasiswa Perempuan**  
   - Bentuk tim konselor atau mentor akademik yang secara aktif memantau kemajuan studi mahasiswa perempuan.  
   - Adakan forum diskusi atau program pengembangan diri untuk membahas tantangan spesifik yang dihadapi mahasiswa perempuan.  
   - Lakukan survei berkala untuk mengidentifikasi penyebab utama dropout pada mahasiswa perempuan dan rumuskan intervensi preventif.

2. **Pemetaan Risiko Berdasarkan Latar Belakang Pendidikan Orang Tua**  
   - Tambahkan variabel pendidikan orang tua dalam sistem informasi akademik sebagai indikator risiko.  
   - Berikan pelatihan keterampilan belajar, manajemen waktu, dan sesi pendampingan khusus untuk mahasiswa dari orang tua berpendidikan rendah.  

3. **Perluasan dan Optimalisasi Program Beasiswa**  
   - Revisi kriteria penerima beasiswa agar mencakup mahasiswa dari keluarga berisiko tinggi mengalami dropout.  
   - Tambahkan skema *risk-based scholarship* yang didasarkan pada model prediktif data akademik dan sosial ekonomi.  
   - Tingkatkan promosi dan sosialisasi program beasiswa agar menjangkau lebih banyak mahasiswa yang membutuhkan.

4. **Pengembangan Sistem Peringatan Dini (*Early Warning System*)**  
   - Bangun dashboard berbasis machine learning untuk memantau potensi dropout mahasiswa secara real time.  
   - Terapkan sistem notifikasi otomatis bagi pihak akademik untuk melakukan intervensi lebih awal pada mahasiswa berisiko tinggi.

5. **Pelatihan Soft Skills dan Literasi Finansial bagi Mahasiswa Baru**  
   - Sediakan modul pelatihan tentang literasi keuangan, soft skills, dan kesiapan akademik saat masa orientasi mahasiswa baru.  
   - Prioritaskan pelatihan ini untuk mahasiswa dengan latar belakang keluarga yang kurang mendukung secara akademik maupun finansial.
