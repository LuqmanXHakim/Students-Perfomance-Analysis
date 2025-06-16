# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Di tengah revolusi digital global, sektor pendidikan mengalami transformasi besar melalui integrasi teknologi, yang melahirkan industri education technology (edutech). Edutech hadir sebagai solusi strategis untuk menjawab tantangan klasik dalam dunia pendidikan, seperti keterbatasan akses, perbedaan kualitas pengajaran, serta kurangnya personalisasi pembelajaran. Dengan memanfaatkan teknologi seperti internet, kecerdasan buatan, dan perangkat mobile, edutech memungkinkan proses pembelajaran yang lebih fleksibel, inklusif, dan berbasis data.

Salah satu kekuatan utama edutech adalah kemampuannya menjangkau pelajar dari berbagai wilayah, termasuk daerah terpencil, tanpa harus mengandalkan infrastruktur fisik. Selain itu, pendekatan interaktif dan multimedia menjadikan pembelajaran lebih menarik dan meningkatkan motivasi belajar siswa. Melalui sistem adaptif, edutech mampu menyesuaikan materi dan metode pengajaran dengan kebutuhan dan perkembangan tiap individu, menciptakan pengalaman belajar yang unik bagi setiap pengguna.

Tidak hanya bermanfaat dari sisi pedagogi, edutech juga membuka peluang bisnis yang luas. Inovasi dalam pengembangan platform, konten edukatif digital, sistem manajemen pendidikan, serta pelatihan dan sertifikasi guru menjadi sektor-sektor yang memiliki potensi pertumbuhan tinggi. Kolaborasi antara penyedia edutech dan institusi pendidikan formal pun menjadi kunci untuk mempercepat transformasi digital pendidikan yang merata dan berkelanjutan.

### Permasalahan Bisnis

Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Cakupan Proyek

Untuk mengantisipasi dan mengidentifikasi potensi siswa dropout di Jaya Jaya Institut, proyek ini akan melibatkan lima tahapan utama sebagai berikut:

- Mengkaji data siswa yang tersedia, termasuk profil, riwayat akademik, dan aspek sosial ekonomi, guna memahami konteks dan struktur data.

- Menangani data yang hilang, duplikat, serta melakukan encoding data kategorik dan normalisasi untuk memastikan kualitas input bagi model prediktif.

- Mengidentifikasi fitur-fitur paling relevan yang mempengaruhi kemungkinan siswa mengalami dropout, serta menganalisis hubungan antarvariabel.

- Membangun dan melatih beberapa algoritma klasifikasi dengan membagi data menjadi data latih dan data uji untuk mengembangkan model prediktif.

- Menguji model dengan metrik evaluasi seperti akurasi dan F1-score, kemudian mengimplementasikan model terpilih ke dalam aplikasi yang dapat digunakan oleh pengguna akhir.

### Persiapan

Sumber data: Predict students' dropout and academic success. UCI Machine Learning Repository.
```
https://doi.org/10.24432/C5MC89
```

Setup environment:
```
Tools yang digunakan:
- Python (pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit)
- Looker Studio (untuk visualisasi dashboard)
- Jupyter Notebook (untuk eksplorasi dan modelling)
```

Setup Google Colab
```
!pip install -r requirements.txt
```

Setup Environment - Anaconda
```
conda create --name main-ds python=3.11
conda activate main-ds
pip install -r requirements.txt
```

Setup Environment - Terminal
```
mkdir student-perfomance
cd student-perfomance
pipenv install
pipenv shell
pip install -r requirements.txt
```

Run streamlit app
```
streamlit run app.py
```

## Business Dashboard
Berikut kalimat alternatif yang berbeda namun memiliki makna yang sama:
Dashboard ini dikembangkan menggunakan Google Looker Studio guna memvisualisasikan distribusi data serta menganalisis dampak masing-masing variabel terhadap performa mahasiswa. Anda dapat melihat dashboard tersebut melalui tautan berikut:

## Menjalankan Sistem Machine Learning
Sistem machine learning yang dikembangkan dapat memprediksi kemungkinan seorang mahasiswa akan dropout atau lulus berdasarkan fitur-fitur akademik dan demografis.

```
https://students-predict-analysis.streamlit.app/
```

![image](https://github.com/user-attachments/assets/eb91b662-d8c6-4466-bef3-97da8175bd75)

```
Mulai aplikasi Streamlit dengan mengetik perintah berikut di terminal:
1. $ streamlit run app.py
2. Isi form input yang tersedia dengan data mahasiswa.
3. Sistem akan memproses data dan menampilkan prediksi status mahasiswa.
4. Model yang digunakan diambil dari file pickle/joblib hasil pelatihan sebelumnya.
```

## Conclusion


### Rekomendasi Action Items
Beberapa langkah strategis yang dapat diterapkan oleh institusi untuk menangani masalah dropout dan mendukung pencapaian tujuan pendidikan:

- Menyediakan pendampingan akademik dan emosional tambahan bagi mahasiswa penerima beasiswa guna membantu mereka menghadapi hambatan dalam studi maupun kehidupan pribadi.
- Melakukan evaluasi terhadap kurikulum saat ini guna memastikan kesesuaiannya dengan perkembangan zaman dan kebutuhan peserta didik.
- Menyediakan layanan pendukung seperti tutor, kelas tambahan, konseling akademik, serta layanan kesehatan mental bagi mahasiswa yang memerlukan.
- Merancang program pendidikan fleksibel seperti kelas daring atau blended lear
