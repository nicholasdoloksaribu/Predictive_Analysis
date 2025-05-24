# Laporan Proyek Machine Learning - Nicholas Juniarto Doloksaribu

## Domain Proyek

Obesitas adalah masalah kesehatan yang makin mengkhawatirkan secara global dan mendesak untuk diatasi. Menurut WHO, pada tahun 2016 lebih dari 1,9 miliar orang dewasa mengalami kelebihan berat badan, dan lebih dari 650 juta di antaranya mengalami obesitas. Kondisi ini meningkatkan risiko penyakit kronis seperti diabetes, jantung, dan kanker. Karena itu, pendeteksian dini tingkat obesitas menjadi semakin penting untuk mencegah komplikasi serius di masa depan. 

Penerapan machine learning dalam klasifikasi obesitas berdasarkan atribut personal seperti usia, jenis kelamin, tinggi badan, berat badan, BMI, dan tingkat aktivitas fisik menawarkan solusi cepat dan akurat untuk mempercepat intervensi medis atau gaya hidup.

**Referensi:**
- [World Health Organization - Obesity and Overweight](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)

## Business Understanding

### Problem Statements
- Bagaimana mengklasifikasikan tingkat obesitas seseorang berdasarkan data personal dan aktivitas fisiknya?
- Model machine learning apa yang paling akurat dalam melakukan klasifikasi ini?

Permasalahan ini penting untuk segera diselesaikan, mengingat angka obesitas dunia yang terus meningkat dan dampaknya terhadap kualitas hidup serta biaya kesehatan jangka panjang. Solusi yang cepat dan tepat melalui model prediksi dapat berperan besar dalam strategi pencegahan.

### Goals
- Membuat model klasifikasi obesitas yang efektif dan akurat.
- Membandingkan performa berbagai model machine learning untuk memilih model terbaik.

### Solution Statements
- Menggunakan algoritma Logistic Regression, Random Forest, dan SVM untuk membangun model klasifikasi.
- Melakukan preprocessing untuk memaksimalkan performa model.
- Menggunakan classification report, confusion matrix, dan akurasi sebagai metrik evaluasi.

## Data Understanding

### Sumber Dataset
- [Kaggle - Obesity Level Prediction Dataset](https://www.kaggle.com/datasets/mrsimple07/obesity-prediction)

### Jumlah Data
- **Jumlah Baris**: 1000 baris
- **Jumlah Kolom**: 7 kolom

### Kondisi Data
- **Missing Value**: Tidak ditemukan missing value.
- **Duplikat**: Tidak ditemukan data duplikat.
- **Outlier**: Outlier ringan ditemukan terutama pada fitur berat badan (Weight), namun masih dalam batas wajar dan tidak dibuang karena tetap merepresentasikan populasi obesitas.

### Fitur-Fitur pada Dataset
| Nama Fitur | Tipe Data | Deskripsi |
|------------|-----------|-----------|
| Gender | Kategorikal (Male/Female) | Jenis kelamin |
| Age | Numerik | Usia dalam tahun |
| Height | Numerik | Tinggi badan (dalam meter) |
| Weight | Numerik | Berat badan (dalam kilogram) |
| BMI | Numerik | Indeks Massa Tubuh |
| PhysicalActivityLevel | Numerik | Skor tingkat aktivitas fisik |
| ObesityCategory | Kategorikal | Target variabel, kategori tingkat obesitas |

### Exploratory Data Analysis (EDA)

1. **Distribusi Kategori Obesitas**  
   ![Distribusi Kategori Obesitas](images/countplot.png)  
   **Penjelasan:** Plot batang di atas menunjukkan distribusi kategori obesitas dalam dataset. Terlihat bahwa kategori 'Normal weight' dan 'Overweight' memiliki jumlah data yang paling banyak, sementara kategori 'Underweight' memiliki jumlah data yang paling sedikit. Hal ini mengindikasikan bahwa dataset mungkin sedikit tidak seimbang.

2. **Korelasi Antar Fitur Numerik**  
   ![Heatmap Korelasi](images/heatmap.png)  
   **Penjelasan:** Heatmap di atas memvisualisasikan korelasi antar fitur numerik dalam dataset. Dari heatmap, dapat dilihat bahwa terdapat korelasi positif yang kuat antara 'Weight' dan 'BMI', yang sesuai dengan definisi BMI yang dihitung berdasarkan berat badan dan tinggi badan. Beberapa fitur lainnya juga menunjukkan korelasi, meskipun tidak sekuat antara 'Weight' dan 'BMI'.

3. **Hubungan Antar Fitur Numerik (Pairplot)**  
   ![Pairplot Fitur](images/pairplot.png)  
   **Penjelasan:** Pairplot di atas menggambarkan hubungan antara beberapa fitur numerik dan kategori obesitas. Terlihat bahwa individu dengan 'Weight' dan 'BMI' yang lebih tinggi cenderung berada pada kategori obesitas yang lebih tinggi. Selain itu, fitur-fitur seperti 'Age', 'Height', dan 'PhysicalActivityLevel' juga menunjukkan variasi yang berbeda di setiap kategori obesitas.

## Data Preparation

Langkah yang dilakukan:
- **Encoding**: Label Encoding pada kolom kategorikal seperti `Gender` dan `ObesityCategory`.
- **Scaling**: Standardisasi fitur numerik menggunakan StandardScaler.
- **Splitting**: Membagi data menjadi training dan testing set (80%:20%).

Tahapan ini penting untuk memastikan model dapat bekerja optimal, khususnya model SVM yang sensitif terhadap skala data.

## Modeling

### Algoritma yang Digunakan

1. **Logistic Regression**  
   - Cara kerja: Algoritma ini mengasumsikan hubungan linier antara input dan log-odds dari target. Logistic Regression mengoutput probabilitas kelas.
   - Parameter: Default parameter (tanpa modifikasi).

2. **Random Forest Classifier**  
   - Cara kerja: Ensemble learning yang membangun banyak decision tree dari subset data yang berbeda dan menggabungkannya untuk meningkatkan akurasi dan mengurangi overfitting.
   - Parameter: Default (`n_estimators=100`, `max_depth=None`, `random_state=42`).

3. **Support Vector Machine (SVM)**  
   - Cara kerja: SVM mencoba menemukan hyperplane terbaik yang memisahkan berbagai kelas dengan margin maksimal.
   - Parameter: Default (`C=1.0`, `kernel='rbf'`).

Dalam hal ini, **Random Forest** dipilih sebagai model terbaik karena memberikan akurasi tertinggi (99.5%). Meskipun Logistic Regression juga memberikan performa yang sangat baik dengan akurasi 97%, Random Forest lebih stabil dalam menangani kompleksitas data dan dapat mengurangi kemungkinan overfitting dengan banyaknya pohon yang digunakan.

## Evaluation

Metrik evaluasi yang digunakan:
- **Accuracy**: Untuk mengukur seberapa banyak prediksi yang benar dari keseluruhan data. Cocok digunakan karena fokus utama adalah klasifikasi multi-kelas dengan distribusi kategori yang relatif mirip setelah balancing.
- **Precision** dan **Recall**: Untuk memahami seberapa baik model mengidentifikasi masing-masing kategori obesitas, terutama penting agar tidak ada kategori yang diabaikan atau salah klasifikasi fatal.
- **F1-Score**: Karena data memiliki sedikit ketidakseimbangan antar kelas, F1-score menjadi metrik penting untuk menangkap keseimbangan antara precision dan recall.
- **Confusion Matrix**: Untuk visualisasi kesalahan dan benar pada masing-masing kategori obesitas.

### Formula Metrik:
- **Accuracy**: Persentase prediksi yang benar dibandingkan dengan total data.  
  Formula:  
  `Accuracy = (True Positives + True Negatives) / Total Samples`

- **Precision**: Mengukur akurasi dari prediksi yang positif.  
  Formula:  
  `Precision = True Positives / (True Positives + False Positives)`

- **Recall (Sensitivity)**: Mengukur seberapa baik model menangkap kelas positif yang sesungguhnya.  
  Formula:  
  `Recall = True Positives / (True Positives + False Negatives)`

- **F1-Score**: Rata-rata harmonis dari Precision dan Recall.  
  Formula:  
  `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

### Hasil Evaluasi Model

Tiga model dibandingkan, yaitu Logistic Regression, Random Forest, dan Support Vector Machine (SVM).

| Model               | Precision | Recall | F1-Score | Akurasi |
|---------------------|-----------|--------|----------|---------|
| Logistic Regression | 0.97      | 0.96   | 0.97     | 0.97    |
| Random Forest       | 1.00      | 0.99   | 0.99     | 0.995   |
| SVM                 | 0.94      | 0.92   | 0.93     | 0.93    |

**Confusion Matrix:**

- **Logistic Regression**
  ```
  [[74  0  0  0]
   [ 0 37  1  0]
   [2  0 57  0]
   [3  0  0 26]]
  ```
- **Random Forest**
  ```
  [[74  0  0  0]
   [0 37  1  0]
   [0  0 59  0]
   [0  0  0 29]]
  ```
- **SVM**
  ```
  [[69  0  4  1]
   [0 37  1  0]
   [2  1 56  0]
   [5  0  0 24]]
  ```

### Analisis Perbandingan Model

- **Random Forest** menunjukkan performa terbaik di semua metrik, dengan akurasi hampir sempurna (99.5%), precision dan recall tinggi, serta error yang sangat minim.
- **Logistic Regression** juga berkinerja baik, namun terdapat kesalahan klasifikasi pada kelas "Underweight" dan "Obesity".
- **SVM** menunjukkan performa yang cukup baik namun sedikit lebih rendah dibandingkan dua model lainnya, terutama pada prediksi kelas "Underweight".

### **Kesimpulan Evaluasi (Berbasis Metrik)**

- Model **Random Forest** terbukti paling efektif untuk klasifikasi tingkat obesitas berdasarkan metrik evaluasi, dengan **akurasi sangat tinggi (99.5%)**, menunjukkan bahwa hampir semua prediksi dilakukan dengan benar dan hanya terdapat kesalahan minimal.
- Model yang dibangun telah **menjawab problem statement** dengan baik berdasarkan nilai **precision (1.00)**, karena model mampu memprediksi kategori obesitas secara tepat tanpa menghasilkan banyak kesalahan prediksi positif.
- **Goal** dalam Business Understanding berhasil **dicapai** karena model menunjukkan **konsistensi antara precision dan recall yang tinggi**, dibuktikan dengan **F1-score sebesar 0.99**, yang menandakan keseimbangan sangat baik dalam mendeteksi semua kelas obesitas secara akurat.
- **Solution statement** terbukti **berdampak positif**, karena model memiliki **recall sebesar 0.99**, yang berarti mampu menangkap hampir seluruh data kategori obesitas yang sebenarnya, sehingga sangat mendukung sistem monitoring untuk pencegahan obesitas.
- Secara keseluruhan, model ini **selaras dengan kebutuhan bisnis**, karena berdasarkan **confusion matrix**, kesalahan klasifikasi sangat minim, menjadikannya **alat prediksi yang cepat, akurat, dan dapat diandalkan** untuk aplikasi praktis di dunia nyata, terutama dalam konteks kesehatan masyarakat.
