Laporan Proyek Machine Learning

Nama : Evriliya Syah Utami

Kelas : Pagi A

Domain Proyek

Estimasi harga mobil ini berfokus pada industri otomotif, dengan tujuan untuk mengembangkan model prediktif yang dapat memprediksi harga mobil berdasarkan berbagai faktor pada mobil.

Business Understanding

Agar kita tahu range harga tertentu sebuah komponen mobil, untuk mempengaruhi kualitas mobil dan membantu menetapkan harga yang tepat.

Problem Statements

Tantangan utama dalam proyek ini adalah mengembangkan model prediktif yang mampu memberikan estimasi harga mobil dengan akurasi tinggi berdasarkan data yang tersedia.

Goals

Mempermudah kita untuk mencari tahu harga sebuah mobil dari komponen mobil tersebut.

* Pengembangan Platfrom Estimasi Harga Mobil Berbasis Web, 
Solusi ini adalah mengembangkan platform estimasi harga mobil berbasis web yang didapatkan dari Kaggle.com untuk memberikan informasi tentang estimasi harga mobil dengan cepat. Platform ini akan menyediakan antarmuka pengguna yang ramah, memungkinkan pengguna mencari estimasi harga mobil berdasarkan kriteria tertentu seperti wheelbase, carheight, curbweight dan lainnya.
* Model yang dihasilkan dari datasets itu menggunakan metode Linear Regression.

Data Understanding

Dataset yang saya gunakan berasal dari Kaggle yang berisi harga mobil. Dataset terdapat lebih dari  26 columns setelah dilakukan data cleaning..

kaggle datasets download -d imgowthamg/car-price

Variabel-variabel pada Car Price adalah sebagai berikut:

* Price : Menunjukkan harga mobil. [Numbers, Min: 5151, Max: 17859167]
* car_ID : Menunjukkan ID Mobil. [Numbers, Min: 1, Max: 205]
* symboling : Menunjukkan nilai simbol. [Numbers, Min: 0, Max: 3]
* wheelbase : Menujukkan nilai jarak sumbu roda. [Numbers, Min: 88.4, Max: 115.6]
* carheight : Menujukkan nilai ketinggian mobil. [Numbers, Min: 48.8, Max: 59.8]
* curbweight : Menunjukkan nilai berat kosong. [Numbers, Min: 1488, Max: 3430]
* enginesize : Menunjukkan nilai ukuran mesin. [Numbers, Min: 61, Max: 326]
*horsepower : Menunjukkan nilai daya kuda. [Numbers, Min: 52, Max: 288]
* compressionratio : Menunjukkan nilai rasio kompresi. [Numbers, Min: 7, Max: 23]

Data Preparation

Data Collection

Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama dataset Car Price, jika anda tertarik dengan datasetnya, anda bisa click link diatas.

Data Discovery And Profiling 

Untuk bagian ini, kita akan menggunakan teknik EDA.
Pertama kita mengimport semua library yang dibutuhkan.

import pandas as pd
import numpy as np
import matplotlib.pypot as plt
import seaborn as sns

Karena kita menggunakan google colab untuk mengerjakannya maka kita akan import files juga.

from google.colab import files

Lalu mengupload token kaggle agar nanti bisa mendownload sebuah dataset dari kaggle melalui google colab

files.upload()

Setelah mengupload filenya, maka kita akan lanjut ke tahap membuat sebuah folder untuk menyimpan file kaggle.json yang sudah diupload tadi

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle

Selesai, lalu mari kita download datasetsnya

!kaggle datasets download -d imgowthamg/car-price

Selanjutnya, kita harus extract file yang tadi telah didownload

!mkdir car-price
!unzip car-price.zip -d car-price
!ls car-price

 Selanjut dengan memasukkan file csv yang telah diextract pada sebuah variable, dan melihat 5 data paling atas dari datasetsnya

 df = pd.read_csv('car-price/CarPrice_Assignment.csv')
 df.head()

 Karena di dalamnya terdapat satu kolom yang tidak kita inginkan, maka kita akan drop satu kolom itu.

 column_to_drop = 'Unnamed: 0'
df = df.drop(columns=[column_to_drop])

Selanjutnya kita akan memeriksa apakah datasetsnya terdapat baris yang kosong atau null dengan menggunakan seaborn.

sns.heatmap(df.isnull())

Untuk berjaga-jaga mari kita check apakah terdapat duplicate data di datasets kita.

df[df.duplicated()]

Kita lanjut dengan data exploration kita.

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)

drop semua data duplicate.

df.drop_duplicates(inplace=True)

Oke sudah aman, mari lanjut. Di lokasi terdapat huruf ".m", ".q", ".r", mari kita pisahkan itu dan masukkan kedalam sebuah kolom baru.

ddf['engine_m_q_r'] = df.engine.str.split().str[-1].str.replace('.', '')
df['engine'] = df['engine'].str.replace('m.', '').str.replace('q.', '').str.replace('r.', '').str.strip()

Kita bisa melihat hasilnya dengan melakukan command ini.

print(df.engine_m_q_r)
print(df.engine)

Selanjutnya kita akan memisahkan apartment yang terletak pada lantai pertama dan lantai paling atas

df['engine_location'] = (df.engine.str.split('/').str[0] == '1').astype(int)
print(df.engine_location.value_counts())
df['engine_type'] = (df.engine.str.split('/').str[0] == df.engine.str.split('/').str[-1]).astype(int)
print(df['engine_size'].value_counts())
df = df.drop(['engine'], axis=1)

Tidak terasa proses data exploration dan cleansing sudah dilaksanakan, mari lanjut dengan modeling.

Modeling

sebelumnya mari kita import library yang nanti akan digunakan.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

Langkah pertama adalah memasukkan kolom-kolom fitur yang ada di datasets dan juga kolom targetnya.

features = ['car_ID', 'symbol', 'wheelbase', 'carheight', 'curbweight', 'horsepower', 'copressionratio', 'engine_m_q_r']
X = df[features]
y = df.price

Selanjutnya kita akan menentukan berapa persen dari datasets yang akan digunakan untuk test dan untuk train, disini kita gunakan 20% untuk test dan sisanya untuk training alias 80%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Mari kita lanjut dengan membuat model Linear Regressionnya

model = LinearRegression()

 Sebelum kita memasukkan X_train dan y_train pada model, kita harus konvert q, r, dan m menjadi integer terlebih dahulu agar nanti bisa dijadikan angka untuk diproses oleh modelnya.

 df['engine_m_q_r'] = df['engine_m_q_r'].map({'q': 0, 'r': 1, 'm': 2}).astype(int)

 Oke mari lanjut, memasukkan X_train dan y_train pada model dan memasukkan value predict pada y_pred.

 model.fit(X_train, y_train)
y_pred = model.predict(X_test)

selesai, sekarang kita bisa melihat score dari model kita.

score = model.score(X_test, y_test)
print(f"this has {score} of score")

Mari kita test menggunakan sebuah array value.

input_data = np.array([[245, 3, 1, 1, 1, 1, 1, 0, 2]])
prediction = model.predict(input_data)
print('Estimasi harga mobil dalam Manat : ', prediction)

sekarang modelnya sudah selesai, mari kita export sebagai sav agar nanti bisa kita gunakan pada project web streamlit kita.

import pickle

filename = "estimasi_mobil.sav"
pickle.dump(model,open(filename,'wb'))

Evaluation

Disini saya menggunakan F1 score sebagai metrik evaluasi.

* F1 Score: F1 score adalah rata-rata harmonis antara presisi dan recall. F1 score memberikan keseimbangan antara presisi dan recall. F1 score dihitung dengan menggunakan rumus:

* Setelah itu saya menerapkannya dalam kode menggunakan fungsi f1_score, seperti berikut :

from sklearn.metrics import precision_recall_curve, f1_score

threshold = 200000

y_pred_binary = (y_pred > threshold).astype(int)
y_test_binary = (y_test > threshold).astype(int)

f1 = f1_score(y_test_binary, y_pred_binary)

print('F1 Score:', f1)

dan hasil yang saya dapatkan adalah 0.576169839351atau 57.6%, itu berarti model ini memiliki keseimbangan yang baik antara presisi dan recall. Karena kita mencari patokan harga untuk membeli Apartment maka model yang presisi sangat dibutuhkan agar kemungkinan terjadinya kesalahan semakin sedikit.

Deployment

[My App Streamlit](https://app-estimasi-6kqfgcjka7nmwmj4dctmwi.streamlit.app/#estimasi-harga-mobil)
![image](https://github.com/evriliyasy/streamlit-estimasi/assets/148839476/45dc1298-a261-4d63-bed3-cb523d0100ec)

