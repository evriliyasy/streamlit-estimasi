# Laporan Proyek Machine Learning

Nama : Evriliya Syah Utami

Kelas : Pagi A

# Domain Proyek

Proyek ini dapat digunakan untuk memperkirakan informasi harga pangan. Pangan merupakan bagian terpenting dari kehidupan sehari-hari dan memiliki dampak yang besar pada kesejahteraan dan ekonomi masyarakat. Estimasi harga pangan ini membantu dalam perencanaan ekonomi, kebijakan pangan, dan memberikan wawasan penting kepada produsen dan konsumen. 

 # Business Understanding

Untuk mengecek perkiraan harga jual pangan yang ada  di Daerah Istimewa Yogyakarta.

# Problem Statements

* Tantangan saya dalam proyek ini adalah, memperkirakan harga pangan berdasarkan faktor-faktor seperti kondisi cuaca, produksi pertanian, permintaan konsumen, dll.

# Goals

* Mengembangkan model prediktif yang dapat         memperkirakan harga pangan dengan akurasi tinggi.
* Memberikan wawasan tentang faktor-faktor yang mempengaruhi fluktuasi harga pangan.

# Solution Statements

Solusi untuk proyek ini adalah mengumpulkan dan menganalisis data terkait dengan kondisi cuaca, produksi pertanian, permintaan konsumen, dan faktor lain yang dapat mempengaruhi harga pangan. Data ini akan digunakan untuk melatih model machine learning yang dapat memprediksi harga pangan di masa depan.

# Data Understanding

Memperkirakan harga pangan di Daerah Istimewa Yogyakarta untuk menjadi estimasi masyarakat di sana.
Proyek ini berguna untuk para ibu-ibu yang membutuhkan jika mereka membeli bahan makanan.
inilah datasets yang saya ambil (https://www.kaggle.com/datasets/nurcholisart/daftar-harga-pangan-daerah-istimewa-yogyakarta/data).

# Variabel-variabel Daftar Harga Pangan Daerah Istimewa Yogyakarta sebagai berikut:

* Beras Termurah = Beras dengan harga terjangkau dengan type (int64)
* Bawang Merah = dengan type (int64)
* Telur Ayam = dengan type (int64)
* Cabe Merah Keriting = dengan type (int64)
* Daging Ayam = dengan type (int64)
* Daging Sapi = dengan type (int64)
* Gula Pasir = dengan type (int64)
* Cabe Merah Besar = dengan type (int64)
* Jagung = dengan type (int64)
* Kedelai = dengan type (int64)
* Minyak Goreng = dengan type (int64)

# Data Preparation

Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama Daftar Harga Pangan Daerah Istimewa Yogyakarta, jika anda tertarik dengan datasetnya, anda bisa click link diatas.

# Data Discovery And Profiling

Untuk bagian ini, kita akan menggunakan teknik EDA.
Pertama kita mengimport semua library yang dibutuhkan,

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Karena kita menggunakan google colab untuk mengerjakannya maka kita akan import files juga,

```python
from google.colab import files
```

Lalu mengupload token kaggle agar nanti bisa mendownload sebuah dataset dari kaggle melalui google colab

```python
files.upload()
```

Setelah mengupload filenya, maka kita akan lanjut dengan membuat sebuah folder untuk menyimpan file kaggle.json yang sudah diupload tadi

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Done, lalu mari kita download datasetsnya

```python
!kaggle datasets download -d nurcholisart/daftar-harga-pangan-daerah-istimewa-yogyakarta
```

Selanjutnya kita harus extract file yang tadi telah didownload 

```python
!mkdir daftar-harga-pangan-daerah-istimewa-yogyakarta
!unzip daftar-harga-pangan-daerah-istimewa-yogyakarta.zip -d daftar-harga-pangan-daerah-istimewa-yogyakarta
!ls daftar-harga-pangan-daerah-istimewa-yogyakarta
```

Mari lanjut dengan memasukkan file csv yang telah diextract pada sebuah variable, dan melihat 5 data paling atas dari datasetsnya

```python
df = pd.read_csv('daftar-harga-pangan-daerah-istimewa-yogyakarta/daftar-harga-pangan-diy.csv')
```

Untuk melihat beberapa baris pertama dari sebuah DataFrame.

```python
df.head()
```
Untuk melihat mengenai type data dari masing masing kolom kita bisa menggunakan property info,

```python
df.info()
```
Untuk melihat beberapa baris terakhir dari sebuah DataFrame.

```python
sns.heatmap(df.isnull())
```

![Screenshot (118)](https://github.com/evriliyasy/streamlit-estimasi/assets/148839476/944d9e5c-6916-42cb-adbb-3d7a4c1d8b0d)

menghasilkan statistik deskriptif tentang DataFrame, seperti rata-rata, median, kuartil, dan lainnya, untuk setiap kolom numerik dalam DataFrame

```python
df.describe()
```
Mari kita lanjut dengan visualisai data kita, dan akan munsul atribut yang numerik atau integer

```python
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
```

![Screenshot (119)](https://github.com/evriliyasy/streamlit-estimasi/assets/148839476/23bedc9b-5e49-4063-8e31-313e97676b5e)

Membuat beberapa plot atau garfik
Pertama membuat grup cabe merah keriting dan bawah merah

```python
cabe_merah_keriting = df.groupby('Cabe Merah Keriting').count()[['Bawang Merah']].sort_values(by='Bawang Merah', ascending=True).reset_index()
cabe_merah_keriting = cabe_merah_keriting.rename(columns={'Bawang Merah': 'numberOfCars'})
```
Lalu membuat tipikal grafik dalam bentuk barplot

```python
fig = plt.figure(figsize=(15,5))
sns.barplot(x=cabe_merah_keriting['Cabe Merah Keriting'], y=cabe_merah_keriting['numberOfCars'], color='royalblue')
plt.xticks(rotation=60)
```
![Screenshot (120)](https://github.com/evriliyasy/streamlit-estimasi/assets/148839476/7f60af1a-3bed-4caa-a697-6d6a82a86b1e)

yang Kedua Membuat grup beras termurah dan cabe merah besar

```python
beras_termurah = df.groupby('Beras Termurah').count()[['Cabe Merah Besar']].sort_values(by='Cabe Merah Besar').reset_index()
beras_termurah = beras_termurah.rename(columns={'Cabe Merah Besar':'count'})
```
Lalu membuat tipikal grafik dalam bentuk barplot

```python
plt.figure(figsize=(15,5))
sns.barplot(x=beras_termurah['Beras Termurah'], y=beras_termurah['count'], color='royalblue')
```
![Screenshot (123)](https://github.com/evriliyasy/streamlit-estimasi/assets/148839476/a9e45c3e-f14c-4656-b7ae-4fef2a4f5d2b)


ketiga membuat plot menggunakan distribusi atau displot

```python
plt.figure(figsize=(10, 5))
sns.displot(df='Cabe Merah Keriting')
```

hasilnya kita mendapatkan nilai distribusi dari cabe merah keriting
![Screenshot (121)](https://github.com/evriliyasy/streamlit-estimasi/assets/148839476/8b0d74c5-1c17-4f08-8343-341f01419194)


Distribusi beras medium

```python
plt.figure(figsize=(10, 5))
sns.displot(df='Beras Medium')
```
hasilnya kita mendapatkan nilai distribusi dari beras medium
![Screenshot (122)](https://github.com/evriliyasy/streamlit-estimasi/assets/148839476/dd238ec6-cec7-4b35-ad36-3f2bc047490f)


kita berhasil melakukan plotdataset, mendeskripsikan dataset dan memberikan informasi dari grafik.

Mari kita lanjut ke modeling

# Modeling

Langkah pertama kita melakukan seleksi fitur karena tidak semua antribut yang ada didataset kita pakai

Memilih fitur yang ada di dataset dan penamaan atau huruf harus sama seperti di dataset supaya terpanggil serta menentukan featurs dan labels

```python
features = ['Beras Termurah','Bawang Merah','Telur Ayam', 'Cabe Merah Keriting', 'Daging Ayam', 'Daging Sapi', 'Gula Pasir', 'Cabe Merah Besar', 'Jagung', 'Kedelai', 'Minyak Goreng']
x = df[features]
y = df['Beras Medium']
x.shape, y.shape
```
Sebelumnya mari kita import library yang nanti akan digunakan

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```
Membuat model regresi linier dan memasukkan modul dari sklearn(memasukkan library)

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
```
nah, selesai, sekarang kita bisa melihat score dari model kita

```python
score = lr.score(x_test, y_test)
print('akurasi model regresi linier =', score)
```
akurasi model regresi linier = 0.7884450185324221 atau 78,84%, alright mari kita test menggunakan sebuah array value

```python
input_data = np.array([[6800,8000,15000,12000,24000,8000,12000,14000,4156,9100,8000]])
prediction = lr.predict(input_data)

print('Estimasi harga pangan dalam persen :', prediction)
```
wow, berhasil!!, sekarang modelnya sudah selesai, mari kita export sebagai sav agar nanti bisa kita gunakan pada project web streamlit kita.

```python
import pickle

filename = 'estimasi_pangan.sav'
pickle.dump(lr,open(filename,'wb'))
```

# Evaluation

# Impor pustaka yang diperlukan

from sklearn.model_selection import train_test_split from sklearn.linear_model import LinearRegression

# Muat Dataset

data = Daftar Harga Pangan Daerah Istimewa Yogyakarta

x = data.data

y = data.target

# Bagi data menjadi data traning dan data testing

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=70)

y_test.shape

# Inisialisasi model

lr = LinearRegression()

lr.fit(x_train,y_train)

pred = lr.predict(x_test)

# Menghitung score atau akurasi

score = lr.score(x_test, y_test)

print('akurasi model regresi linier =', score)

# Deployment

![Screenshot 2023-10-31 074621](https://github.com/evriliyasy/streamlit-estimasi/assets/148839476/e0a51952-65e1-4a99-bcc4-f80094a443ce)

