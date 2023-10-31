import pickle
import streamlit as st

# membaca model
estimasi_pangan = pickle.load(open('estimasi_pangan.sav', 'rb'))

#judul web
st.title('Estimasi Harga Pangan')

Beras_Termurah = st.number_input('input nilai Beras_Termurah')


Bawang_Merah = st.number_input('input nilai Bawang_Merah')


Telur_Ayam = st.number_input('input nilai Telur_Ayam')


Cabe_Merah_Keriting = st.number_input('input nilai Cabe_Merah_Keriting')


Daging_Ayam = st.number_input('input nilai Daging_Ayam')


Daging_Sapi = st.number_input('input nilai Daging_Sapi')


Gula_Pasir = st.number_input('input nilai Gula_Pasir')


Cabe_Merah_Besar = st.number_input('input nilai Cabe_Merah_Besar')


Jagung = st.number_input('input nilai Jagung')


Kedelai = st.number_input('input nilai Kedelai')


Minyak_Goreng = st.number_input('input nilai Minyak_Goreng')

# code untuk prediksi
predict = ''

# membuat tombol untuk prediksi
if st.button('Estimasi Harga') :
    predict = estimasi_pangan.predict(
           [[Beras_Termurah, Bawang_Merah, Telur_Ayam, Cabe_Merah_Keriting, Daging_Ayam,Daging_Sapi, Gula_Pasir, Cabe_Merah_Besar, Jagung, Kedelai, Minyak_Goreng]]
        )
st.write ('Estimasi harga pangan dalam : USD' , predict)
st.write ('Estimasi harga pangan dalam IDR (Juta) :', predict*19000)
