import pickle
import streamlit as st

# membaca model
estimasi_mobil = pickle.load(open('estimasi_mobil.sav','rb'))

#judul web
st.title('Estimasi Harga Mobil')

car_ID = st.number_input('input nilai car_ID')


symboling = st.number_input('input nilai symboling')


wheelbase = st.number_input('input nilai wheelbase')


carheight = st.number_input('input nilai carheight')


curbweight = st.number_input('input nilai curbweight')


enginesize = st.number_input('input nilai enginesize')


horsepower = st.number_input('input nilai horsepower')


compressionratio = st.number_input('input nilai compressionratio')


# code untuk prediksi
predict = ''

# membuat tombol untuk prediksi
if st.button('Estimasi Harga') :
    predict = estimasi_mobil.predict(
           [[car_ID, symboling, wheelbase, carheight, curbweight, enginesize, horsepower, compressionratio]]
        )
st.write ('Estimasi harga mobil dalam Ponds : ' , predict)
st.write ('Estimasi harga mobil dalam IDR (Juta) :', predict*19000)