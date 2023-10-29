import pickle
import numpy as np
import streamlit as st

model = pickle.load(open('D:/projects/load_prediction/load_prediction/regmodel.pkl','rb'))
#create a function for prediction

def load_prediction(prediction):
    input_data_as_numpy_array=np.asarray(prediction,dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]
def main():
    st.title('Load Prediction Web App')
    Month = st.text_input('Month')
    average_temperature = st.text_input('Temperature')
    average_cloud_cover = st.text_input('Cloud Cover')
    average_dew_point=st.text_input('Dew Point')
    average_wind_speed = st.text_input('Wind Speed')
    average_humidity = st.text_input('Humidity')
    apparent_temperature = st.text_input('Apparent Temperature')
    load = ''
    if st.button('Predict The Load'):
        load = load_prediction([Month, average_temperature, average_cloud_cover, average_dew_point, average_wind_speed, average_humidity, apparent_temperature])
        st.success(load)

if __name__ == '__main__':
    main()