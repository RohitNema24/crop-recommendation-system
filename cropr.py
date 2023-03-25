import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings

st.set_page_config(page_title="Crop Recommendation System")
def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model
def main():
    st.title('Crop Recommendation System By Team AgroTech 🌱')
    col1, col2 = st.columns([2, 2])

    with col1:
        with st.expander(" ℹ️ Information", expanded=True):
            st.write("""
            Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. 
            However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.
            """)
        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''

    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen",1,100)
        P = st.number_input("Phosporus", 1,100)
        K = st.number_input("Potassium", 1, 100)
        temp = st.number_input("Temperature", 0.0, 100000.0,format="%0.6f")
        humidity = st.number_input("Humidity in %", 0.0, 100000.0)
        ph = st.number_input("Ph", 0.0, 100000.0)
        rainfall = st.number_input("Rainfall in mm", 0.0, 100000.0)
        d = {
            "Kidneybeans": "राजमा",'rice':'चावल','Jute': 'जूट', 'Coffee':'कॉफ़ी'
        }
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        if st.button('Predict In English'):
            loaded_model = load_model('model_rf.pkl')
            prediction = loaded_model.predict(single_pred)
            ans=prediction.item().title()
            col1.write('''
		    # Results 🔍 
		    ''')
            col1.success(f"{prediction.item().title()} are recommended by the A.I for your farm.")
        if st.button('हिंदी में परिणाम जानिए'):
            loaded_model = load_model('model_rf.pkl')
            prediction = loaded_model.predict(single_pred)
            ans = prediction.item().title()
            col1.write('''
            		    # परिणाम 🔍 
            		    ''')
            col1.success(f"एआई द्वारा आपके खेत के लिए :--->> {d[ans]} <---: की खेती करने का सुझाव दिया गया है।")
    st.warning(
        "Note: This A.I application is for farmers to get recommendation of crop that would be suitable according to their conditions")


if __name__ == '__main__':
    main()