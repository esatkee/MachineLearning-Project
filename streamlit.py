import streamlit as st
import joblib
import numpy as np


def load_model():
    try:
        model = joblib.load('results/models/saved_models/best_model.pkl')
        encoder = joblib.load('results/models/saved_models/label_encoder.pkl')
        return model, encoder
    except Exception as e:
        st.error(f"Model yüklenemedi {str(e)}")
        return None, None


def main():
    st.title("🌪️ Hava Durumu Tahmini")

    model, encoder = load_model()
    if not model:
        return

    st.subheader("Aşağıdaki 5 özelliği doldurun:")

    col1, col2 = st.columns(2)
    with col1:
        temp_f = st.number_input("Sıcaklık (°F)", value=0)
        dew_f = st.number_input("Çiy Noktası (°F)", value=0)
        humidity = st.number_input("Nem (%)", value=0)
    with col2:
        wind_mph = st.number_input("Rüzgar Hızı (mph)", value=0)
        pressure_inhg = st.number_input("Basınç (inHg)", value=0)

    if st.button("Tahmin"):
        try:
            input_data = np.array([[temp_f, dew_f, humidity, wind_mph, pressure_inhg]])

            if model.n_features_in_ > 5:
                full_input = np.zeros((1, model.n_features_in_))
                full_input[0, :5] = input_data
                input_data = full_input

            prediction = model.predict(input_data)
            result = encoder.inverse_transform(prediction)[0]

            st.markdown(f"""
            <div style='background-color:#f0f2f6; padding:20px; border-radius:10px;'>
                <h2 style='color:#0068c9;'>Tahmin Sonucu: {result}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.table({
                "Parametre": ["Sıcaklık", "Çiy Noktası", "Nem", "Rüzgar Hızı", "Basınç"],
                "Değer": [f"{temp_f} °F", f"{dew_f} °F", f"{humidity}%", f"{wind_mph} mph", f"{pressure_inhg} inHg"]
            })

        except Exception as e:
            st.error(f"Hata: {str(e)}")
            st.error("Model beklediği özellik sayısı: " + str(getattr(model, 'n_features_in_', 'Bilinmiyor')))


if __name__ == "__main__":
    main()