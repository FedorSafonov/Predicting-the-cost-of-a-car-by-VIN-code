import streamlit as st
from car_price_predictor import CarPricePredictor

def main():

    predictor = CarPricePredictor()

    lang_text = "Язык (Language):"
    lang = st.radio(lang_text, ["Русский", "English"])

    if lang == "Русский":
        lang = "Ru"
        button_text = "Предсказать"
        title_text = "Предсказание стоимости автомобиля по VIN-коду"
        vin_code_text = """Введите VIN-код:

( 12-17 знаков с большой буквы, пример: 1FTGE2EW0EDA31393 )"""
        decode_vin_text = "Расшифровывать VIN"

    else:
        lang = "En"
        button_text = "Predict"
        title_text = "Predicting the cost of a car by VIN code"
        vin_code_text = """Enter the VIN code:

( 12-17 characters with a capital letter, example: 1FTGE2EW0EDA31393 )"""
        decode_vin_text = "Decrypt the VIN"

    st.title(title_text)

    vin_code = st.text_input(vin_code_text, '')

    decode_vin = st.checkbox(decode_vin_text)

    if st.button(button_text):
        results = predictor.predict(vin_code, price_only=not decode_vin, lang=lang)
        st.write(results.T)

if __name__ == "__main__":
    main()