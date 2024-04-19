class CarPricePredictor:
    """Класс для предсказания цены автомобиля по VIN-коду."""

    def __init__(self, model_path='final_model'):
        """Инициализирует экземпляр класса.

        Args:
            model_path (str, optional): Путь к файлу с обученной моделью. 
                                         По умолчанию 'final_model'.
        """
        self.model = load_model(model_path, verbose=False)

    def predict(self, vin_code, price_only=True, lang='Ru'):
        """Предсказывает цену автомобиля по VIN-коду (или списку кодов).

        Args:
            vin_code (str или list): VIN-код автомобиля или список VIN-кодов.
            price_only (bool, optional): Если True, возвращает только предсказанную цену (DataFrame). 
                                         Иначе, возвращает DataFrame с информацией о VIN и ценой.
                                         По умолчанию True.
            lang (str, optional): Язык вывода информации. 'Ru' для русского, 'En' для английского.
                                  По умолчанию 'Ru'.
                                  
        Returns:
            pd.DataFrame:  DataFrame с информацией о VIN и ценой.

        Exemple:
            predictor = CarPricePredictor()

            vin_codes = ['1FTNE2EW0EDA31393', '1FANE2EW0EDA31393', '3FANE2EW0EDA31393']
            results = predictor.predict(vin_codes, price_only=False, lang='Ru')
        """
        results = []
        if isinstance(vin_code, str):
            vin_code = [vin_code]

        for vin in vin_code:
            vin_info = extract_vin_info(vin)
            manufacturer = vin_info['manufacturer']
            security_codes = vin_info['security_codes']
            model_car = vin_info['model']
            engine = vin_info['engine']
            check_digit = vin_info['check_digit']
            model_year = vin_info['model_year']
            plant_code = vin_info['plant_code']
            serial_number = vin_info['serial_number']
            
            data = pd.DataFrame({'manufacturer': [manufacturer], 
                                'security_codes': [security_codes],
                                'model': [model_car],
                                'engine': [engine],
                                'check_digit' : [check_digit],
                                'model_year': [model_year],
                                'plant_code': [plant_code],
                                'serial_number' : [serial_number]})
            
            prediction = predict_model(self.model, data=data.drop(['check_digit','serial_number'], axis=1))
            price = data.copy()
            price['predicted_price'] = prediction['prediction_label'].round(2)
            price['VIN'] = vin
            price = price.set_index('VIN')

            if lang == 'Ru':
                price.columns = ['Производитель', 'Код безопасности', 'Модель', 
                                'Двигатель','Знак качества', 'Год','Завод', 'Серийный номер',
                                'Предсказанная цена']
            results.append(price)

        if len(results) == 1:
            final = results[0]
        else:
            final = pd.concat(results)

        if price_only:
            return pd.DataFrame(final.iloc[:, -1])
        else:
            return final

def predict_car_price(vin_code, price_only = True, lang='Ru'):
    """Предсказывает цену автомобиля по VIN-коду (или списку кодов).

    Args:
        vin_code (str или list): VIN-код автомобиля или список VIN-кодов.

        price_only (bool, optional): Если True, возвращает только предсказанную цену (DataFrame). 
                                     Иначе, возвращает DataFrame с информацией о VIN и ценой.
                                     По умолчанию True.
                                     
        lang (str, optional): Язык вывода информации. 'Ru' для русского, 'En' для английского.
                              По умолчанию 'Ru'.

    Returns:
        pd.DataFrame:  DataFrame с информацией о VIN и ценой.
    
    Exemple:
    
        vin_codes = ['1FTNE2EW0EDA31393','1FANE2EW0EDA31393','3FANE2EW0EDA31393']
        result = predict_car_price(vin_codes,price_only=False)
    """

    
    model = load_model('final_model',verbose=False)

    results = []
    if isinstance(vin_code, str):
        vin_code = [vin_code]
    
    for vin in vin_code:
        vin_info = extract_vin_info(vin)
        manufacturer = vin_info['manufacturer']
        security_codes = vin_info['security_codes']
        model_car = vin_info['model']
        engine = vin_info['engine']
        check_digit = vin_info['check_digit']
        model_year = vin_info['model_year']
        plant_code = vin_info['plant_code']
        serial_number = vin_info['serial_number']
        
        data = pd.DataFrame({'manufacturer': [manufacturer], 
                            'security_codes': [security_codes],
                            'model': [model_car],
                            'engine': [engine],
                            'check_digit' : [check_digit],
                            'model_year': [model_year],
                            'plant_code': [plant_code],
                            'serial_number' : [serial_number]})
        
        prediction = predict_model(model, data=data.drop(['check_digit','serial_number'], axis=1))
        price = data.copy()
        price['predicted_price'] = prediction['prediction_label'].round(2)
        price['VIN'] = vin
        price = price.set_index('VIN')

        if lang == 'Ru':
            price.columns = ['Производитель', 'Код безопасности', 'Модель', 
                            'Двигатель','Знак качества', 'Год','Завод', 'Серийный номер',
                            'Предсказанная цена']
        
        results.append(price)
    
    if len(results) == 1:
        final = results[0]

    else:  
        final =  pd.concat(results)
    
    if price_only:
        return pd.DataFrame(final.iloc[:,-1])

    else:
        return final