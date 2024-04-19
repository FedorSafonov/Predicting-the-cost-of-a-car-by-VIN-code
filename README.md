# Вступление

В данном проекте была разработана модель машинного обучения для предсказания стоимости автомобиля по его VIN-коду. VIN-код (Vehicle Identification Number) - это уникальный идентификатор транспортного средства, который содержит информацию о его производителе, модели, годе выпуска, характеристиках и других параметрах.

# Цель проекта

Основной целью проекта было создание модели, способной предсказывать стоимость автомобиля с приемлемой точностью на основе информации, извлеченной из его VIN-кода.

# Данные

Для обучения и тестирования модели использовался набор данных, содержащий VIN-коды автомобилей и соответствующие цены. Данные были предварительно обработаны с использованием регулярных выражений для извлечения информации о производителе, модели, двигателе, годе выпуска и других характеристиках.

# Методология

1. **Подготовка данных:** Данные были очищены, категориальные признаки были закодированы.
2. **Выбор инструмента:** Для моделирования был выбран Python с библиотекой PyCaret, которая автоматизирует многие задачи машинного обучения.
3. **Сравнение моделей:** PyCaret был использован для сравнения различных моделей машинного обучения, таких как линейная регрессия, случайный лес, XGBoost и другие. XGBoost показал наилучшие результаты по метрикам оценки.
4. **Настройка модели:** Гиперпараметры модели XGBoost были настроены для дальнейшего повышения точности.
5. **Оценка модели:** Производительность модели была оценена на тестовом наборе данных и сравнена с константной моделью. Модель XGBoost значительно превзошла константную модель, демонстрируя свою эффективность.
6. **Анализ важности признаков:** Был проведен анализ, чтобы определить, какие признаки вносят наибольший вклад в предсказание цены. Модель автомобиля оказалась наиболее важным признаком.
7. **Создание функции предсказания:** Была написана функция `predict_car_price` и класс `CarPricePredictor`, которые принимает VIN-код (или список кодов), извлекает необходимую информацию, предсказывает цену автомобиля и выводит результаты.
8. **Создание приложения** Для еще более удобного взаимодействия с моделью было разработано интерактивное веб-приложение с использованием библиотеки Streamlit.

# Результаты

Разработанная модель XGBoost показала высокую точность предсказания стоимости автомобилей по VIN-кодам. Функция `predict_car_price` и класс `CarPricePredictor`позволяют удобно использовать модель для предсказания цены по одному или нескольким VIN-кодам.

# Streamlit-приложение
Ссылка на веб-приложение https://predicting-the-cost-of-a-car-by-vin-code.streamlit.app/

Для еще более удобного взаимодействия с моделью было разработано интерактивное веб-приложение с использованием библиотеки Streamlit. Приложение позволяет пользователям вводить VIN-код автомобиля, выбирать язык вывода (русский или английский) и получать предсказание стоимости. Также доступна опция расшифровки VIN-кода для получения подробной информации об автомобиле.

## Функциональность

*   Ввод VIN-кода
*   Предсказание цены
*   Расшифровка VIN-кода (опционально)
*   Выбор языка (русский или английский)

## Использование

1.  Перейдите по ссылке: https://predicting-the-cost-of-a-car-by-vin-code.streamlit.app/
2.  Введите VIN-код автомобиля.
3.  Выберите язык (русский или английский).
4.  (Опционально) Отметьте опцию "Расшифровывать VIN", чтобы получить подробную информацию о VIN-коде.
5.  Нажмите кнопку "Предсказать" или "Predict".

# Выводы

Проект успешно достиг поставленной цели – создания модели для предсказания стоимости автомобиля по VIN-коду и разработки удобного приложения для ее использования. Полученные результаты демонстрируют потенциал использования машинного обучения для решения подобных задач.

# Дальнейшие шаги

* **Сбор большего количества данных** для повышения точности и обобщающей способности модели.
* **Эксперименты с другими моделями** или ансамблевыми методами для поиска еще более эффективных решений.

# Заключение

Этот проект демонстрирует возможности машинного обучения в области оценки стоимости автомобилей. Разработанная модель, функция предсказания и ***Streamlit-приложение***  могут быть полезны для частных лиц, автодилеров и других организаций, работающих с автомобилями.

# Статус
**Завершён**

# Стэк

***NumPy, Pandas, PyCaret, scikit-learn, Matplotlib, re, time, skimpy, chime, Pipeline, CatBoost, XGBoost, LightGBM***