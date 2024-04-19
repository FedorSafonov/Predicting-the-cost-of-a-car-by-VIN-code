# Вступление

В данном проекте была разработана модель машинного обучения для предсказания стоимости автомобиля по его VIN-коду. VIN-код (Vehicle Identification Number) - это уникальный идентификатор транспортного средства, который содержит информацию о его производителе, модели, годе выпуска, характеристиках и других параметрах.

# Цель проекта

Основной целью проекта было создание модели, способной предсказывать стоимость автомобиля с приемлемой точностью на основе информации, извлеченной из его VIN-кода.

# Данные

Для обучения и тестирования модели использовался набор данных, содержащий VIN-коды автомобилей и соответствующие цены. Данные были предварительно обработаны с использованием регулярных выражений для извлечения информации о производителе, модели, двигателе, годе выпуска и других характеристиках.

# Методология

1. **Подготовка данных:** Данные были очищены, обработаны пропущенные значения (если таковые имелись) и категориальные признаки были закодированы.
2. **Выбор инструмента:** Для моделирования был выбран Python с библиотекой PyCaret, которая автоматизирует многие задачи машинного обучения.
3. **Сравнение моделей:** PyCaret был использован для сравнения различных моделей машинного обучения, таких как линейная регрессия, случайный лес, XGBoost и другие. XGBoost показал наилучшие результаты по метрикам оценки.
4. **Настройка модели:** Гиперпараметры модели XGBoost были настроены для дальнейшего повышения точности.
5. **Оценка модели:** Производительность модели была оценена на тестовом наборе данных и сравнена с константной моделью. Модель XGBoost значительно превзошла константную модель, демонстрируя свою эффективность.
6. **Анализ важности признаков:** Был проведен анализ, чтобы определить, какие признаки вносят наибольший вклад в предсказание цены. Модель автомобиля оказалась наиболее важным признаком.
7. **Создание функции предсказания:** Была написана функция `predict_car_price`, которая принимает VIN-код (или список кодов), извлекает необходимую информацию, предсказывает цену автомобиля и выводит результаты.

# Результаты

Разработанная модель XGBoost показала высокую точность предсказания стоимости автомобилей по VIN-кодам. Функция `predict_car_price` позволяет удобно использовать модель для предсказания цены по одному или нескольким VIN-кодам.

# Выводы

Проект успешно достиг поставленной цели – создания модели для предсказания стоимости автомобиля по VIN-коду. Полученные результаты демонстрируют потенциал использования машинного обучения для решения подобных задач.

# Дальнейшие шаги

* **Сбор большего количества данных** для повышения точности и обобщающей способности модели.
* **Эксперименты с другими моделями** или ансамблевыми методами для поиска еще более эффективных решений.
* **Разработка пользовательского интерфейса** (веб-приложение или API) для удобного использования модели. 

# Заключение

Этот проект демонстрирует возможности машинного обучения в области оценки стоимости автомобилей. Разработанная модель и функция предсказания могут быть полезны для частных лиц, автодилеров и других организаций, работающих с автомобилями.

# Статус
**Завершён** (возможно создание веб-сервиса с помощью Streamlit)

# Стэк

***NumPy, Pandas, PyCaret, scikit-learn, Matplotlib, re, time, skimpy, chime, Pipeline, CatBoost, XGBoost, LightGBM***