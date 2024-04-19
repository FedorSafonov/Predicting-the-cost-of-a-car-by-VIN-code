#!/usr/bin/env python
# coding: utf-8

# # Предсказание стоимости автомобиля по его VIN-номеру

# ## Описание
# 
# **Вступление**
# 
# В данном проекте была разработана модель машинного обучения для предсказания стоимости автомобиля по его VIN-коду. VIN-код (Vehicle Identification Number) - это уникальный идентификатор транспортного средства, который содержит информацию о его производителе, модели, годе выпуска, характеристиках и других параметрах.
# 
# **Цель проекта**
# 
# Основной целью проекта было создание модели, способной предсказывать стоимость автомобиля с приемлемой точностью на основе информации, извлеченной из его VIN-кода.
# 
# **Данные**
# 
# Для обучения и тестирования модели использовался набор данных, содержащий VIN-коды автомобилей и соответствующие цены. Данные были предварительно обработаны с использованием регулярных выражений для извлечения информации о производителе, модели, двигателе, годе выпуска и других характеристиках.
# 

# ## Импорты

# Есле нужно установить библиотеки, то можно разкомментировать этот код

# In[ ]:


# pip install -r requirements.txt


# In[163]:


# Стандартные библиотеки
import re
import time

# Библиотеки для работы с данными
import numpy as np
import pandas as pd

# Библиотеки для машинного обучения
from pycaret.regression import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error as mae, mean_absolute_percentage_error as mape,
                             mean_squared_error as mse, r2_score as r2)

# Библиотеки для визуализации
import matplotlib.pyplot as plt
import seaborn as sns

# Прочие библиотеки
import chime
import skimpy

# Настройки
plt.style.use(('dark_background'))
chime.theme('mario')
get_ipython().run_line_magic('load_ext', 'chime')

# Ranndom state
S = 100500


# ## Загрузка и обзор данных

# In[164]:


pattern = r"\[(\w{17}):(\d+)\]"

with open("vin_ford_train.txt", "r") as file:
    data = file.readline()
    matches = re.findall(pattern, data)
    
    for match in matches[:5]:
        print(match)


# In[165]:


df = pd.DataFrame(matches, columns=["VIN", "Price"])
df.head()


# In[166]:


def extract_vin_info(vin):
    """Извлекает информацию из VIN-кода:

manufacturer         Производитель
security_codes       Ремни безопаности / тормоза
model                Модель
engine               Двигатель
check_digit          Контрольный знак
model_year           Год
plant_code           Завод
serial_number        Серийный номер"""
    
    pattern = r"(\w{3})(\w)(\w{3})(\w)(\w)(\w)(\w)(\w{6})"
    match = re.match(pattern, vin)
    if match:
        manufacturer = match.group(1)
        security_codes = match.group(2)
        model = match.group(3)
        engine = match.group(4)
        check_digit = match.group(5)
        model_year = match.group(6)
        plant_code = match.group(7)
        serial_number = match.group(8)
        return pd.Series(
            {
                "manufacturer": manufacturer,
                "security_codes": security_codes,
                "model": model,
                "engine": engine,
                "check_digit": check_digit,
                "model_year": model_year,
                "plant_code": plant_code,
                "serial_number": serial_number,
            }
        )
    else:
        return pd.Series(
            {
                "manufacturer": None,
                "security_codes": None,
                "model": None,
                "engine": None,
                "check_digit": None,
                "model_year": None,
                "plant_code": None,
                "serial_number": None,
            }
        )

vin_example = df["VIN"][0]

print('Информация о VIN-коде:')
extract_vin_info(vin_example)


# In[167]:


get_ipython().run_cell_magic('time', '', '%%chime\n\ndf[\n    [\n        "manufacturer",\n        "security_codes",\n        "model",\n        "engine",\n        "check_digit",\n        "model_year",\n        "plant_code",\n        "serial_number",\n    ]\n] = df["VIN"].apply(extract_vin_info)\n\ndf.head()\n')


# ## EDA (разведочный анализ)

# In[168]:


skimpy.skim(df)


# In[169]:


df.info()


# In[170]:


df.describe()


# In[171]:


(df[['Price']].astype('float')).describe()


# In[172]:


df['Price'] = df['Price'].astype('uint')


# In[173]:


df.drop(columns='VIN').duplicated().sum()


# - Пропусков не обнаружено.
# 
# - Price представляет собой целочисленные значения в диопозоне от 1000 дл 36500
# 
# - Есть дубликаты в serial_number, но полные дубликаты отсуствуют. 

# ### Price

# In[174]:


plt.hist(df['Price'], bins=20,color='coral')
plt.xlabel('Цена')
plt.ylabel('Количество автомобилей')
plt.title('Распределение цен автомобилей')
plt.show()


# Судя по гистограмме, распределение цен имеет вид, близкий к нормальному, но с некоторым смещением вправо. Большинство автомобилей имеют цену в диапазоне от 5000 до 15000, но есть и более дорогие варианты.

# ### Соотношение моделей и производителей

# In[175]:


display(df[['manufacturer']].value_counts())
print()
df[['model']].value_counts()


# In[176]:


df[['model']].value_counts().head(15)


# In[177]:


df[['model']].value_counts().tail(15)


# In[178]:


round(df.shape[0]/1000)


# In[179]:


percentage = round(df.shape[0]/1000)

print(f'0.1% данных: {percentage}')
print(f'Модели которые встречаються реже: {(df.model.value_counts() < percentage).sum()}')
print('----------------------------------')
print(f'0.1% данных: {round(percentage/10)}')
print(f'Модели которые встречаються реже: {(df.model.value_counts() < percentage/10).sum()}')
print('----------------------------------')
print(f'Модели которые встречаються 1 раз: {(df.model.value_counts() == 1).sum()}')


# Судя по результатам, в данных представлено 15 разных производителей и 365 разных моделей.
# 
# * **Производители**: Наиболее представленными производителями являются `1FM`, `1FT` и `1FA`. Остальные производители представлены в значительно меньшем количестве.
# * **Модели**: Среди моделей наблюдается большее разнообразие. Модель `P0H` является самой распространенной, но есть много моделей, которые представленые в небольшом количестве.

# ### Зависимость цены от признаков

# #### Год

# In[180]:


df[['model_year']].value_counts()


# In[181]:


sns.boxplot(x='model_year',y='Price', data=df)
plt.xlabel('Год выпуска')
plt.ylabel('Цена')
plt.title('Зависимость цены от года выпуска (Box Plot)')
plt.show()


# Можно сделать следующие наблюдения:
# 
# * Больше всего автомобилей распределены по годам D, E, C, B.
# * Есть годы в которых всего 1 автомобиль (H,K,P)
# * Год F обладает самыми дорогими автомобилями
# * Цены на автомобили варьируются в зависимости от года выпуска.
# * Наблюдается значительный разброс цен внутри каждого года выпуска. Это может быть связано с разными моделями, комплектациями и состоянием автомобилей.
# 

# #### Производитель

# In[182]:


sns.boxplot(x='manufacturer',y='Price', data=df)
plt.xlabel('Производитель')
plt.ylabel('Цена')
plt.title('Зависимость цены от производителя (Box Plot)')
plt.xticks(rotation=30)
plt.show()


# * Цены на автомобили варьируются в зависимости от производителя.
# * Производители `2FM` и `1FB` имеют в среднем более высокие цены, чем остальные.
# * Наблюдается значительный разброс цен внутри каждого производителя. Это может быть связано с разными моделями, комплектациями и состоянием автомобилей.

# #### Модель

# In[183]:


model_groups = (round(df.groupby('model')['Price'].agg(['mean', 'median', 'std', 'count']))
                .sort_values(by='count', ascending=False))
model_groups.head(30)


# In[184]:


top_30_models = model_groups.head(30)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
top_30_models['mean'].plot(kind='bar',colormap='summer')
plt.xlabel('Модель')
plt.ylabel('Средняя цена')
plt.title('Средняя цена для топ-30 моделей')

plt.subplot(1, 2, 2)
top_30_models['median'].plot(kind='bar')
plt.xlabel('Модель')
plt.ylabel('Медианная цена')
plt.title('Медианная цена для топ-30 моделей')

plt.tight_layout() 
plt.show()


# * **Распределение цен:** Как средние, так и медианные цены варьируются в зависимости от модели. Это указывает на то, что модель автомобиля является важным фактором, влияющим на его цену.
# * **Сходство между средней и медианной:** В большинстве случаев средняя цена и медианная цена для каждой модели довольно близки. Это говорит о том, что распределение цен внутри каждой модели относительно симметрично и не имеет сильных выбросов.
# 

# #### Двигатель

# In[185]:


df[['engine']].value_counts()


# In[186]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='engine', y='Price', data=df)
plt.xlabel('Код двигателя')
plt.ylabel('Цена')
plt.title('Зависимость цены от кода двигателя (Box Plot)')
plt.show()


# * **Распределение цен:** Цены на автомобили варьируются в зависимости от кода двигателя. Это может указывать на то, что разные коды соответствуют разным типам двигателей с разной стоимостью.
# * **Разброс цен:** Наблюдается значительный разброс цен внутри каждого кода двигателя. Это может быть связано с разными моделями, комплектациями и состоянием автомобилей.
# * **Выбросы:** Для некоторых кодов двигателя (например, 'B', 'S', 'V') наблюдаются выбросы, то есть автомобили с очень высокой ценой. Это может быть связано с редкими или особенными типами двигателей.
# * **Самые дорогие автомобилии** имеет двигатели `T`, `F`, `8`.
# 

# #### Безопасность (Ремни/тормоза)

# In[187]:


df['security_codes'].value_counts()


# In[188]:


sns.boxplot(x='security_codes', y='Price', data=df)
plt.xlabel('Безопасноть (4-й символ в VIN)')
plt.ylabel('Цена')
plt.title('Зависимость цены от производителя (Box Plot)')
plt.xticks(rotation=30) 
plt.show()


# In[189]:


security_codes_groups = (round(df.groupby('security_codes')['Price'].agg(['mean', 'median', 'std', 'count']))
                .sort_values(by='count', ascending=False))
security_codes_groups


# In[190]:


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
security_codes_groups['mean'].plot(kind='bar',colormap='summer',rot=0)
plt.xlabel('Код безопасности')
plt.ylabel('Средняя цена')
plt.title('Средняя цена')

plt.subplot(1, 2, 2)
security_codes_groups['median'].plot(kind='bar',rot=0)
plt.xlabel('Код безопасности')
plt.ylabel('Медианная цена')
plt.title('Медианная цена')

plt.tight_layout() 
plt.show()


# * **Распределение цен:** Цены на автомобили варьируются в зависимости от кода безопасности.
# * **Разброс цен:** Наблюдается значительный разброс цен внутри некоторых кодов.
# * **Выбросы:** Для некоторых кодов безопасности наблюдаются выбросы, то есть автомобили с очень высокой ценой. Это может быть связано с редкими или особенными типами двигателей.
# * **Самые дорогие автомобилии** среди топ-10 по количеству имеют код безопасности `5`, `J`, `8` 
# 

# #### Контрольный знак

# In[191]:


df[['check_digit']].value_counts()


# In[192]:


sns.boxplot(x='check_digit', y='Price', data=df)
plt.xlabel('Контрольный знак')
plt.ylabel('Цена')
plt.title('Зависимость цены от Контрольный знака (Box Plot)')
plt.xticks(rotation=0) 
plt.show()


# * **Распределение цен и автомобилей:** одинаково для всех контрольных знаков.
# * Этот признак не несёт ценности для обучения модели
# 

# #### Завод

# In[193]:


df[['plant_code']].value_counts()


# In[194]:


sns.boxplot(x='plant_code', y='Price', data=df)
plt.xlabel('Завод')
plt.ylabel('Цена')
plt.title('Зависимость цены от Завода (Box Plot)')
plt.xticks(rotation=0) 
plt.show()


# * **Влияние завода на цену:** Цены на автомобили демонстрируют заметную зависимость от завода-изготовителя.
# * **Разброс цен:** Наблюдается значительная вариация цен внутри продукции некоторых заводов (особенно завода E), что может указывать на разнообразие моделей или комплектаций.
# * **Объемы производства:** Наибольшее количество автомобилей производится на заводах R, K и L.
# * **Ценовой сегмент:**
# * **Премиум сегмент:** Самые дорогие автомобили (с точки зрения медианной цены) производятся на заводах F, G и B.
# * **Бюджетный сегмент:** Самые доступные автомобили производятся на заводах с наименьшим объемом выпуска, что может свидетельствовать о специализации на нишевых моделях или о меньшей эффективности производства.

# ### Серийный номер

# In[195]:


print('Количество дубликатов серийного номера')
df.duplicated('serial_number',False).sum()


# In[196]:


df[df.duplicated('serial_number',False)].sort_values('serial_number').head(20)


# * **Дубликаты в серийном номере:** достаточно большое количество (около 10%), но никаких закономерностей с другими признаками не обнаружено.
# * **Для обучения модели** признак не несёт пользы на данном этапе.

# In[197]:


df.info()


# In[37]:


df.to_csv('prepared_df.csv', index=False)


# ## Обучение и выбор модели

# In[38]:


df = df.set_index('VIN')


# In[39]:


X = df[[ 'manufacturer', 'security_codes', 'model', 'engine', 'model_year', 'plant_code']]
y = df[['Price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=S)


# In[40]:


train = pd.concat([X_train,y_train],axis=1)


# In[41]:


train.sample(5)


# In[42]:


X_test.head(3)


# In[43]:


y_test.head(3)


# ### OHE

# In[246]:


exp = setup(train,train_size=0.8,target='Price',rare_to_value=0.0001,max_encoding_ohe=300,session_id=S)


# In[248]:


get_ipython().run_cell_magic('time', '', "%%chime\nbest_model = exp.compare_models(sort='MAPE',include=['dummy','lightgbm','xgboost','catboost','et','rf'])\n")


# In[ ]:


exp.tune_model()


# In[252]:


exp.plot_model(best_model,'feature')


# ### TargetEncoder

# In[65]:


get_ipython().run_cell_magic('time', '', "%%chime\n\nexp2 = setup(train,train_size=0.8,target='Price',rare_to_value=0.0001,\n            max_encoding_ohe=3,session_id=S, normalize = True)\n            \nbest_model_2 = exp2.compare_models(sort='MAPE',include=['dummy','lightgbm','xgboost','catboost','et','rf'])\n")


# * Скорость возрала многогратно, метрика не ухудшилась. Оставляем этот варианат.
# * Константная модель в среднем ошибается на *`102 %`* , наша модель ошибается на *`20 %`* - проверка на адекватность пройдена.

# #### Важность признаков

# In[251]:


# plt.style.use('dark_background')
exp2.plot_model(best_model_2,'feature')


# Видно, что модель считает признак `"model"` (модель автомобиля) наиболее важным для предсказания цены. Это вполне логично, так как разные модели автомобилей имеют разную стоимость.

# #### Оптимизация (подбор оптимальных гиперпараметров)

# In[254]:


tuned_model = exp2.tune_model(best_model_2,optimize='MAPE',n_iter=30,search_library='optuna')


# In[256]:


tuned_model = exp2.tune_model(tuned_model,optimize='MAPE',n_iter=100, search_library='optuna')


# #### Финализация и сохранение модели

# In[258]:


exp2.save_experiment('experement')
exp2.save_model(tuned_model,'tuned_model')


# In[45]:


tuned_model = load_model('tuned_model')
exp = load_experiment('experement',train)


# In[46]:


final_model = exp.finalize_model(tuned_model)


# In[47]:


final_model


# In[48]:


save_model(final_model,'final_model')


# ## Тестирование модели

# In[52]:


preds = final_model.predict(X_test)


# In[53]:


mape(y_test, preds)


# In[66]:


print('MAPE', mape(y_test, preds).round(4))
print('MAE', mae(y_test, preds).round(2))
print('RMSE', np.sqrt(mse(y_test, preds)).round(2))


# In[50]:


predict_model(final_model,X_test.sample(1))


# ## Приминение

# In[157]:


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


# In[161]:


predict_car_price(['1FTGE2EW0EDA31393','1FANE2EC0EDA31393','3FANF2EE0EDA31393'], price_only=False)


# ## Итог
# 
# ### Результаты
# 
# Разработанная модель XGBoost показала высокую точность предсказания стоимости автомобилей по VIN-кодам. Функция `predict_car_price` позволяет удобно использовать модель для предсказания цены по одному или нескольким VIN-кодам.
# 
# ### Выводы
# 
# Проект успешно достиг поставленной цели – создания модели для предсказания стоимости автомобиля по VIN-коду. Полученные результаты демонстрируют потенциал использования машинного обучения для решения подобных задач.
# 
# ### Дальнейшие шаги
# 
# * **Сбор большего количества данных** для повышения точности и обобщающей способности модели.
# * **Эксперименты с другими моделями** или ансамблевыми методами для поиска еще более эффективных решений.
# * **Разработка пользовательского интерфейса** (веб-приложение или API) для удобного использования модели.
# 
# ### Заключение
# 
# Этот проект демонстрирует возможности машинного обучения в области оценки стоимости автомобилей. Разработанная модель и функция предсказания могут быть полезны для частных лиц, автодилеров и других организаций, работающих с автомобилями.
# 
