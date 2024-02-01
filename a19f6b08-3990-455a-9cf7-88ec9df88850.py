


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, r2_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from scipy import stats as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer
from sklearn.dummy import DummyRegressor


# In[2]:


#Данные находятся в трёх файлах:
#gold_recovery_train_new.csv — обучающая выборка;
#gold_recovery_test_new.csv — тестовая выборка;
#gold_recovery_full_new.csv — исходные данные.

data_train = pd.read_csv('/datasets/gold_recovery_train_new.csv')
data_test = pd.read_csv('/datasets/gold_recovery_test_new.csv')
data_full = pd.read_csv('/datasets/gold_recovery_full_new.csv')

display(data_train.head(5))
display(data_test.head(5))
display(data_full.head(5))
display(data_train.info())
display(data_test.info())
display(data_full.info())
display(data_train.describe())
display(data_test.describe())
display(data_full.describe())


print(data_test.isna().sum())
print(data_full.isna().sum())



print('Размер обучающей выборки: {} '.format(data_train.shape))
print('Размер тестовой выборки: {} '.format(data_test.shape))
print('Размер общей выборки: {} '.format(data_full.shape))


# Формула для расчета 
# 
# recovery = ((C * (F - T)) / (F * (C - T))) * 100%
# 
# где:
# 
# C — доля золота в концентрате после флотации/очистки;
# F — доля золота в сырье/концентрате до флотации/очистки;
# T — доля золота в отвальных хвостах после флотации/очистки.
# 
# Наименование признаков должно быть такое:
# [этап].[тип_параметра].[название_параметра]
# Пример: rougher.input.feed_ag
# 
# C = data_train['rougher.output.concentrate_au']
# F = data_train['rougher.input.feed_au']
# T = data_train['rougher.output.tail_au']

# In[6]:


#Проверьте, что эффективность обогащения рассчитана правильно. 

C = data_train['rougher.output.concentrate_au']
F = data_train['rougher.input.feed_au']
T = data_train['rougher.output.tail_au']

def effectiveness_recovery(C, F, T):
    try:
        recovery = ((C * (F - T)) / (F * (C - T))) * 100
        return recovery
    except:
        return np.nan 
print(effectiveness_recovery(C, F, T))

#Вычислите её на обучающей выборке для признака rougher.output.recovery. 

data_train['rougher.output.recovery'].describe()


#Найдите MAE между вашими расчётами и значением признака. Опишите выводы.

mae=mean_absolute_error(effectiveness_recovery(C, F, T),data_train['rougher.output.recovery'])
print("MAE:", mae)


# MAE дает нам понять, насколько точно и показывает отклонение от значении. Здесь мы видим, что MAE нашего рассчета и значениями признака равна 9.682896147825551e-15= = 9.682896147825551 ∙ 10-15 = 9.682896147825551 ∙ 0.000000000000001 = 0.00000000000000968
# 
# Разница очееень маленькая. Посмотрим на графике. 


recovery = effectiveness_recovery(C, F, T)
plt.hist([data_train['rougher.output.recovery'], recovery])


# Ранее проверяли размеры выборок и получили такие результаты.
# Размер обучающей выборки: (14149, 87) 
# Размер тестовой выборки: (5290, 53) 
# Размер общей выборки: (19439, 87) 
# 
# 87-53 = 34 отсутствующих признаков. Проверим что за параметры. 

# In[8]:


#Проанализируйте признаки, недоступные в тестовой выборке. Что это за параметры? К какому типу относятся?
train_cols = data_train.columns
test_cols = data_test.columns

common_cols = train_cols.intersection(test_cols)
train_not_test = train_cols.difference(test_cols)

print('Общие колонки', len(common_cols), common_cols)
print('Недоступные колонки', len(train_not_test), train_not_test)


#1) тест наращиваем (мержим/джоиним) ТОЛЬКО таргетом c помощью поля дата.
data_test = data_test.merge(data_full[['date','rougher.output.recovery','final.output.recovery']], on='date', how='left')

data_test.info()

print(data_test['rougher.input.feed_ag'])



train = data_train.columns.to_list()
test = data_test.columns.to_list()

cols_to_drop = list(set(train) - set(test))
print(cols_to_drop)
data_train = data_train.drop(cols_to_drop, axis=1)


# In[11]:


data_train.info()
data_test.info()


# In[ ]:

data_train = data_train.fillna(method='ffill')
data_train = data_train.dropna()
display(data_train.isna().sum())
display(data_train.shape)

data_test = data_test.fillna(method='ffill')
data_test.info()



print('Размер обучающей выборки: {} '.format(data_train.shape))
print('Размер тестовой выборки: {} '.format(data_test.shape))
print('Размер общей выборки: {} '.format(data_full.shape))



for name in ['ag', 'pb', 'au']:
    data_full_m = data_full[["rougher.output.concentrate_" + name,"primary_cleaner.output.concentrate_" + name, "final.output.concentrate_" + name]]
    data_full_m.plot(kind='hist',alpha=0.7,
        bins=30,
        title='Гистограмма концентрации металлов на различных этапах очистки',
        rot=45,
        grid=True,
        figsize=(8,5))


# Проверили на гистограмме наглядно как меняется концентрация металлов (Au, Ag, Pb) на различных этапах очистки. Концентрация AU после всех этапов очистки возврастает по сравнению с другими металлами. 




#2.2. Сравните распределения размеров гранул сырья на обучающей и тестовой выборках. 
#Если распределения сильно отличаются друг от друга, оценка модели будет неправильной.

#ax = data_train['rougher.input.feed_size'].plot()
#data_test['rougher.input.feed_size'].plot(ax=ax, kind='bar', figsize=(8,5))

data_train[['rougher.input.feed_size']].plot(kind='hist')
plt.show()
data_test[['rougher.input.feed_size']].plot(kind='hist')
plt.show()
data_train[['rougher.input.feed_size']].boxplot()
plt.show()
data_test[['rougher.input.feed_size']].boxplot()
plt.show()
display(data_test[['rougher.input.feed_size']].describe())
display(data_train[['rougher.input.feed_size']].describe())


# Распределения отличаются друг от друга. Возможно из за количества выборок в тестовых и обучаю. данных.



#2.3. Исследуйте суммарную концентрацию всех веществ на разных стадиях: в сырье, в черновом и финальном концентратах.


#test data

data_full['rougher_sum']=data_full['rougher.input.feed_ag']+data_full['rougher.input.feed_pb']+data_full['rougher.input.feed_sol']+data_full['rougher.input.feed_au']

data_full['primary_sum']=data_full["primary_cleaner.output.concentrate_ag"]+data_full["primary_cleaner.output.concentrate_pb"]+data_full['primary_cleaner.output.concentrate_sol']+data_full['primary_cleaner.output.concentrate_au']  

data_full["final_sum"]=data_full["final.output.concentrate_ag"]+data_full["final.output.concentrate_pb"]+data_full['final.output.concentrate_sol']+data_full['final.output.concentrate_au']

data_full_m = data_full[["rougher_sum", "primary_sum","final_sum"]]
data_full_m.plot(kind='hist',alpha=0.7,
        bins=30,
        title='Гистограмма суммарной концентрации металлов на различных этапах очистки в сырье, в черновом и финальном концентратах',
        rot=45,
        grid=True,
        figsize=(8,5))


# Наблюдаются нулевые значения в обучающей выборке больше. Возможно аномалии, нужно их удалить и проверить заново. 
# 
# 

# In[18]:


data_full=data_full.loc[(data_full['rougher_sum']>20) & (data_full['primary_sum']>20) & (data_full['final_sum']>20)]

#data_train=data_train.loc[(data_train['rougher_sum']>20) & (data_train['primary_sum']>20) & (data_train['final_sum']>20)]


# In[19]:


data_full_m = data_full[["rougher_sum", "primary_sum","final_sum"]]
data_full_m.plot(kind='hist',alpha=0.7,
        bins=30,
        title='Гистограмма суммарной концентрации металлов на различных этапах очистки в сырье, в черновом и финальном концентратах',
        rot=45,
        grid=True,
        figsize=(8,5))




data_full = data_full.replace(0,np.nan)
data_test = data_test.replace(0, np.nan)
data_train = data_train.replace(0, np.nan)
print(data_full.isna().sum())
print(data_test.isna().sum())
print(data_train.isna().sum())
data_test.dropna(inplace=True)
data_train.dropna(inplace=True)
data_full.dropna(inplace=True)



data_test.drop(['date'], axis=1, inplace=True)
data_train.drop(['date'], axis=1, inplace=True)


# ## Модель

# Нужно спрогнозировать сразу две величины:
# * эффективность обогащения чернового концентрата rougher.output.recovery; 
# * эффективность обогащения финального концентрата final.output.recovery. 
# Итоговая метрика складывается из двух величин:
# 

# In[21]:


#3.1. Напишите функцию для вычисления итоговой sMAPE.
def smape(target, predict):
    try:
        smape=((1/len(target)) * np.sum(2 * np.abs(target-predict) / (np.abs(target) + np.abs(predict)))) * 100
        return smape
    except:
        return np.nan
    
def smape_final(smape_rough, smape_final):
    smape_final = 0.25 * smape_rough + 0.75 * smape_final
    return smape_final
    




#3.2. Обучите разные модели и оцените их качество кросс-валидацией. 
target_rougher = data_train['rougher.output.recovery']
features_rougher = data_train.drop(['rougher.output.recovery'], axis=1)

features_rougher_train, features_rougher_valid, target_rougher_train, target_rougher_valid = train_test_split(
        features_rougher, target_rougher, test_size=0.25, random_state=12345)

target_final = data_train['final.output.recovery']

features_final = data_train.drop(['final.output.recovery'], axis=1)

features_final_train, features_final_valid, target_final_train, target_final_valid = train_test_split(
        features_final, target_final, test_size=0.25, random_state=12345)


features_train = data_train.drop(['rougher.output.recovery', 'final.output.recovery'], axis=1)
target_train_rougher = data_train['rougher.output.recovery']
target_train_final = data_train['final.output.recovery']


scaler_r = StandardScaler()
scaler_r.fit(features_rougher_train)
features_rougher_train = scaler_r.transform(features_rougher_train)
features_rougher_valid = scaler_r.transform(features_rougher_valid)


scaler_f= StandardScaler()
scaler_f.fit(features_final_train)
features_final_train = scaler_f.transform(features_final_train)
features_final_valid = scaler_f.transform(features_final_valid)


# In[23]:


smape_scorer = make_scorer(smape, greater_is_better=False)


# In[24]:


features_rougher.shape


import warnings
warnings.filterwarnings('ignore')

model = LinearRegression()
scores = cross_val_score(model, features_train, target_train_rougher, scoring=smape_scorer, cv=5) 
final_score = sum(scores) / len(scores)
print('Средняя оценка качества модели LR:', final_score)

model = DecisionTreeRegressor(random_state=12345)
scores = cross_val_score(model, features_train, target_train_rougher, scoring=smape_scorer, cv=5)
final_score = sum(scores) / len(scores)
print('Средняя оценка качества модели DT:', final_score)

model = RandomForestRegressor(random_state=12345)
scores = cross_val_score(model, features_train, target_train_rougher, scoring=smape_scorer, cv=5)
final_score = sum(scores) / len(scores)
print('Средняя оценка качества модели RF:', final_score)


model = LinearRegression()
scores = cross_val_score(model, features_train,target_train_final, scoring=smape_scorer, cv=5) 
final_score = sum(scores) / len(scores)
print('Средняя оценка качества модели LR final:', final_score)

model = DecisionTreeRegressor(random_state=12345)
scores = cross_val_score(model, features_train,target_train_final, scoring=smape_scorer, cv=5)
final_score = sum(scores) / len(scores)
print('Средняя оценка качества модели DT final:', final_score)

model = RandomForestRegressor(random_state=12345)
scores = cross_val_score(model, features_train,target_train_final, scoring=smape_scorer, cv=5)
final_score = sum(scores) / len(scores)
print('Средняя оценка качества модели RF final:', final_score)



features_train = data_train.drop(['rougher.output.recovery', 'final.output.recovery'], axis=1)
target_train_rougher = data_train['rougher.output.recovery']
target_train_final = data_train['final.output.recovery']



random_state = 12345
cv = 5
models = [DecisionTreeRegressor(random_state = random_state), 
          RandomForestRegressor(random_state=random_state), 
          LinearRegression()]


results_cross_val = []

for model in models: 
    
    scorer = make_scorer(smape, greater_is_better=False) 
    
    cross_val_score_rougher = cross_val_score(model, 
                                              features_train, 
                                              target_train_rougher, 
                                              cv=cv, scoring=scorer).mean()
    cross_val_score_final = cross_val_score(model, 
                                            features_train, 
                                            target_train_final, 
                                            cv=cv, scoring=scorer).mean()

    results_cross_val.append({'model name': model.__class__.__name__, 
                              'cross_val_score_rougher': cross_val_score_rougher, 
                              'cross_val_score_final': cross_val_score_final}) 
              
pd.DataFrame(results_cross_val)

random_state = 123
cv = 5
models = [DecisionTreeRegressor(random_state = random_state), 
          RandomForestRegressor(random_state=random_state), 
          LinearRegression()]


results_cross_val = []

for model in models: 
    
    scorer = make_scorer(smape, greater_is_better=False) 
    
    cross_val_score_rougher = cross_val_score(model, 
                                              features_train, 
                                              target_rougher_train, 
                                              cv=cv, scoring=scorer).mean()
    cross_val_score_final = cross_val_score(model, 
                                            features_train, 
                                            target_final_train, 
                                            cv=cv, scoring=scorer).mean()

    results_cross_val.append({'model name': model.__class__.__name__, 
                              'cross_val_score_rougher': cross_val_score_rougher, 
                              'cross_val_score_final': cross_val_score_final}) 
              
pd.DataFrame(results_cross_val)
# In[27]:


#начнем с Linear Regression

def model_LR(features_train,target_train, features_valid,target_valid, target_type):
    model = LinearRegression()
    model.fit(features_train, target_train) #Обучите модель и сделайте предсказания на валидационной выборке.
    predictions_valid = model.predict(features_valid) #Обучите модель и сделайте предсказания на валидационной выборке.
    print("Linear Regression результат для" , target_type)
    #print(mean_absolute_error(target_valid, predictions_valid))
    result = mean_squared_error(target_valid, predictions_valid)**0.5
    predictions_valid_mean = pd.Series(predictions_valid).mean()
    scores = cross_val_score(model, features_train, target_train, scoring=smape_scorer, cv=5) 
    # < посчитайте оценки, вызвав функцию cross_value_score с пятью блоками >
    final_score = sum(scores) / len(scores)
    print('sMAPE:', smape(target_valid, predictions_valid))
    print('Средняя оценка качества модели:', final_score)
    print("Среднее предсказанного на валидационной выборке:",target_type, predictions_valid_mean)
    print("RMSE модели линейной регрессии на валидационной выборке:", result)


# In[28]:


model_LR(features_rougher_train,target_rougher_train, features_rougher_valid,target_rougher_valid, 'rougher train')


# In[29]:


model_LR(features_final_train,target_final_train, features_final_valid,target_final_valid, 'final train')


# In[30]:


best_model = None
best_result = 10000
def model_DT(features_train,target_train, features_valid,target_valid, target_type):
    for depth in range(1, 10, 1):
                model = DecisionTreeRegressor(max_depth=depth, random_state=12345)#initialize 
                model.fit(features_train, target_train) #Обучите модель и сделайте предсказания на валидационной выборке.
                predictions_valid = model.predict(features_valid) #Обучите модель и сделайте предсказания на валидационной выборке.
                result = mean_squared_error(target_valid, predictions_valid)**0.5
                predictions_valid_mean = pd.Series(predictions_valid).mean()
                scores = cross_val_score(model, features_train, target_train, scoring=smape_scorer, cv=5) 
    # < посчитайте оценки, вызвав функцию cross_value_score с пятью блоками >
                final_score = sum(scores) / len(scores)
                if result < best_result: 
                    best_model = model
                    best_result_dt = result
                    best_depth = depth
                    print("RMSE наилучшей модели на валидационной выборке:", best_result_dt)
                    print("Количество деревьев:", best_depth)
                    print('sMAPE:', smape(target_valid, predictions_valid))
                    print('Средняя оценка качества модели:', final_score)
                    print("Среднее предсказанного на валидационной выборке:",target_type, predictions_valid_mean)
                    print("RMSE модели на валидационной выборке:", result)
                    print('******') 


# In[31]:


model_DT(features_rougher_train,target_rougher_train, features_rougher_valid,target_rougher_valid, 'rougher train')


# Лучший результат:
# RMSE наилучшей модели на валидационной выборке: 4.609555010493079
# Количество деревьев: 8
# sMAPE: 3.7434498263915024
# Средняя оценка качества модели: -3.8547235537276663
# Среднее предсказанного на валидационной выборке: rougher train 84.48734759071743
# RMSE модели на валидационной выборке: 4.609555010493079
# <br>
# <font color='red'>   

# 

# In[32]:


model_DT(features_final_train,target_final_train, features_final_valid,target_final_valid, 'final train')


# Лучший результат:
# RMSE наилучшей модели на валидационной выборке: 7.0250717271292125
# Количество деревьев: 7
# sMAPE: 7.176764692622594
# Средняя оценка качества модели: -7.1967734902648655
# Среднее предсказанного на валидационной выборке: final train 66.81511323017678
# RMSE модели на валидационной выборке: 7.0250717271292125
# <br>
# <font color='red'>   

# In[33]:


best_model = None
best_result = 10000
def model_RF(features_train,target_train, features_valid,target_valid, target_type):
    for est in range(1,10):
        for depth in range(1, 5):
                    model = RandomForestRegressor(max_depth=depth, random_state=12345, n_estimators=est)#initialize 
                    model.fit(features_train, target_train) #Обучите модель и сделайте предсказания на валидационной выборке.
                    predictions_valid = model.predict(features_valid) #Обучите модель и сделайте предсказания на валидационной выборке.
                    result = mean_squared_error(target_valid, predictions_valid)**0.5
                    predictions_valid_mean = pd.Series(predictions_valid).mean()
                    scores = cross_val_score(model, features_train, target_train, scoring=smape_scorer, cv=5) 
        # < посчитайте оценки, вызвав функцию cross_value_score с пятью блоками >
                    final_score = sum(scores) / len(scores)
                    if result < best_result: 
                        best_model = model
                        best_result_dt = result
                        best_depth = depth
                        print("RMSE наилучшей модели на валидационной выборке:", best_result_dt)
                        print("Est:", est)
                        print("Количество деревьев:", best_depth)
                        print('sMAPE:', smape(target_valid, predictions_valid))
                        print('Средняя оценка качества модели:', final_score)
                        print("Среднее предсказанного на валидационной выборке:",target_type, predictions_valid_mean)
                        print("RMSE модели на валидационной выборке:", result)
                        print('******') 


# In[34]:


model_RF(features_rougher_train,target_rougher_train, features_rougher_valid,target_rougher_valid, 'rougher train')


# RMSE наилучшей модели на валидационной выборке: 3.5477861403009916
# Est: 5
# Количество деревьев: 4
# sMAPE: 2.9654368149251358
# Средняя оценка качества модели: -2.94365773872709
# Среднее предсказанного на валидационной выборке: rougher train 84.59195851351531
# RMSE модели на валидационной выборке: 3.5477861403009916




model_RF(features_final_train,target_final_train, features_final_valid,target_final_valid, 'final train')


# RMSE наилучшей модели на валидационной выборке: 5.491639392775829
# Est: 8
# Количество деревьев: 4
# sMAPE: 6.103941449917389
# Средняя оценка качества модели: -6.352209230744545
# Среднее предсказанного на валидационной выборке: final train 66.78613432141124
# RMSE модели на валидационной выборке: 5.491639392775829

# #Выберите лучшую модель и проверьте её на тестовой выборке. Опишите выводы.
# 
# Из всех моделей лучше всего показала LinearRegression с результатами
# 
# Linear Regression результат для rougher train
# sMAPE: 1.3887807228221332
# Средняя оценка качества модели: -1.4168984772221451
# Среднее предсказанного на валидационной выборке: rougher train 84.53551601257541
# RMSE модели линейной регрессии на валидационной выборке: 1.7878433843787123
# 
# 
# Linear Regression результат для final train
# sMAPE: 1.9557717600038136
# Средняя оценка качества модели: -2.023075669139406
# Среднее предсказанного на валидационной выборке: final train 66.68045701987354
# RMSE модели линейной регрессии на валидационной выборке: 2.281346649055247
# 
# 
# 


# RMSE наилучшей модели на валидационной выборке: 7.068096449214682
# Est: 9
# Количество деревьев: 4
# sMAPE: 7.277031720644452
# Средняя оценка качества модели: -7.1336197400636525
# Среднее предсказанного на валидационной выборке: final train 66.82555778307571
# RMSE модели на валидационной выборке: 7.068096449214682
# <br>
# <font color='red'>  
# 
# 

# In[25]:


#3.2. Обучите разные модели и оцените их качество кросс-валидацией. 

target_rougher_test = data_test['rougher.output.recovery']
features_rougher_test = data_test.drop(['rougher.output.recovery'], axis=1)

target_final_test = data_test['final.output.recovery']
features_final_test = data_test.drop(['final.output.recovery'], axis=1)

#features = data_train.drop(['rougher.output.recovery', 'final.output.recovery'], axis=1)

scaler_r = StandardScaler()
scaler_r.fit(features_rougher_test)
features_rougher_test = scaler_r.transform(features_rougher_test)

scaler_f = StandardScaler()
scaler_f.fit(features_final_test)
features_final_test = scaler_f.transform(features_final_test)


print('test', target_rougher_test.shape)
print('test', features_rougher_test.shape)

print('train',features_rougher_train.shape)
print('train',target_rougher_train.shape)

# Лучшей моделью оказалась Linear Regression, 

# In[37]:


model = LinearRegression()#initialize 
model.fit(features_rougher_train, target_rougher_train) #Обучите модель и сделайте предсказания на валидационной выборке.
predictions_valid = model.predict(features_rougher_test) #Обучите модель и сделайте предсказания на валидационной выборке.
result = mean_squared_error(target_rougher_test, predictions_valid)**0.5
predictions_valid_mean = pd.Series(predictions_valid).mean()
scores = cross_val_score(model, features_rougher_test, target_rougher_test, scoring=smape_scorer, cv=5) 
    # < посчитайте оценки, вызвав функцию cross_value_score с пятью блоками >
final_score = sum(scores) / len(scores)
smape_r = smape(target_rougher_test, predictions_valid)
print('sMAPE:', smape_r)
print('Средняя оценка качества модели:', final_score)
print("Среднее предсказанного на валидационной выборке:", predictions_valid_mean)
print("RMSE модели на валидационной выборке:", result)


model = LinearRegression()#initialize 
model.fit(features_final_train, target_final_train) #Обучите модель и сделайте предсказания на валидационной выборке.
predictions_valid = model.predict(features_final_test) #Обучите модель и сделайте предсказания на валидационной выборке.
result = mean_squared_error(target_final_test, predictions_valid)**0.5
predictions_valid_mean = pd.Series(predictions_valid).mean()
scores = cross_val_score(model, features_final_test, target_final_test, scoring=smape_scorer, cv=5) 
    # < посчитайте оценки, вызвав функцию cross_value_score с пятью блоками >
final_score = sum(scores) / len(scores)
smape_f = smape(target_final_test, predictions_valid)
print('sMAPE:', smape_f)
print('Средняя оценка качества модели:', final_score)
print("Среднее предсказанного на валидационной выборке:" , predictions_valid_mean)
print("RMSE модели на валидационной выборке:", result)

print(smape_final(smape_r,smape_f))



# Итоговый smape получился 7.184567242901533. 
# 

# Итоговый smape получился 9.814653057800577.



dummy_regressor_rougher = DummyRegressor(strategy="median")
dummy_regressor_rougher.fit(features_rougher_train, target_rougher_train)
dummy_rougher_pred = dummy_regressor_rougher.predict(features_rougher_test)
smape_dummy_rougher = smape(target_rougher_test, dummy_rougher_pred)
print('Rougher: ',smape_dummy_rougher) 

dummy_regressor_final = DummyRegressor(strategy="median")
dummy_regressor_final.fit(features_final_train, target_final_train)
dummy_final_pred = dummy_regressor_final.predict(features_final_test)
smape_dummy_final = smape(target_final_test, dummy_final_pred)
print('Final: ',smape_dummy_final) 



print(smape_final(smape_dummy_rougher,smape_dummy_final))


# In[ ]:


*********


# <div style="background: #cceeaa; padding: 5px; border: 1px solid green; border-radius: 5px;">
# <font color='green'> 
#     <b><u>КОММЕНТАРИЙ СТУДЕНТА</u></b>
# <font color='green'><br>  
# Итоговый smape получился 9.814653057800577 на линейной регрессии. 
# 
# На константной модели итоговый smape 7.321349364796761. Это говорит о том, что модель на линейной регрессии не совсем адекватна =( Возможно потому, что не правильно разделила данные. Жду подсказку =( <br>
#  
# <font color='green'>    
# <br>
dummy_regressor_rougher = DummyRegressor(strategy="median")
dummy_regressor_rougher.fit(X_train, y_train_rougher)
dummy_rougher_pred = dummy_regressor_rougher.predict(X_test)
smape_dummy_rougher = smape(y_test_rougher, dummy_rougher_pred)
print(smape_dummy_rougher) 






