#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.simplefilter('ignore')
sns.set()


# ### FUNCTIONS USED IN THIS EDA

# In[3]:


# возвращает первую и третью квантили, а также межквартильный диапазон,
# но не очищает выбросы


def calc_outliers(db_col):
    Q1 = db_col.quantile(0.25)
    Q3 = db_col.quantile(0.75)
    IQR = Q3-Q1
    return(Q1, Q3, IQR)


# In[4]:


# заменяет пустые значения в заданной колонке на значение 'exch'

def empty_to_value(column, exch):
    data[column] = data[column].apply(lambda x: exch if pd.isnull(x) else x)


# In[5]:


# если пустых данных в столбцах не более 5%, то их замена на медианное
# значение не может сильно исказить конечный результат


def act_w_categ(column):

    T = data[column].count()
    E = data[column].isna().sum()

    if E/T <= 0.05:
        empty_to_value(column, data[column].value_counts().index[0])
    else:
        print(f'В столбце {column} слишком много пустых значений:        {E} из {T}  ({round(E/T*100,2)}%).         Невозможно заполнить на данном этапе')


# In[6]:


# создает графическое пространство для построения графиков типа 'boxplot', для
# визуальной оценки влияния категорийных признаков на 'score'


def get_boxplot(column):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=column, y='score', data=data)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# In[7]:


# проводит тест Стьюдента для нулевой гипотезы, согласно которой показатели
# 'score' в рамках одного категорийного признака практически одинаковы


def get_stat_dif(column):
    cols = data.loc[:, column].value_counts().index
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(data.loc[data.loc[:, column] == comb[0], 'score'],
                     data.loc[data.loc[:, column] == comb[1], 'score']).pvalue <= 0.05/len(combinations_all):
            print('Найдены статистически значимые различия для колонки', column)
            break


# ### DATA LOADING AND OVERVIEW

# In[8]:


data = pd.read_csv('stud_math.csv')
data.info()


# числ 13 строк 17

# In[9]:


data.head(10)


# для удобства работы переименуем столбец 'studytime, granular' 'studytime_granular', а также разобъем колонки исходя из их содержания на числовые и категориальные

# In[10]:


data.rename(
    columns={'studytime, granular': 'studytime_granular'}, inplace=True)


# In[11]:


numeric = ['age', 'absences', 'score']
categorial = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Medu',
              'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime',
              'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
              'activities', 'nursery', 'higher',
              'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health']


# в задании отсутствует описание столбца 'studytime_granular' и в данных есть столбец 'studytime'. Проверим насколько эти два столбца скореллировны

# In[12]:


temp = pd.DataFrame({'col1': data.studytime, 'col2': data.studytime_granular})
temp.corr()


# очевидна полная обратная корелляция, поэтому столбец 'studytime_granular' можно убрать

# In[13]:


data = data.drop('studytime_granular', axis=1)


# #### numeric

# In[14]:


data[numeric].info()


# в числовых данных мало пропусков, можно заменить их на медианное значение

# In[15]:


for column in numeric:
    empty_to_value(column, data[column].median())


# In[16]:


data[numeric].info()


# ##### age

# In[17]:


data.age.describe()


# In[18]:


data.age.hist(bins=data.age.nunique())


# In[19]:


Q1 = calc_outliers(data.age)[0]
Q3 = calc_outliers(data.age)[1]
IQR = calc_outliers(data.age)[2]

data[(data['age'] > Q3+1.5*IQR) | (data['age'] < Q1-1.5*IQR)]['age'].value_counts()


# пустых строк нет; Распределение сдвинуто влево, заметно сильное сокращение учащихся старше 18 лет, что, предположительно, указывает на то, что большая часть учащихся заканчивает школу в возрасте не старше этого возраста. Единственного учащегося в возрасте 22 лет по формуле межквартильного диапазона можно было бы отсеить, как выброс, однако, можно предположить, что более возрастные учащиеся будут иметь низкую посещаемость и/или низкие оценки. Проверим при анализе. Если гипотеза подтвердится, это будет означать, что возрастные ученики входят в группу риска.  

# ##### absences

# In[20]:


data.absences.describe()


# Пустых значений нет. Очевидно разброс очень большой, проверим на наличие выбросов

# In[21]:


Q1 = calc_outliers(data.absences)[0]
Q3 = calc_outliers(data.absences)[1]
IQR = calc_outliers(data.absences)[2]

over_abs = data[~data.absences.between(Q1-1.5*IQR, Q3+1.5*IQR)]
over_abs


# In[22]:


over_abs.absences.describe()


# слишком много данных потеряем, если избавиться от выбросов, однако, судя по максимальному значению, есть очень сильно отличающиеся данные в самом диапазоне выбросов, попробуем отсечь хотя бы их

# In[23]:


Q1 = calc_outliers(over_abs.absences)[0]
Q3 = calc_outliers(over_abs.absences)[1]
IQR = calc_outliers(over_abs.absences)[2]


over_abs[~over_abs.absences.between(Q1-1.5*IQR, Q3+1.5*IQR)].absences


# In[24]:


data.absences = data[data.absences < 212].absences
data.absences.hist(bins=4)


# большая часть учащихся пропускает не более 20 занятий, при анализе надо проверить успеваемость остальных

# ##### score

# In[25]:


data.score.describe()


# Значений 'nan' не много; заполним их медианным значением

# In[26]:


data.score.hist(bins=data.score.nunique())


# Гистограма распределена нормально, но вызывает вопрос большое количество нулевых баллов. Возможно, это ошибочные данные, либо есть прямая связь с посещаемостью. 

# In[27]:


data[data.score == 0]


# во всех строках, где score =0, также =0 и  absences, при этом остальные значения заполнены различающимися значениями, что позволяет предположить, что в данных допущена ошибка и вместо нулей в score и absences должны стоять реальные пропуски занятий и оценки. Во избежаниe потери большого количества данных ровно как и их сильного искажения, заменим нули в обоих столбцах на их медианные значения

# In[28]:


data.score = data.score.apply(lambda x: data.score.median() if x == 0 else x)


# In[29]:


data.absences = data.absences.apply(
    lambda x: data.absences.median() if x == 0 else x)


# In[30]:


data.score.hist(bins=11, align='mid')


# Выделяется большое число 50-56 -балльных оценок, что говорит о большом количестве 'среднячков'. Поскольку наша задача состоит в построении модели, которая может определить учащихся, потенциально находящихся в группе риска, то следует обратить особое внимание на баллы до 50.  

# In[31]:


data.absences[data.absences < 40].hist(bins=4)


# большая часть учащихся (около 350) посещает занятия с минимальными пропусками - до 10. При анализе надо будет обратить отдельное вниамние на учащихся с большим количеством пропусков и проследить взаимосвязь с оценками 

# #### categorial

# In[32]:


data[categorial].info()


# много столбцов с пустыми значениями; заполним пустые значения наиболее часто встречающимися значениями конкретного столбца, только в тех столбцах где количество пустых значений не превышает 5% всех значений. Остальные пока заполнить не сможем.

# In[33]:


for column in categorial:
    act_w_categ(column)


# ### DATA EXPLORATION

# #### numeric

# In[34]:


sns.pairplot(data[numeric], kind='reg')


# In[35]:


data[numeric].corr()


# числовые показатели слабо скореллированы;
# 
# при подготовке числовых показателей был замечен ряд особенностей, проверим догадки:

# ##### age

# In[36]:


sns.boxplot(x='age', y='score', data=data[data.age >= 19])


# догадка по возрасту частично подтвердилась. Учащиеся в возрасте 21 и 22 года имеют низкие баллы (40 и ниже), при этом у учащихся в возрасте 20 лет баллы достаточно высокие. Это позволяет сделать вывод, что возраст учащихся является определяющим критерием при значениях выше 20 лет. 

# ##### absences

# In[37]:


fig = plt.subplots(figsize=(18, 4))
sns.boxplot(x='absences', y='score', data=data)


# догадка по пропускам также подтверждается. Учащиеся с количеством пропусков от 21 и выше в подавляющем большинстве имеют средний балл ниже 50, следовательно количество пропусков от 21 и выше следует считать еще одним критерием выявления студентов зоны риска

# #### categorial

# In[38]:


for column in categorial:
    get_boxplot(column)


# в столбце Fedu есть ошибка - значение 40, хотя, согласно описанию, значения здесь должны варьироваться от 0 до 4. Скорее, всего была допущена ошибка и это не 40, а 4.0. Исправим это:

# In[39]:


data.Fedu = data.Fedu.apply(lambda x: 4.0 if x == 40.0 else x)


# в столбце famrel тоже есть ошибка - значение '-1', скорее всего, это 1, исправим: 

# In[40]:


data.famrel = data.famrel.apply(lambda x: 1.0 if x == -1.0 else x)


# In[41]:


for column in categorial:
    get_boxplot(column)


# Проверим нулевую гипотезу о том, что распределениe баллов по различным параметрам неразличимы:

# In[42]:


for col in categorial:
    get_stat_dif(col)


# Как было замечено при анализе столбца score, в исследуемой выборке очень много среднячков, поэтому во всех боксплотах можно видеть примерный средний уровень оценок в районе 50-60 баллов. Поскольку нас интересует выявление студентов, находящихся в группе риска, то оценки ниже 50 имеют для нас особую важность. 
# 
# Столбец 'sex' имеет статистически значимые различия, однако при визуальном анализе боксплотов можно заметить, что в целом средний показатель по оценкам для разных полов неизменен, при этом большая часть оценок (Q1 и выше) находится над уровнем 50 баллов, поэтому в модель этот показатель включать не будем.

# ### CONCLUSIONS

# Данные в основном чистые, обнаружены всего пара незначительных ошибок, но при этом подавляющее большинство признаков содержит пустые значения. Из них в 14 было решено пропуски не заполнять, так как в них было более 5% пустых значений, и их заполнение могло исказить реальную картину. 
# 
# В модель будем включать следующие параметры: 
# - age
# - absences
# - address
# - Medu
# - Fedu
# - Mjob
# - Fjob
# - studytime
# - failures
# - schoolsup
# - goout
