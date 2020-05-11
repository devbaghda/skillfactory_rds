''' Компьютер загадывает число от 1 до 100 и наша программа его должна отгадать.
    Логика поиска будет следующей: первая цифра (predct) подбирается наугад. Если она меньше/больше задуманной,
    то интервал подбора значений сужается снизу/сверху соответственно'''

import numpy as np

def predictor(numb):
    #устанавливаем верхнюю и нижнюю границу интервала для подпбора значений:
    lst_a=[0]
    lst_b=[100]

    count=0

    for i in range(1,101):

        predct=np.random.randint(max(lst_a),min(lst_b))
   #исключаеем вероятность повторных подборов значений:
        if predct in lst_a or predct in lst_b:
            continue

        count+=1

        if  predct==numb:
            break

        elif predct<numb:
            a=predct
            lst_a.append(a)
        else:
            b=predct
            lst_b.append(b)

    return count

def multi_try(predictor):

    count_lst=[]

    np.random.seed(1)
    random_array=np.random.randint(1,101,size=1000)

    for numb in random_array:

        count_lst.append(predictor(numb))

    result=int(np.mean(count_lst))

    print (f'Программа угадывает число в среднем за {result} раз(а)')

multi_try(predictor)
