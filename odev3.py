import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold,cross_val_score,cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV,LassoCV,ElasticNetCV, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from mnist import MNIST

mnist = datasets.load_digits()
X = mnist.data
y = mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)

model = Perceptron()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print(classification_report(y_test, y_pred))
print ("Confusion matrix")
print(confusion_matrix(y_test, y_pred))

# i - çok sınıflı problemlerde her sınıf için ayrı bir precion, recall özellikleri mevcut.
# Macro ortalamalarını alır. bu soruda spesifik olarak bir sütunun yani 1'in precision değerini
# nası hesaplayabileceğimizi sormuş, bunun için formül şu şekildedir
# precision = (true_positives / (true_positives + false_positives))
conf = confusion_matrix(y_test, y_pred)
summary = 0

true_positives = conf[1, 1]
arr = conf[:, 1]
s = 0
for j in range(0,len(arr)):
    s = s + arr[j]

false_positives = s - true_positives
precision = (true_positives / (true_positives + false_positives))

print("Precision for 1: " , precision)

# ii - iters ve tols parametreleri üzerinde değişiklikler yaparak
# Perceptron'un performansını geliştirmemizi istemiş 
# Aşağıdaki parametrelerde değişikliklere gidip farklı sonuçlar gözlemleyebiliriz

iters = [1000,1500,2000,10000]
tols = [0.001,0.1,0.5,0.9,0.0001]

for i in range (0,len(iters)):
    for j in range (0,len(tols)):
        model = Perceptron(max_iter=iters[i], tol=tols[j])
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        resultStr = "iter : " + str(iters[i]) + " tol : " + str(tols[j]) + " score : " + str(score)
        print(resultStr)

# iii - SVC classını kernel = linear ile kullanarak C ve decision_function_shape parametrelerinde 
# değişiklikler yaparak test etmemizi ve sonuçları yorumlamamızı istemiş
# burda sonuçlarda pek artış göremiyorum sebebi için de bir fikrim yok açıkçası
# Allahın takdiri diyelim geçelim

shapes = ['ovr','ovo']
cValues = [0.0001,0.001,0.1,0.5,1.0,1.1,1.5,2.0,2.5,5.0,9.0,10000]
for i in range(0,len(shapes)):
    for j in range(0,len(cValues)):
        svc = SVC(kernel='linear',decision_function_shape=shapes[i],C=cValues[j])
        model = svc.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print("decision_function_shape: " ,shapes[i] , "C : ", cValues[j], " Score: " , score)

# iv - linearSVC classı için dual parametresini true, false olarak denememizi
# hangisinin daha iyi sonuç verdiğini ve nedenini istemiş 
# daul = false kullanmak daha hızlıdır ve hafıza kullanımı daha az olacaktır
# çünkü dual false verildiği için ayrı bir sınıflandırıcı eğitmez
# çok büyük olmayan veri setlerinde false değeri daha iyi sonuçlar verir

dualParameter = False
for i in range (0,2):
    linearSvc = LinearSVC(dual=dualParameter)
    model = linearSvc.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Dual : ", dualParameter, " Score : ", score)
    dualParameter = not dualParameter


# v - logisticRegression classının solvers parametresini değiştirerek denememiz istenmiş
# bunun için önce farklı solver değerlerini bir array'e attım ve for döngüsü ile 3 farklı 
# şekilde oluşturulmuş logisticRegression classının score'unu ekrana yazdırdım
# sonuçlara göre en iyi score'u sağlayan solver : 'newton-cg'

solvers = ['newton-cg','lbfgs','liblinear']
for i in range(0,len(solvers)):
    logisticRegression = LogisticRegression(solver=solvers[i])
    model = logisticRegression.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Solver : ", solvers[i], " Score : ", score)