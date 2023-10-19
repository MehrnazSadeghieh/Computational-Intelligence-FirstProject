import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.svm import SVC
from scipy.stats import kurtosis
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
from scipy.stats import skew


import random
import os


print("------------------------------------ start --------------------------------")
seed = 57

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)


x = pickle.load(open('x.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

x_normal = np.concatenate((x[:300], x[400:]), axis=0)
x_seizure = x[300:400]
# print(x_normal.shape)
# print(x_seizure.shape)
sampling_freq = 173.6 #based on info from website

b, a = butter(3, [0.5,40], btype='bandpass',fs=sampling_freq)


x_normal_filtered = np.array([lfilter(b,a,x_normal[ind,:]) for ind in range(x_normal.shape[0])])
x_seizure_filtered = np.array([lfilter(b,a,x_seizure[ind,:]) for ind in range(x_seizure.shape[0])])
# print(x_normal.shape)
# print(x_seizure.shape)

#1
mean_np = np.zeros((500,1))
i = 0
for column in x:
    mean_np[i][0]= np.mean(column)
    i = i + 1
# print("mean: ", mean_np.shape)

# print(mean_np)

#2
i = 0
var_np = np.zeros((500,1))
for column in x:
    var_np[i][0] = np.var(column)
    i = i + 1

# print("var: ", var_np.shape)

#3
i = 0
median_np = np.zeros((500,1))
for column in x:
    median_np[i][0] = np.median(column)
    i = i + 1


#4
i = 0
rms_np = np.zeros((500,1))
for column in x:
    rms_np[i][0] = np.sqrt(np.mean(column**2))
    i = i + 1
# print("rms: ", rms_np.shape)


#5
i = 0
max_np = np.zeros((500,1))
for column in x:
    max_np[i][0] = np.max(column)
    i = i + 1


#6
i = 0
crest_factor_margin_np = np.zeros((500,1))
for column in x:
    crest_factor_margin_np[i][0] = np.max(column)/ np.sqrt(np.mean(column**2))
    i = i + 1


#7
i = 0
min_np = np.zeros((500,1))
for column in x:
    min_np[i][0] = np.min(column)
    i = i + 1


#8
i = 0
margin_factor_np = np.zeros((500,1))
for column in x:
    margin_factor_np[i][0] = np.max(column) / np.var(column)
    i = i + 1


#9
i = 0
sum_of_squares_np = np.zeros((500,1))
for column in x:
    sum_of_squares_np[i][0] = np.sum(column**2)
    i = i + 1


#10
i = 0
skewness_np = np.zeros((500,1))
for column in x:
    skewness_np[i][0] = skew(column)
    i = i + 1


#11
i = 0
ptp_np = np.zeros((500,1))
for column in x:
    ptp_np[i][0] = np.max(column) - np.min(column)
    i = i + 1


#12
i = 0
sum_np = np.zeros((500,1))
for column in x:
    sum_np[i][0] = np.sum(column)
    i = i + 1


#13
i = 0
std_np = np.zeros((500,1))
for column in x:
    std_np[i][0] = np.std(column)
    i = i + 1


#14
i = 0
a_factor_np = np.zeros((500,1))
for column in x:
    a_factor_np[i][0] = np.max(column)/(np.std(column) * np.var(column))
    i = i + 1


#15
i = 0
kortosis_np = np.zeros((500,1))
for column in x:
    kortosis_np[i][0] = kurtosis(column)
    i = i + 1


#### normalize data ###

def normalize (x):
    normalizedData = (x-np.min(x))/(np.max(x)-np.min(x))
    return normalizedData


kortosis_np = normalize(kortosis_np)
a_factor_np = normalize(a_factor_np)
std_np = normalize(std_np)
sum_np = normalize(sum_np)
ptp_np = normalize(ptp_np)
skewness_np = normalize(skewness_np)
sum_of_squares_np = normalize(sum_of_squares_np)
margin_factor_np = normalize(margin_factor_np)
min_np = normalize(min_np)
crest_factor_margin_np = normalize(crest_factor_margin_np)
max_np = normalize(max_np)
rms_np = normalize(rms_np)
median_np = normalize(median_np)
var_np = normalize(var_np)
mean_np = normalize(mean_np)


features_np = np.zeros((500,15))
features_np = np.concatenate((mean_np,var_np,median_np,rms_np,max_np,crest_factor_margin_np,min_np,margin_factor_np,sum_of_squares_np,skewness_np,ptp_np,sum_np,std_np,a_factor_np,kortosis_np),1)
# features_np = np.concatenate((std_np,ptp_np,rms_np),1)

# print("features:" ,features_np.shape)

x_normal = x_normal_filtered
x_seizure = x_seizure_filtered

x = np.concatenate((x_normal,x_seizure))
y = np.concatenate((np.zeros((400,1)),np.ones((100,1))))


print(x.shape)
print(y.shape)

x = features_np
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=seed,test_size=0.2)

print(x_test.shape)

#SVC

####### Linear #######
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("SVC with Linear kernel accuracy: " , accuracy_score(y_test,y_pred))
print("SVC precision: ", precision_score(y_test,y_pred))
print("SVC recall: ", recall_score(y_test,y_pred))

####### Poly #######
clf = SVC(kernel='poly')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("SVC with poly kernel accuracy: " , accuracy_score(y_test,y_pred))
print("SVC precision: ", precision_score(y_test,y_pred))
print("SVC recall: ", recall_score(y_test,y_pred))

####### sigmoid #######
clf = SVC(kernel='sigmoid')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("SVC with sigmoid kernel accuracy: " , accuracy_score(y_test,y_pred))
print("SVC precision: ", precision_score(y_test,y_pred))
print("SVC recall: ", recall_score(y_test,y_pred))

####### RBF #######

clf = SVC(kernel='rbf')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("SVC with RBF kernel accuracy: " , accuracy_score(y_test,y_pred))
print("SVC precision: ", precision_score(y_test,y_pred))
print("SVC recall: ", recall_score(y_test,y_pred))

#Random forest

rf = RandomForestClassifier(max_depth = 5, random_state = 0)
rf.fit(x_train, np.ravel(y_train))

y_pred = rf.predict(x_test)

print("random forest with max depth = 5 and random state = 0: ")
print("random forest accuracy: ", accuracy_score(y_test, y_pred))
print("random forest precision: ", precision_score(y_test, y_pred))
print("random forestrandom forest recall: ", recall_score(y_test, y_pred))
print("random forest cross validation: ", cross_val_score(rf, x, np.ravel(y), cv = 5), "\n")


#KNN

knn = neighbors.KNeighborsClassifier(n_neighbors=10 , weights='uniform')
knn.fit(x_train, np.ravel(y_train))
y_pred = knn.predict(x_test)
print("KNN with 10 neighbors: ")
print("KNN accuracy: ", accuracy_score(y_test, y_pred))
print("KNN precision: ", precision_score(y_test, y_pred))
print("KNN recall: ", recall_score(y_test, y_pred))
print("KNN cross validation: ", cross_val_score(knn, x, np.ravel(y), cv = 5), "\n")

knn = neighbors.KNeighborsClassifier(n_neighbors=50 , weights='uniform')
knn.fit(x_train, np.ravel(y_train))
y_pred = knn.predict(x_test)
print("KNN with 50 neighbors: ")
print("KNN accuracy: ", accuracy_score(y_test, y_pred))
print("KNN precision: ", precision_score(y_test, y_pred))
print("KNN recall: ", recall_score(y_test, y_pred))
print("KNN cross validation: ", cross_val_score(knn, x, np.ravel(y), cv = 5), "\n")

print("matris gomrahi:" , confusion_matrix(y_test, y_pred), "\n")

RocCurveDisplay.from_predictions(y_test, y_pred, color= "darkorange")
plt.show()

print("------------------------------------ End --------------------------------")
