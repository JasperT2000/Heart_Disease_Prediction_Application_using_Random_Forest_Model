import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
# Loading dataset as Data Frame object
df = pd.read_csv("/Users/jasper/Downloads/heart.csv")
# Handling outliers
df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = 247
df.loc[df['RestingBP'] == 0, 'RestingBP'] = 132
# Feature Engineering
encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
Sex = {index: label for index, label in enumerate(encoder.classes_)}
df['ChestPainType'] = encoder.fit_transform(df['ChestPainType'])
ChestPainType = {index: label for index, label in enumerate(encoder.classes_)}
df['RestingECG'] = encoder.fit_transform(df['RestingECG'])
RestingECG = {index: label for index, label in enumerate(encoder.classes_)}
df['ExerciseAngina'] = encoder.fit_transform(df['ExerciseAngina'])
ExerciseAngina = {index: label for index, label in enumerate(encoder.classes_)}
df['ST_Slope'] = encoder.fit_transform(df['ST_Slope'])
ST_Slope = {index: label for index, label in enumerate(encoder.classes_)}
# Dividing the target variable from the dataset
x = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
# Scaling the values present in the dataset
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(x)
# Splitting into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=275, random_state=0)
# Training the Random Forest Classifier Model
model = RandomForestClassifier(n_estimators=1000)  # , max_depth=5, random_state=1
model.fit(x_train, y_train)
Y_pred = model.predict(x_test)
score = model.score(x_train, y_train)
#print('Training Score:', score)
score = model.score(x_test, y_test)
#print('Testing Score:', score)
output = pd.DataFrame({'Predicted': Y_pred})  # Heart-Disease yes or no? 1/0
#print(output.head())
people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0:
    rate_people = len(people) / len(output)
#print("% of people predicted with heart-disease:", rate_people)
score_rfc = score
out_rfc = output
#print(classification_report(y_test, Y_pred))
CM = confusion_matrix(y_test, Y_pred)
#print('Confusion Matrix is : \n', CM)
# drawing confusion matrix
#sns.heatmap(CM, center=True, annot=True)
#plt.show()
"""
Getting input from the user and predicting

age = int(input("Enter your age : "))
gender = input("Enter the value corresponding to your gender : 0-Female 1-Male")
cpt = input("Enter the value corresponding to Chest pain type : 0-ASY 1-ATA 2-NAP 3-TA")
rbp = int(input("Enter the resting bp value : "))
chol = int(input("Enter the cholesterol value : "))
fbs = input("Enter the value corresponding to FastingBS condition : 0 or 1")
ecg = input("Enter the value corresponding to Resting ECG results : 0-LVH 1-Normal 2-ST")
mhr = int(input("Enter the value of MaxHR : "))
exer = input("Enter the value corresponding to Exercise Angina presence : 0-N 1-Y")
oldpeak = input("Enter the value of oldpeak : ")
st = input("Enter the value corresponding to the type of ST_Slope : 0-Down 1-Flat 2-Up")
#input_data = (age, gender, cpt, rbp, chol, fbs, ecg, mhr, exer, oldpeak, st)
input_data = [age, gender, cpt, rbp, chol, fbs, ecg, mhr, exer, oldpeak, st]
#input_data_as_num = np.asarray(input_data)
input_data_as_num = [np.array(input_data)]
#reshaped = input_data_as_num.reshape(1, -1)
#prediction = model.predict(reshaped)
prediction = model.predict(input_data_as_num)
#probability = model.predict_proba(reshaped)
probability = model.predict_proba(input_data_as_num)
"""
pickle.dump(model,open('model.pkl','wb'))