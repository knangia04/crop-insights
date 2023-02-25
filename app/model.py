from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import seaborn as sns
import pandas as pd
import pickle 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('Crop_recommendation.csv')
df.head()

c=df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target']=c.cat.codes

# print(df)

y=df.target
X=df[['N','P','K','temperature','humidity','ph','rainfall']]

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# print(X_test.head())

# grad = GradientBoostingClassifier().fit(X_train, y_train)


# # save model
# pickle.dump(grad, open(filename, "wb"))

# # load model
# grad = pickle.load(open(filename, "rb"))


filename = "my_model.pickle"
# load model
grad = pickle.load(open(filename, "rb"))
print('Gradient Boosting accuracy : {}'.format(grad.score(X_test,y_test)))
entry = pd.DataFrame([[50,50,50,30,80,6.5,200]],columns=['N','P','K','temperature','humidity','ph','rainfall'])
crop = grad.predict(entry)

# convert the prediction to the correct label after it was encoded 
print(targets[crop[0]])
 