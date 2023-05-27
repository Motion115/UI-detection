import os
import pickle

# use SVM to classify
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

config = {
    'all': "./enrico_corpus/150-embds/enrico_embedding_150.pkl",
    'cv_only': "./enrico_corpus/150-embds/enrico_embedding_cv_150.pkl",
    'nlp_only': "./enrico_corpus/150-embds/enrico_embedding_nlp_150.pkl"
}

mode = 'nlp_only'

# read
with open(os.path.join(config[mode]), 'rb') as f:
    corpus = pickle.load(f)

# create a numpy 2-D array, with each row as a 150-dim vector
X = []
for element in corpus:
    # element['numpy_embedding'] is 2D array, turn it into one D
    element['numpy_embedding'] = element['numpy_embedding'].flatten()
    X.append(element['numpy_embedding'])

# create the y vector for SVM
y = []
for element in corpus:
    y.append(element['label_id'])

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC(probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred) * 100.0))

# save the model
import joblib
joblib.dump(clf, './weights/ML-classification/' + mode + "_svm.pkl")

# load the model
loaded = joblib.load('./weights/ML-classification/' + mode + "_svm.pkl")
# calculalte the top-k accuracy
y_pred = loaded.predict_proba(X_test)
# calculate top-2 accuracy
top2, top3 = 0, 0
for i in range(len(y_pred)):
    top2 += (y_test[i] in y_pred[i].argsort()[-2:])
    top3 += (y_test[i] in y_pred[i].argsort()[-3:])
print("Top-2 Accuracy: %.2f%%" % (top2 / len(y_pred) * 100.0))
print("Top-3 Accuracy: %.2f%%" % (top3 / len(y_pred) * 100.0))



