# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
# %%
import seaborn as sea
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, neighbors, neural_network
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# %%
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# %%
train_data.head()
test_data.head()


# %%
train_data.columns[train_data.isna().any()].tolist()


# %%

data = [test_data, train_data]

for train_data in data:
    train_data.drop(columns=['PassengerId', 'Ticket', 'Cabin'], inplace=True)

    train_data['Alone'] = 1
    train_data['Alone'].loc[train_data['Parch'] + train_data['SibSp'] > 1] = 0

    train_nan_map = {'Age': train_data['Age'].mean(), 'Fare': train_data['Fare'].mean(
    ), 'Embarked': train_data['Embarked'].mode()[0]}

    train_data.fillna(value=train_nan_map, inplace=True)

    # columns_map = {'Embarked': {'C': 0, 'Q': 1, 'S': 2},
    #                'Sex': {'male': 0, 'female': 1}}
    # train_data.replace(columns_map, inplace=True)

    train_data['Title'] = train_data['Name'].str.split(
        ", ", expand=True)[1].str.split(".", expand=True)[0]
    stat_min = 10
    title_names = (train_data['Title'].value_counts() < stat_min)

    train_data['Title'] = train_data['Title'].apply(
        lambda x: 'Misc' if title_names.loc[x] == True else x)

    encoder = LabelEncoder()
    encode = ['Embarked', 'Sex', 'Title']
    for col in encode:
        train_data[col] = encoder.fit_transform(train_data[col])

X_train = train_data.loc[:, train_data.columns != 'Survived']
X_train = train_data.loc[:, train_data.columns != 'Name']

y_train = train_data.loc[:, 'Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.33, random_state=10)

# %%
print(X_train.head())
print(y_train.head())
print(test_data.head())

# %%

pp = sea.pairplot(train_data, hue='Survived', palette='deep', size=1.2,
                  diag_kind='kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))
pp.set(xticklabels=[])

# %%
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(X_train.values, y_train.values)
print(tree_clf.score(X_test.values, y_test.values))


# %%
knn_clf = neighbors.KNeighborsClassifier()
knn_clf.fit(X_train.values, y_train.values)
print(knn_clf.score(X_test.values, y_test.values))


# %%
NN_clf = neural_network.MLPClassifier()
NN_clf.fit(X_train.values, y_train.values)
print(NN_clf.score(X_test.values, y_test.values))


# %%
svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(X_train.values, y_train.values)
print(svm_clf.score(X_test.values, y_test.values))
y_pred = svm_clf.predict(X_test.values)
y_truth = y_test.values


# %%
tn, fp, fn, tp = confusion_matrix(y_truth, y_pred).ravel()
print("Confusion Matrix")
print(confusion_matrix(y_truth, y_pred, labels=[0, 1]))
print("")
print("True Negatives", tn)
print("False Positives", fp)
print("False Negatives", fn)
print("True Positives", tp)


# %%


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_truth, y_pred)
class_names = ['0', '1']
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()


# %%
predictions = svm_clf.predict(test_data.values)


# %%
type(predictions)


# %%
pred_df = pd.DataFrame(
    predictions, index=test_data.index, columns=['Survived'])
type(pred_df)


# %%
pred_df.to_csv('predictions.csv', header=True, sep=',')


# %%
