# organize imports
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

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

# load the user configs
with open('conf/conf.json') as f:    
  config = json.load(f)

# config variables
model_name      = config["model"]
weights         = 'imagenet'
test_size       = config["test_size"]
seed            = config["seed"]
num_classes     = config["num_classes"]
include_top     = config["include_top"]
train_path      = config["train_path"]
test_path       = config["test_path"]
features_path   = "output/"+model_name+"/features.h5"
labels_path     = "output/"+model_name+"/labels.h5"
results         = "output/"+model_name+"/results.txt"
classifier_path = "output/"+model_name+"/classifier.pickle"
model_path      = "output/"+model_name+"/model_" + str(test_size)
plot_path       = "output/"+model_name+"/confusion-matrix-plot.png"
plot_path_norm  = "output/"+model_name+"/confusion-matrix-plot-norm.png"

# import features and labels
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] features shape: {}".format(features.shape))
print ("[INFO] labels shape: {}".format(labels.shape))

print ("[INFO] training started...")
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed)

print ("[INFO] splitted train and test data...")
print ("[INFO] train data  : {}".format(trainData.shape))
print ("[INFO] test data   : {}".format(testData.shape))
print ("[INFO] train labels: {}".format(trainLabels.shape))
print ("[INFO] test labels : {}".format(testLabels.shape))

# use logistic regression as the model
print ("[INFO] creating model...")
model = LogisticRegression(random_state=seed)
model.fit(trainData, trainLabels)

# use rank-1 and rank-5 predictions
print ("[INFO] evaluating model...")
f = open(results, "w")
rank_1 = 0
rank_5 = 0

# loop over test data
for (label, features) in zip(testLabels, testData):
  # predict the probability of each class label and
  # take the top-5 class labels
  predictions = model.predict_proba(np.atleast_2d(features))[0]
  predictions = np.argsort(predictions)[::-1][:5]

  # rank-1 prediction increment
  if label == predictions[0]:
    rank_1 += 1

  # rank-5 prediction increment
  if label in predictions:
    rank_5 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100

# write the accuracies to file
f.write("Rank-1: {:.2f}%\n".format(rank_1))
f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of test data
preds = model.predict(testData)

# get the list of training lables
labelz = sorted(list(os.listdir(train_path)))

# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds, target_names=labelz)))
f.close()

# dump classifier to file
print ("[INFO] saving model...")
pickle.dump(model, open(classifier_path, 'wb'))

# display the confusion matrix
print ("[INFO] confusion matrix")
'''
testLabels2 = []
preds2 = []

for n, i in enumerate(testLabels):
  testLabels2.append(labelz[i])
  preds2.append(     labelz[i])

# plot the confusion matrix
cm = confusion_matrix(testLabels2, preds2)
sns.heatmap(cm,
            annot=True,
            cmap="Set2")

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=labelz,
                      title='Confusion matrix, without normalization')

#plt.show()
plt.savefig(plot_path)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=labelz, normalize=True,
                      title='Normalized confusion matrix')
#plt.show()
plt.savefig(plot_path_norm)
'''