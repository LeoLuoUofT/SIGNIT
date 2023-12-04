# %% [markdown]
# ## CREATING TEST AND TRAIN DATA

# %%
# The original MNIST image dataset of handwritten digits is a popular benchmark for image-based machine learning methods but researchers have renewed efforts to update it and develop drop-in replacements that are more challenging for computer vision and original for real-world applications. As noted in one recent replacement called the Fashion-MNIST dataset, the Zalando researchers quoted the startling claim that "Most pairs of MNIST digits (784 total pixels per sample) can be distinguished pretty well by just one pixel". To stimulate the community to develop more drop-in replacements, the Sign Language MNIST is presented here and follows the same CSV format with labels and pixel values in single rows. The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion).

# %%
# About this Dataset
# The original MNIST image dataset of handwritten digits is a popular benchmark for image-based machine learning methods but researchers have renewed efforts to update it and develop drop-in replacements that are more challenging for computer vision and original for real-world applications. As noted in one recent replacement called the Fashion-MNIST dataset, the Zalando researchers quoted the startling claim that "Most pairs of MNIST digits (784 total pixels per sample) can be distinguished pretty well by just one pixel". To stimulate the community to develop more drop-in replacements, the Sign Language MNIST is presented here and follows the same CSV format with labels and pixel values in single rows. The American Sign Language letter database of hand gestures represent a multi-class problem with 24 classes of letters (excluding J and Z which require motion).

# The dataset format is patterned to match closely with the classic MNIST. Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions). The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST but otherwise similar with a header row of label, pixel1,pixel2….pixel784 which represent a single 28x28 pixel image with grayscale values between 0-255. The original hand gesture image data represented multiple users repeating the gesture against different backgrounds. The Sign Language MNIST data came from greatly extending the small number (1704) of the color images included as not cropped around the hand region of interest. To create new data, an image pipeline was used based on ImageMagick and included cropping to hands-only, gray-scaling, resizing, and then creating at least 50+ variations to enlarge the quantity. The modification and expansion strategy was filters ('Mitchell', 'Robidoux', 'Catrom', 'Spline', 'Hermite'), along with 5% random pixelation, +/- 15% brightness/contrast, and finally 3 degrees rotation. Because of the tiny size of the images, these modifications effectively alter the resolution and class separation in interesting, controllable ways.

# This dataset was inspired by the Fashion-MNIST 2 and the machine learning pipeline for gestures by Sreehari 4.

# A robust visual recognition algorithm could provide not only new benchmarks that challenge modern machine learning methods such as Convolutional Neural Nets but also could pragmatically help the deaf and hard-of-hearing better communicate using computer vision applications. The National Institute on Deafness and other Communications Disorders (NIDCD) indicates that the 200-year-old American Sign Language is a complete, complex language (of which letter gestures are only part) but is the primary language for many deaf North Americans. ASL is the leading minority language in the U.S. after the "big four": Spanish, Italian, German, and French. One could implement computer vision in an inexpensive board computer like Raspberry Pi with OpenCV, and some Text-to-Speech to enabling improved and automated translation applications.

# Input (105.8 MB)
# #

# %%
import pandas as pd
import numpy as np
train_data_df = pd.read_csv(r"/content/training.csv", header = None)
test_data_df = pd.read_csv(r"/content/testing.csv", header = None)

# %%
cols_in_data = ['label'] + [f'pixel{i}' for i in range(1,10001)]
train_data_df.columns = cols_in_data
test_data_df.columns = cols_in_data

# %%
train_data_df

# %%
test_data_df

# %%
# train_data_df[train_data_df.label == 9]

# %%
# train_data_df[train_data_df.label == 25]

# %%
# train_data_df = train_data_df[train_data_df['label'] != 9]
# train_data_df = train_data_df[train_data_df['label'] != 25]

# %%
train_data_df

# %%
# train_data_df['label'] = train_data_df['label'].apply(lambda x: x - 1 if x > 9 else x)

# %%
train_data_df

# %%
# test_data_df['label'] = test_data_df['label'].apply(lambda x: x - 1 if x > 9 else x)

# %%
test_data_df

# %%
x_train_df = train_data_df.iloc[: , 1:]


# %%
x_train_df

# %%
y_train_df = train_data_df.iloc[: , :1]

# %%
y_train_df

# %%
max(y_train_df['label'])

# %%
min(y_train_df['label'])

# %%
x_test_df = test_data_df.iloc[: , 1:]


# %%
x_test_df

# %%
y_test_df = test_data_df.iloc[: , :1]

# %%
y_test_df

# %%
print(x_train_df.shape)
print(y_train_df.shape)
print(x_test_df.shape)
print(y_test_df.shape)

# %%
# Normalize the data (greyscale normalization for faster convergence)
x_train_df = x_train_df/255
x_test_df = x_test_df/255



# %%
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.appName("create spark df").getOrCreate()

# %%

# spark_x_train_df = spark.createDataFrame(x_train_df)

# %%
# spark_x_train_df

# %%
x_train_df

# %%
x_test_df

# %%
y_train_df

# %%
y_test_df

# %%
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train_df, y_train_df, test_size = 0.3, random_state = 42)

# %%
#dataframe to numpy
x_train = x_train_df.values
y_train = y_train_df.values
x_val = x_train_df.values
y_val = y_train_df.values
x_test = x_test_df.values
y_test = y_test_df.values

# %%
y_train

# %%
# Reshaping the data from 1-D to 3-D as required through input by CNN's
x_train = x_train.reshape(-1,100,100,1)
x_val = x_val.reshape(-1,100,100,1)
x_test = x_test.reshape(-1,100,100,1)

# %%
x_train.shape

# %%
x_train.shape[0]

# %%
x_train[0].shape

# %%
x_train[0]

# %%
y_train.shape

# %%
y_test.shape

# %%
# class_names = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# %%
class_names = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Del', 'N', 'Space']

# %%
class_names[28]

# %%
len(class_names)

# %%
# class_names_datawise = ['A','B','C','D','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# %%
# len(class_names_datawise)

# %%
y_train

# %%
class_names[int(y_train[1])]

# %%
int(y_train[1])

# %%
import matplotlib.pyplot as plt

for i in range(15):
    plt.imshow(x_train[i], cmap='gray')  #can add subplotting here later
    plt.title(f"Image Number: {i} | Class: {str(y_train[i])} | Label: {class_names[int(y_train[i])]}") #add 2 in image number to see that row in excel sheet data
    plt.show()

# %%
class_names[20]

# %%
len(class_names)

# %% [markdown]
# ## DATA PREPARATION

# %%
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train_lb = label_binarizer.fit_transform(y_train)
y_val_lb = label_binarizer.fit_transform(y_val)
y_test_lb = label_binarizer.fit_transform(y_test)

# %%
y_train.shape

# %%
# from tensorflow.keras.utils import to_categorical
# y_train_ohe = to_categorical(y_train)
# y_test_ohe = to_categorical(y_test)

# %%
np.max(y_train)

# %%
np.min(y_train)

# %%
y_train_lb

# %%
print(x_train.shape)
print(y_train_lb.shape)
print(x_val.shape)
print(y_val_lb.shape)
print(x_test.shape)
print(y_test_lb.shape)

# %% [markdown]
# ## can even add data augmentation here

# %% [markdown]
# ## MODEL BUILDING

# %%
# import tensorflow as tf
# import keras
# from tensorflow.keras import layers, models
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# %%
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
# model.add(MaxPooling2D(pool_size = (2, 2)))

# model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))

# model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2)))

# model.add(Flatten())
# model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.20))

# model.add(Dense(24, activation = 'softmax'))

# %%
# model.summary()

# %%
# model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
#               metrics=['accuracy'])

# %%
# batch_size = 128
# epochs = 50

# %%
# history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)

# %%
# # plt.plot(history.history['acc'])
# # plt.plot(history.history['val_acc'])
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title("Accuracy")
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend(['train','test'])

# plt.show()

# %%


# %%
# MODEL BUIDLING AGAIN

# %%
from tensorflow.keras.models import Sequential

# %%
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, Activation, Flatten

# %%
model = Sequential()
model.add(Conv2D(filters =32, kernel_size= (3,3), strides = (1,1), padding= "valid"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters =32, kernel_size= (3,3), strides = (1,1), padding= "valid"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters =32, kernel_size= (2,2), strides = (1,1), padding= "valid"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(29, activation="softmax"))

# %%
model.compile(loss = "categorical_crossentropy", optimizer ="adam", metrics=["accuracy"])

# %%
y_test_lb.shape

# %%
y_test_lb

# %%
x_train
# print(y_train_lb.shape)
# print(x_val.shape)
# print(y_val_lb.shape)
# print(x_test.shape)
# print(y_test_lb.shape)

# %%
print(x_train.shape)
print(y_train_lb.shape)
print(x_val.shape)
print(y_val_lb.shape)

# %%
model.fit(x = x_train,
         y = y_train_lb,
         batch_size = 1000,
         validation_data = (x_val, y_val_lb),
         epochs = 95)

# %%
model.summary()

# %%


# %%
train_loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
train_accuracy = model.history.history['accuracy']
validation_accuracy = model.history.history['val_accuracy']

# %%
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LOSS CURVE")
plt.legend((['Train', 'Validation']))
plt.grid()
plt.show()

# %%
plt.plot(train_accuracy)
plt.plot(validation_accuracy)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ACCURACY CURVE")
plt.legend((['Train', 'Validation']))
plt.grid()
plt.show()

# %%
from sklearn.metrics import classification_report, confusion_matrix

# %%
preds = model.predict(x_test)
preds_class = np.argmax(preds, axis =1)
preds.shape

# %%
y_test.shape

# %%
y_test

# %%
y_test[1][0]

# %%
max(y_test)

# %%
imgs = np.random.randint(0,29, 29)

for n in imgs:
  plt.imshow(x_test[n].reshape(100,100), cmap="gray")
  plt.title(f"Prediction: {class_names[np.argmax(preds[n])]}  |  Actual: {class_names[y_test[n][0]]}")
  plt.show()

# %%
imgs

# %%
preds_class.shape

# %%
y_test

# %%
print(conf_matrix)

# %%
conf_matrix = confusion_matrix(y_true = y_test, y_pred = preds_class)
print(pd.DataFrame(conf_matrix, columns = class_names, index = class_names))

# %%
print(classification_report(y_true = y_test, y_pred = preds_class))

# %%
############################################################################################################

# %%
###################################### FINISHED #####################################################################

# %%
###############################################################################################################

# %%
###################################################################################################

# %%
#####################################################################################################

# %%
#trying sparkML now (RANDOM FOREST MODEL)

# %%
!pip install pyspark

# %%
from pyspark.sql import SparkSession, types
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# %%
spark = SparkSession.builder.appName("SignLanguageClassifier").getOrCreate()

# %%
sign_schema = types.StructType([
    types.StructField('label', types.IntegerType()),
    *[types.StructField(f'pixel{i}', types.IntegerType()) for i in range(1,10001)]
])

# %%
from functools import reduce


data = spark.read.csv(r'/content/training.csv', header = True, schema = sign_schema)
# data = spark.read.csv(r'/content/training.csv', header = False, schema = sign_schema)

# cols_in_data = ['label'] + [f'pixel{i}' for i in range(1,10001)]
# for i, col_name in enumerate(cols_in_data):
#     data = data.withColumnRenamed('_c{}'.format(i), col_name)
# data.columns = cols_in_data

train, validation = data.randomSplit([0.75, 0.25])


# cols_in_data = ['label'] + [f'pixel{i}' for i in range(1, 10001)]
# rename_map = {f'_c{i}': col_name for i, col_name in enumerate(cols_in_data)}
# data = reduce(lambda df, col_rename: df.withColumnRenamed(*col_rename), rename_map.items(), data)
# train, validation = data.randomSplit([0.75, 0.25])

# %%
data.printSchema()

# %%
# data.show(5)

# %%
features_data_cols = [f"pixel{i}" for i in range(1, 10001)]
features_data_cols

# %%
len(features_data_cols)

# %%
features = VectorAssembler(inputCols = features_data_cols, outputCol = 'features')

# %%
sign_classifier = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')

# %%
sign_pipeline = Pipeline(stages =[features, sign_classifier])

# %%
sign_model = sign_pipeline.fit(train)

# %%
predictions = sign_model.transform(validation)

# %%
sign_evaluator = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol ='prediction', metricName = 'accuracy')

# %%
sign_accuracy = sign_evaluator.evaluate(predictions)

# %%
print(f"Accuracy: {sign_accuracy}")

# %%
spark.stop()

# %%
#SPARK ML (MLP model)

# %%
!pip install pyspark

from pyspark.sql import SparkSession, types
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder.appName("SignLanguageClassifier").getOrCreate()

sign_schema = types.StructType([
    types.StructField('label', types.IntegerType()),
    *[types.StructField(f'pixel{i}', types.IntegerType()) for i in range(1,10001)]
])

# %%
data = spark.read.csv(r'/content/training.csv', header = True, schema = sign_schema)

# %%
label_col = data.columns[0]

# %%
feature_cols = data.columns[1:]

# %%
assembler = VectorAssembler(inputCols = feature_cols, outputCol = "features")

# %%
assembled_data = assembler.transform(data)
assembled_data = assembled_data.select("features", label_col)

# %%
train, validation = assembled_data.randomSplit([0.75, 0.25])

# %%
first_layer_size = len(features)
last_layer_size = 29

# %%
layers = [first_layer_size, 64, 64, last_layer_size]

# %%
mlp_model = MultilayerPerceptronClassifier(layers = layers, seed =42, maxIter = 95, labelCol = label_col)

# %%
trained_model = mlp_model.fit(train)

# %%
predictions = trained_model.transform(validation)

# %%
mlp_evaluator = MulticlassClassificationEvaluator(label_col = label_col, predictionCol = "prediction", metricName = "accuracy")

# %%
accuracy = mlp_evaluator(predictions)

# %%
print(accuract)

# %%
spark.stop()

# %%


# %%


# %%


# %%



