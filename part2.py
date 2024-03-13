import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage import feature
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('./fer_data/train.csv')
print(df.info())

#https://www.kaggle.com/code/ahmedmahmoud16/facial-expression-recognition-with-logistic

tqdm.pandas(desc="Progress")
# image_array=[]
# for i, row in enumerate(df.index):
#         image = np.fromstring(df.loc[row, ' pixels'], dtype=int, sep=' ')
#         image_array.append(image)
df['ImageArray'] = df['pixels'].progress_apply(lambda x: np.uint8(np.fromstring(x, dtype=int, sep=' ').reshape(48,48)))



# Remove the ImagePath and ImageName columns
df.drop('pixels', axis=1, inplace=True)
# df.drop(' Usage', axis=1, inplace=True)

print(df["emotion"].value_counts())
print("[Emoji data loaded]")

# Extract Edge, Color Histogram, LBP, SIFT, HOG features
def preprocess_image(image_array):
    edges = cv2.Canny(image_array, 100, 200)
    return edges

def extract_lbp_features(image_array, P=8, R=1):
    lbp = feature.local_binary_pattern(image_array, P, R, 'uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_sift_features(image_array):
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image_array, None)
    if descriptors is None:
        return []
    return descriptors

def extract_hog_features(image_array):
    hog_features= feature.hog(image_array, visualize=False)
    # # Enhance the contrast of the HOG image for better visualization
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_features




# Add the features to the dataframe
df['HOGFeatures'] = df['ImageArray'].progress_apply(lambda x: extract_hog_features(x))
print("[Generate HOG Features]")
df['LBPFeatures'] = df['ImageArray'].progress_apply(lambda x: extract_lbp_features(x))
print("[Generate lbp features]")
df['SIFTFeatures'] = df['ImageArray'].progress_apply(lambda x: extract_sift_features(x))
print("[Generate sift features]")
df['ImageEdge'] = df['ImageArray'].progress_apply(lambda x: preprocess_image(x))
print("[Edge detected using Canny]")

print("The data frame after extract features:")
print(df.info())


X = np.array(df["HOGFeatures"].apply(lambda x: np.array(x).flatten()))
X = np.array([np.array(xi) for xi in X])
y = np.array(df['emotion'])

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'{LogisticRegression} Accuracy: {accuracy * 100:.2f}%')

# # Model training and evaluation
# classifiers = {}
# # classifiers["SVC"] = SVC(kernel='linear')
# classifiers["DecisionTree"] = DecisionTreeClassifier()
# classifiers["RandomForest"] = RandomForestClassifier()
# classifiers["KNN"] = KNeighborsClassifier(n_neighbors=3)
# # classifiers["XGBoost"] = XGBClassifier()
# classifiers["LogisticRegression"] = LogisticRegression(max_iter=1000)

# # Get the accuracy of the models
# def get_models_accuracy(classifiers,X_train,X_test,y_train,y_test):
#     for cls_name in classifiers.keys():
#         classifier = classifiers[cls_name]
#         classifier.fit(X_train, y_train)
#         y_pred = classifier.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         print(f'{cls_name} Accuracy: {accuracy * 100:.2f}%')

# # Feature list
# feature_list = ["ImageArray","HOGFeatures","LBPFeatures","ImageEdge"]
# for img_feature in feature_list:
#     print("="*20,img_feature,"="*20)
#     X = np.array(df[img_feature].apply(lambda x: np.array(x).flatten()))
#     X = np.array([np.array(xi) for xi in X])
#     y = np.array(df['emotion'])

#     # Normalization
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     print("="*10,"scaled result","="*10)

#     # # Data split
#     # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)
#     # get_models_accuracy(classifiers,X_train,X_test,y_train,y_test)

#     # Dimensionality reduction
    
#     pca = PCA(n_components=32)
#     if img_feature in ["LBPFeatures"]:
#         pca = PCA(n_components=10)
#     X_pca = pca.fit_transform(X_scaled)
#     print("="*10,"scaled+PCA result","="*10)
#     X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
#     get_models_accuracy(classifiers,X_train,X_test,y_train,y_test)

