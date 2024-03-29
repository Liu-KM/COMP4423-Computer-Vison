import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from skimage import feature, exposure
import time
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score , ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Read the image data
def collect_images_info(root_dir):
    data = []
    for main_class in os.listdir(root_dir):
        main_class_path = os.path.join(root_dir, main_class)
        if os.path.isdir(main_class_path):
            for img_name in os.listdir(main_class_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(main_class_path, img_name)
                    label = f"{main_class}"
                    data.append((img_name, img_path, label))

    return data

# Convert the image to array
def image_to_array(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize((72,72))
    img_array = np.array(img)
    return img_array


root_dir = './dataset'  
images_info = collect_images_info(root_dir)
df = pd.DataFrame(images_info, columns=['ImageName', 'ImagePath', 'Label'])


tqdm.pandas(desc="Progress")
# Convert the image to array
df['ImageArray'] = df['ImagePath'].progress_apply(image_to_array)

# Convert the label to encoded label
label_encoder = LabelEncoder()
df['EncodedLabel'] = label_encoder.fit_transform(df['Label'])
# get the label mapping
label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
print(label_mapping)
# Remove the ImagePath and ImageName columns
df.drop('ImagePath', axis=1, inplace=True)
df.drop('ImageName', axis=1, inplace=True)

print("[Emoji data loaded]")
print(df["Label"].value_counts())

# ================Uncomment to load the fer2013 dataset===========================

# #https://www.kaggle.com/code/ahmedmahmoud16/facial-expression-recognition-with-logistic
# df = pd.read_csv('./fer_data/train.csv')
# print(df.info())
# tqdm.pandas(desc="Progress")
# df['ImageArray'] = df['pixels'].progress_apply(lambda x: np.uint8(np.fromstring(x, dtype=int, sep=' ').reshape(48,48)))
# # rename column emotion to EncodedLabel
# df.rename(columns={'emotion':'EncodedLabel'}, inplace=True)

# # change grayscale to RGB
# df['ImageArray'] = df['ImageArray'].apply(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB))

# # Retain only 20 per cent of the data in each category owing to the volume of data
# df = df.groupby('EncodedLabel').apply(lambda x: x.sample(frac=0.2)).reset_index(drop=True)




# Extract Edge, Color Histogram, LBP, SIFT, HOG features
def preprocess_image(image_array):
    gray_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)
    return edges

def extract_color_histogram(image_array, bins=32):
    histogram = [cv2.calcHist([image_array], [i], None, [bins], [0, 256]) for i in range(3)]
    histogram = np.concatenate(histogram)
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

def extract_lbp_features(image_array, P=8, R=1):
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    lbp = feature.local_binary_pattern(gray_image, P, R, 'uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_sift_features(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return []
    return descriptors

def extract_hog_features(image_array):
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    # Calculate HOG features
    hog_features= feature.hog(gray_image, visualize=False)
    # # Enhance the contrast of the HOG image for better visualization
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_features




# Add the features to the dataframe

df['HOGFeatures'] = df['ImageArray'].progress_apply(lambda x: extract_hog_features(x))
print("[Generate HOG Features]")
df['ColorHistogram'] = df['ImageArray'].progress_apply(lambda x: extract_color_histogram(x))
print("[Generate color histogram]")
df['LBPFeatures'] = df['ImageArray'].progress_apply(lambda x: extract_lbp_features(x))
print("[Generate lbp features]")
df['SIFTFeatures'] = df['ImageArray'].progress_apply(lambda x: extract_sift_features(x))
print("[Generate sift features]")
df['ImageEdge'] = df['ImageArray'].progress_apply(lambda x: preprocess_image(x))
print("[Edge detected using Canny]")

print("The data frame after extract features:")
print(df.info())


# Model training and evaluation
classifiers = {}
classifiers["SVC"] = SVC(kernel='linear')
classifiers["DecisionTree"] = DecisionTreeClassifier()
classifiers["RandomForest"] = RandomForestClassifier()
classifiers["KNN"] = KNeighborsClassifier(n_neighbors=3)
classifiers["XGBoost"] = XGBClassifier() 
classifiers["LogisticRegression"] = LogisticRegression(max_iter=1000)

# Get the accuracy of the models
def get_models_accuracy(classifiers,X_train,X_test,y_train,y_test):
    for cls_name in classifiers.keys():
        classifier = classifiers[cls_name]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{cls_name} Accuracy: {accuracy * 100:.2f}%')

# Feature list
feature_list = ["ImageArray","ColorHistogram","HOGFeatures","LBPFeatures","ImageEdge"]
for img_feature in feature_list:

    print("="*20,img_feature,"="*20)
    X = np.array(df[img_feature].apply(lambda x: np.array(x).flatten()))
    X = np.array([np.array(xi) for xi in X])
    y = np.array(df['EncodedLabel'])

    # Normalization
    X_scaled = StandardScaler().fit_transform(X)
    print("="*10,"scaled result","="*10)
    print("XGBoost model may take a long time to train")

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)
    get_models_accuracy(classifiers,X_train,X_test,y_train,y_test)

    # Dimensionality reduction
    pca = PCA(n_components=10)  
    X_pca = pca.fit_transform(X_scaled)
    print("="*10,"scaled+PCA result","="*10)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    get_models_accuracy(classifiers,X_train,X_test,y_train,y_test)

# feature_list = ["ImageArray","ColorHistogram","HOGFeatures","LBPFeatures","ImageEdge"]
# for img_feature in feature_list:
#     print("="*10,"test the model with unseen data","="*10)
#     X = np.array(df[img_feature].apply(lambda x: np.array(x).flatten()))
#     X = np.array([np.array(xi) for xi in X])
#     y = np.array(df['EncodedLabel'])
#     X_scaled = X/255
#     for classifier in classifiers:
#         start = time.time()
#         classifiers[classifier].fit(X_scaled, y)
#         # unseen_img = [image_to_array('./unseen_data/1.png'),
#         #               image_to_array('./unseen_data/2.png'),
#         #               image_to_array('./unseen_data/3.png'),
#         #               image_to_array('./unseen_data/4.png')]
#         unseen_img = [image_to_array(f'./unseen_data/{i}.png') for i in range(1,9)]
#         print(f"Time taken to train {classifier}: {time.time() - start:.2f} seconds")
#         count = 0
#         for img in unseen_img:
#             count += 1
#             plt.imshow(Image.fromarray(img))
#             img = img.flatten()
#             img = np.array([img])
#             img = img/255
#             y_pred = classifiers[classifier].predict(img)  # Pass img as a list
#             predict_result = label_mapping[y_pred[0]]
#             #plot the unseen image with the predicted label
#             plt.title(f"Predicted label: {predict_result}")
#             plt.savefig(f'./unseen_data/{count}_{classifier}.png')
#             print(f"{predict_result}")
#             plt.clf()
