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
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


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


def image_to_array(img_path):
    img = Image.open(img_path)
    #因为图片有透明度4个通道
    img = img.convert('RGB')
    img = img.resize((72,72))
    img_array = np.array(img)
    return img_array

# 收集图片信息
root_dir = './dataset'  # 更改为你的图片根目录
images_info = collect_images_info(root_dir)

# 创建DataFrame
df = pd.DataFrame(images_info, columns=['ImageName', 'ImagePath', 'Label'])

tqdm.pandas(desc="Progress")

# 假设 image_to_array 是一个自定义函数，用于将图像路径转换为数组
df['ImageArray'] = df['ImagePath'].progress_apply(image_to_array)


# 移除不再需要的ImagePath列
df.drop('ImagePath', axis=1, inplace=True)
df.drop('ImageName', axis=1, inplace=True)

# 对标签进行编码
label_encoder = LabelEncoder()
df['EncodedLabel'] = label_encoder.fit_transform(df['Label'])
print("[Emoji data loaded]")

def preprocess_image(image_array):
    # 颜色空间转换 - RGB到灰度
    gray_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # 边缘检测 - 使用Canny边缘检测器
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




df['ColorHistogram'] = df['ImageArray'].apply(lambda x: extract_color_histogram(x))
print("[Generate color histogram]")
df['LBPFeatures'] = df['ImageArray'].apply(lambda x: extract_lbp_features(x))
print("[Generate lbp features]")
df['SIFTFeatures'] = df['ImageArray'].apply(lambda x: extract_sift_features(x))
print("[Generate sift features]")
df['ImageEdge'] = df['ImageArray'].apply(lambda x: preprocess_image(x))
print("[Edge detected using Canny]")

print("The data frame after extract features:")
print(df.info())




import matplotlib.pyplot as plt

def plot_color_histogram(color_histogram):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        plt.plot(color_histogram[i * 32:(i + 1) * 32], color=col)
        plt.xlim([0, 32])
    plt.title('Color Histogram')
    plt.savefig("color-histogram.png")
    print("color histogram example is saved in ./color-histogram.png")

# 假设你已经提取了某个图像的颜色直方图并存储在ColorHistogram列中
# 选择第一个图像的颜色直方图进行可视化
plot_color_histogram(df['ColorHistogram'].iloc[0])

def plot_lbp_features(lbp_features):
    plt.plot(lbp_features)
    plt.title('LBP Features Histogram')
    plt.savefig("lbp-features.png")
    print("color histogram example is saved in ./lbp-features.png")

plot_lbp_features(df['LBPFeatures'].iloc[0])

def plot_edge_detection(image_edges):
    # print(image_edges.shape)
    plt.figure(figsize=(6, 6))
    plt.imshow(image_edges, cmap='gray')
    plt.title('Edge Detection Result')
    plt.axis('off') # 不显示坐标轴
    plt.savefig("edge-detection.png")
    print("color histogram example is saved in ./edge-detection.png")

plot_edge_detection(df['ImageEdge'].iloc[0])

def show_lbp_image(image_array):
    # 首先将图像转换为灰度图，因为LBP通常在灰度图上计算
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    # 计算LBP特征
    lbp = feature.local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    # 将LBP特征图标准化到0-255范围内以便可视化
    lbp_image = (lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255
    lbp_image = lbp_image.astype("uint8")

    # 显示LBP图像
    plt.figure(figsize=(6, 6))
    plt.imshow(lbp_image, cmap='gray')
    plt.title('LBP Result')
    plt.axis('off')  # 不显示坐标轴
    plt.savefig("lbp-img.png")
    print("color histogram example is saved in ./lbp-img.png")

# 使用示例
# 假设your_image_array是你想要处理的图像数组
show_lbp_image(df['ImageArray'].iloc[0])

def visualize_sift_features(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    sift_image = cv2.drawKeypoints(gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(sift_image, cmap='gray')
    plt.title('SIFT Features')
    plt.axis('off')
    plt.savefig("sift_features.png")
    print("sift features example is saved in ./sift-features.png")

visualize_sift_features(df['ImageArray'].iloc[0])




classifiers = {}
classifiers["SVC"] = SVC(kernel='linear')
classifiers["DecisionTree"] = DecisionTreeClassifier()
classifiers["RandomForest"] = RandomForestClassifier()
classifiers["KNN"] = KNeighborsClassifier(n_neighbors=3)
classifiers["XGBoost"] = XGBClassifier()
classifiers["LogisticRegression"] = LogisticRegression(max_iter=1000)

def get_models_accuracy(classifiers,X_train,X_test,y_train,y_test):
    for cls_name in classifiers.keys():
        classifier = classifiers[cls_name]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{cls_name} Accuracy: {accuracy * 100:.2f}%')

feature_list = ["ImageArray","ColorHistogram","LBPFeatures","ImageEdge"]
for img_feature in feature_list:
    print("="*20,img_feature,"="*20)
    X = np.array(df[img_feature].apply(lambda x: np.array(x).flatten()))
    X = np.array([np.array(xi) for xi in X])
    y = np.array(df['EncodedLabel'])

    # 归一化处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("="*10,"scaled result","="*10)
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)
    get_models_accuracy(classifiers,X_train,X_test,y_train,y_test)


    # 降维处理
    if img_feature not in ["LBPFeatures"]:
        pca = PCA(n_components=32)  # 假设我们想要降到2维
        X_pca = pca.fit_transform(X_scaled)
        print("="*10,"scaled+PCA result","="*10)
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
        get_models_accuracy(classifiers,X_train,X_test,y_train,y_test)


