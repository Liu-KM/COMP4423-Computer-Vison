import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage import filters

# 函数定义
def plot_color_histogram(image, ax, title="Color Histogram"):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
    ax.set_xlim([0, 256])
    ax.set_title(title)
    ax.set_xlabel('Bin')
    ax.set_ylabel('Frequency')

def plot_hog_features(image, ax, title="HOG Features"):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True)
    ax.imshow(hog_image, cmap='gray')
    ax.set_title(title)

def plot_lbp_features(image, ax, title="LBP Features"):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image_gray, P=8, R=1, method="uniform")
    ax.imshow(lbp, cmap='gray')
    ax.set_title(title)

def plot_sift_features(image, ax, title="SIFT Features"):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
    ax.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    ax.set_title(title)

def plot_image_edges(image, ax, title="Image Edges"):
    edges = cv2.Canny(image, 100, 200)
    ax.imshow(edges, cmap='gray')
    ax.set_title(title)

# 加载图片
# ./dataset\hand\emoji_u1faf7_1f3ff.png
# ./dataset\hand\emoji_u1faf8.png
# ./dataset\hand\emoji_u1faf8_1f3fb.png
    
images = [cv2.imread(f'./images/train/angry/0.jpg'),cv2.imread(f'./images/train/disgust/299.jpg'),cv2.imread(f'./images/train/fear/2.jpg')]
#images = [cv2.imread(f'./dataset/hand/emoji_u1faf7_1f3ff.png'),cv2.imread(f'./dataset\worker\emoji_u1f477_1f3fc.png'),cv2.imread(f'./dataset\wizard\emoji_u1f9d9.png')]


# 创建对比图
fig, axs = plt.subplots(6, 3, figsize=(15, 20))

for i, image in enumerate(images):
    axs[0, i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, i].set_title(f'Original Image {i+1}')
    plot_color_histogram(image, axs[1, i], "Color Histogram")
    plot_hog_features(image, axs[2, i], "HOG Features")
    plot_lbp_features(image, axs[3, i], "LBP Features")
    plot_sift_features(image, axs[4, i], "SIFT Features")
    plot_image_edges(image, axs[5, i], "Image Edges")

for ax in axs.flat:
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

# 仅对颜色直方图显示坐标轴
for ax in axs[1]:
    ax.axis('on')

plt.tight_layout()
plt.savefig("facial-image-features.png")
plt.show()

