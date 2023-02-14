import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png', 0)
k = np.ones((15, 15)) / 225


def conv(A, k):
    kh, kw = k.shape
    h, w = A.shape
    B = np.ones((h, w))
    for i in range(0, h - kh + 1):
        for j in range(0, w - kw + 1):
            sA = A[i:i + kh, j:j + kw]
            B[i, j] = np.sum(k * sA)
    B = B[0:h - kh + 1, 0:w - kw + 1]
    return B


B = conv(img, k)
imgB = np.array(B, dtype='uint8')
plt.imshow(img, cmap='gray')
plt.show()
plt.imshow(imgB, cmap='gray')
plt.show()
# %%
img = cv2.imread('lenna.png')
negative = 256 - 1 - img
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.show()

# %%
plt.rcParams.update({'font.size': 7})
i = cv2.imread('img.png')
i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
img = np.array(i, dtype='float')
maxV = 255 / np.log(1 + np.max(i))
vals = np.linspace(0, maxV, 8, dtype=int)

rows = 3
cols = 3

fig = plt.figure(figsize=(3, 3), dpi=600)

item = fig.add_subplot(rows, cols, 1)
item.set_title('Anh goc')
plt.axis('off')
plt.imshow(i)

for i, c in enumerate(vals):
    log_image = c * (np.log(img + 1))
    log_image = np.array(log_image, dtype='uint8')
    item = fig.add_subplot(rows, cols, i + 2)
    item.set_title('c={}'.format(c))
    plt.axis('off')
    plt.imshow(log_image)
fig.tight_layout()
plt.show()

# %%
