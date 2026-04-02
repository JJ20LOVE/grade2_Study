import cv2
import numpy as np

img_path = "/data/workspace/myshixun/原图/image.png"
########Begin########

# 1. 图像以灰度图输入 (IMREAD_GRAYSCALE)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 2. 图像缩放到 256×256
img_resized = cv2.resize(img, (256, 256))

# 3. 图像的直方图 (使用 cv2.calcHist)
hist = cv2.calcHist([img_resized], [0], None, [256], [0, 256])

# 4. 灰度图均衡化 (使用 cv2.equalizeHist)
equ = cv2.equalizeHist(img_resized)

# 5. 水平拼接原图与均衡化结果
result = np.hstack((img_resized, equ))

########End########
# 保存结果
cv2.imwrite("/data/workspace/myshixun/学员文件/equ.jpg", equ)
# 打印结果用于测试校验（拼接后整幅图的像素和）
print(np.sum(result))
