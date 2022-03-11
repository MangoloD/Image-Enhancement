"""
Project ：Image-Enhancement 
File    ：img_show.py
Author  ：MangoloD
Date    ：2022/3/10 15:12 
"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image/BGRT2.jpg')
img_ = cv2.imread('result/MSRCP.jpg')
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Offical')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('MSRCP')

plt.show()
