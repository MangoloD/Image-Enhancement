"""
Project ：Image-Enhancement 
File    ：run.py
Author  ：MangoloD
Date    ：2022/3/9 13:58 
"""
import os
import cv2
import json
import retinex

data_path = 'image'
img_list = os.listdir(data_path)
if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

for img_name in img_list:
    if img_name == '.gitkeep':
        continue

    img = cv2.imread(os.path.join(data_path, img_name))
    cv2.imshow('Image', img)

    print('msrcr processing......')
    img_msrcr = retinex.MSRCR(
        img,
        config['sigma_list'],
        config['G'],
        config['b'],
        config['alpha'],
        config['beta'],
        config['low_clip'],
        config['high_clip']
    )
    cv2.imshow('MSRCR retinex', img_msrcr)
    cv2.imwrite("MSRCR_retinex.tif", img_msrcr);

    print('amsrcr processing......')
    img_amsrcr = retinex.automatedMSRCR(
        img,
        config['sigma_list']
    )
    cv2.imshow('autoMSRCR retinex', img_amsrcr)
    cv2.imwrite('AutomatedMSRCR_retinex.tif', img_amsrcr)

    print('msrcp processing......')
    img_msrcp = retinex.MSRCP(
        img,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']
    )

    shape = img.shape

    cv2.imshow('MSRCP', img_msrcp)
    cv2.imwrite('MSRCP.tif', img_msrcp)
    cv2.waitKey()
