# -*- coding: utf-8 -*-
"""
@File    : main.py
@Time    : 2024/1/23 1:20
@Author  : beall
@Email   : beallhuang@163.com
@Software: PyCharm
"""
import torch
import time
import requests
from paddleocr import PaddleOCR
from PIL import Image, ImageOps
from io import BytesIO
from models.birefnet import BiRefNet
import os
import json

# dir_path = r"C:\Users\beall\Downloads\ilovepdf_pages-to-jpg"
# for file in os.listdir(dir_path):
#     if file.endswith(('.jpg', '.png', '.jpeg')):
#         os.rename(os.path.join(dir_path, file), os.path.join(dir_path, file.split('_')[1]))

# result_dict = {}
# ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True)
# for file in os.listdir(dir_path):
#     if file.endswith(('.jpg', '.png', '.jpeg')):
#         print(file)
#         result = ocr.ocr(os.path.join(dir_path, file), cls=True)
#         pos_dict = {detection[1][0]: detection[0][0] for line in result for detection in line}
#         result_dict[file.split('.')[0]] = pos_dict
# print(json.dumps(result_dict), file=open(os.path.join(dir_path, 'pos_dict.json'), 'w'))


# import re
# from collections import defaultdict
#
# pos_dict = defaultdict(dict)
# result_dict = json.load(open(os.path.join(dir_path, 'pos_dict.json'), 'r'))
# for k, v in result_dict.items():
#     for key, value in v.items():
#         if re.search(r'\d{4}.\d{2}.\d{2}', key):
#             pos_dict[k]['date'] = tuple(value)
#         if re.search(r'报告|款式分析|BELLE|百丽|直播分析|东风向标|日报|评论|活动分析|监控', key):
#             pos_dict[k]['title'] = tuple(value)
#
# print(pos_dict)


