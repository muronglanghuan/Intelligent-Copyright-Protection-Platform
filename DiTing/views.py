import os.path
import json
import pandas as pd
# import pymysql
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views import View
# from debt_risk import models
from datetime import datetime
# from graphviz import Source
import re
import numpy as np

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from PIL import Image
import io
import requests

# import requests
import base64
from collections import OrderedDict
from PIL import Image
from sentence_transformers import SentenceTransformer

# 线性映射调整
def adjust_similarity(x):
    if x < 0.5:
        return 0.2 * x
    else:
        return 0.1 + 1.8 * (x - 0.5)

class IndexView(View):
    def get(self, request):
        next_ = request.GET.get("direction")
        data = {"direction": next_}
        return render(request, "index.html", context=data)

class LoginView(View):
    def get(self, request):
        next_ = request.GET.get("direction")
        data = {"direction": next_}
        return render(request, "login.html", context=data)

class InformationView(View):
    def get(self, request):
        next_ = request.GET.get("direction")
        data = {"direction": next_}
        return render(request, "information.html", context=data)


from django.views import View
from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
import base64
import io
import requests

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 准备就绪
except ImportError:
    model = None  # 如果没装 sentence-transformers，可以做个容错

# class SubmitView(View):
#     def get(self, request):
#         next_ = request.GET.get("direction")
#         data = {"direction": next_}
#         return render(request, "submit.html", context=data)
#
#     def post(self, request):
#         files = request.FILES.getlist('file')
#         results = []
#
#         for file in files:
#             image = Image.open(file)
#             buffered = io.BytesIO()
#             image.save(buffered, format="PNG")
#             base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
#
#             data = {
#                 "image": base64_image,
#                 "model": "wd14-convnext",
#                 "threshold": 0.35
#             }
#
#             response = requests.post('http://127.0.0.1:7860/tagger/v1/interrogate', json=data)
#             if response.status_code == 200:
#                 json_data = response.json()
#                 # 处理返回的JSON数据
#                 caption_dict = json_data['caption']
#                 sorted_items = sorted(caption_dict.items(), key=lambda x: x[1], reverse=True)
#                 # 转换成百分比并格式化输出字符串
#                 formatted_captions = ['{}: {:.2%}'.format(key, value) for key, value in sorted_items[:10]]
#                 results.append(formatted_captions)
#             else:
#                 return JsonResponse({'error': response.text}, status=response.status_code)
#
#         return JsonResponse({'results': results})
class SubmitView(View):
    def get(self, request):
        next_ = request.GET.get("direction")
        data = {"direction": next_}
        return render(request, "submit.html", context=data)

    def post(self, request):
        """
        处理最多 10 张图片：
        - 如果只有 1 张，则只返回 Top10 标签，不做相似度计算。
        - 如果 >= 2 张，则对每张图片求加权向量，生成 N×N 相似度矩阵。
        """
        files = request.FILES.getlist('file')
        if not files:
            return JsonResponse({'error': '未检测到上传的图片文件'}, status=400)

        # 最多只处理前 10 张
        files = files[:10]
        n = len(files)

        # 存储每张图片的处理结果
        images_info = []

        # ===============================
        # 1. 循环处理每张图片，得到 top10 标签 & 概率 & 向量
        # ===============================
        for idx, file in enumerate(files):
            image = Image.open(file)
            # 转成 base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            data = {
                "image": base64_image,
                "model": "wd14-convnext",
                "threshold": 0.35
            }
            response = requests.post('http://127.0.0.1:7860/tagger/v1/interrogate', json=data)
            if response.status_code != 200:
                return JsonResponse({'error': response.text}, status=response.status_code)

            json_data = response.json()
            caption_dict = json_data['caption']
            # 拿到前 10 个标签
            sorted_items = sorted(caption_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            if not sorted_items:
                # 如果阈值较高，可能没有标签，通过一个空替代
                sorted_items = []

            # 分离标签和概率
            tags, probs = zip(*sorted_items) if sorted_items else ([], [])
            tags = list(tags)
            probs = np.array(probs, dtype=np.float32)

            # ========== 将标签转成向量并加权 ==========
            if model is not None and len(tags) > 0:
                vecs = model.encode(tags)  # shape: (k, emb_dim)
                # 加权
                weighted_vecs = (vecs.T * probs).T  # (k, emb_dim)
                sum_probs = np.sum(probs)
                if sum_probs > 0:
                    image_vec = np.sum(weighted_vecs, axis=0) / sum_probs
                else:
                    image_vec = np.zeros(vecs.shape[1])
            else:
                # model 为 None 或者没有标签时
                image_vec = np.zeros(384)  # 假设 'all-MiniLM-L6-v2' 默认384维

            # 保存信息 (包括此图片的Top10标签、加权后向量)
            images_info.append({
                'file_name': file.name,
                'top10': [{'tag': t, 'prob': float(p)} for t, p in sorted_items],
                'vector': image_vec.tolist()
            })
        # print(images_info)
        # ===============================
        # 2. 若只有 1 张图，则不计算相似度
        # ===============================
        if n == 1:
            return JsonResponse({
                'message': '只上传了 1 张图片，不进行相似度计算。',
                'image_info': images_info[0],  # 只返回这唯一一张的 top10 标签等
            })

        # ===============================
        # 3. 若 >= 2 张图，计算相似度矩阵 (n x n)
        # ===============================
        #   - 这里用余弦相似度
        # ===============================
        vectors = [info['vector'] for info in images_info]

        def cosine_similarity(a, b):
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            if a_norm == 0 or b_norm == 0:
                return 0.0
            return float(np.dot(a, b) / (a_norm * b_norm))

        # 构建 n×n 矩阵
        sim_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    # 自身与自身相似度设为 1.0
                    sim_val = 1.0
                else:
                    sim_val = cosine_similarity(vectors[i], vectors[j])
                # 映射到你想要的分布
                sim_val_adjusted = adjust_similarity(sim_val)
                row.append(sim_val_adjusted)
            sim_matrix.append(row)

        # 组装返回
        return JsonResponse({
            'num_images': n,
            'images_info': [
                {
                    'file_name': info['file_name'],
                    'top10': info['top10'],
                }
                for info in images_info
            ],
            'similarity_matrix': sim_matrix
        })