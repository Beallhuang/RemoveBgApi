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
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps
from io import BytesIO
from torchvision import transforms
from models.birefnet import BiRefNet
from fastapi.responses import RedirectResponse

app = FastAPI()

# Load weights from Hugging Face BiRefNet Models
device = 'cuda'
birefnet = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet')
torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to(device)
birefnet.eval()

# load paddleocr model, 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True)
headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,'
              'application/signed-exchange;v=b3;q=0.9',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7',
    'cache-control': 'max-age=0',
    'cookie': 'cna=/VW9GxZeZz0CATr7AYSy/iUI; isg=BAwM2xuBg8NHF5e_GeAAAmh23Wo-RbDvmFP7vWbNGLda8az7jlWAfwJAkflJuehH',
    'if-modified-since': 'Mon, 24 Oct 2022 02:25:20 GMT',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'cross-site',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) '
                  'Version/13.0.3 Mobile/15E148 Safari/604.1',
}


@app.get("/resize")
def resize_img_keep_ratio(url, target_size=[800, 800]):
    file_content = requests.get(url, headers=headers).content
    image = Image.open(BytesIO(file_content))
    old_size = image.size  # 原始图像大小
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = image.resize((new_size[0], new_size[1]))  # 根据上边的大小进行放缩
    pad_w = target_size[0] - new_size[0]  # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[1] - new_size[1]  # 计算需要填充的像素数目（图像的高这一维度上）
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    pad_other = 50
    img_new = ImageOps.expand(img, border=(left + pad_other, top + pad_other, right + 100, bottom + 100),
                              fill=(255, 255, 255))
    img_new.save(rf'c:/Users/beall/Desktop/{url.split("/")[-1]}')


@app.exception_handler(RequestValidationError)
def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"参数不对{request.method} {request.url}")
    return JSONResponse({"code": "400", "message": exc.errors()})


@app.get("/remove")
def remove_bg(url: str = None):
    start_time = time.time()
    input_path = requests.get(url, headers=headers).content

    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(BytesIO(input_path), )
    input_images = transform_image(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)

    img2 = Image.new('RGB', size=(image.width, image.height), color=(255, 255, 255))
    img2.paste(image, (0, 0), mask=image)

    # img转bytes
    img2_buffer = BytesIO()
    img2.save(img2_buffer, format='JPEG')
    data = img2_buffer.getvalue()
    data = BytesIO(data)

    print(f"remove_bg cost {round(time.time() - start_time, 2)} s")
    return StreamingResponse(content=data, media_type="image/jpeg",
                             headers={"Additional-Info": f"{round(time.time() - start_time, 2)} s"})


@app.get("/ocr")
def get_text(url: str = None):
    start_time = time.time()

    input_path = requests.get(url, headers=headers).content
    result = ocr.ocr(input_path, cls=True)
    print(result)
    result = [m[-1][0] for i in result if i for m in i]
    print(f"ocr cost {round(time.time() - start_time, 2)} s")
    return JSONResponse({'code': 200, 'content': result, 'total_time': f"{round(time.time() - start_time, 2)} s"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app='main:app', host="127.0.0.1", port=8001)
