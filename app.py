from flask import Flask, request, send_file, render_template_string
import cv2, numpy as np, io, base64
from color_model import detect_masks, overlay_and_contours, stats
app = Flask(__name__)
HTML = '''<!doctype html><meta charset=utf-8><title>荧光三分类平台</title>
<style>body{font-family:system-ui;background:#111;color:#eee;text-align:center}img{max-width:30%;margin:5px;border:1px solid #444}.box{background:#222;padding:20px;margin:20px auto;max-width:1100px;border-radius:12px}</style>
<div class=box><h2>坏死骨 / 坏组织 / 健康组织 识别平台</h2>
<p>亮红=坏组织 | 暗红=健康组织 | 黑=坏死骨(可叠加在亮红下,白色描边)</p>
<form method=post action=/predict enctype=multipart/form-data><input type=file name=file accept="image/*"><button>识别划分</button></form>
{% if res %}<h3>面积占比: {{res}}</h3><div>{% for im in imgs %}<img src="data:image/png;base64,{{im}}">{% endfor %}</div>
<p>左:原图 中:半透明叠加 右:边界勾勒</p>{% endif %}</div>'''
def enc(img):
    _,b=cv2.imencode('.png',img); return base64.b64encode(b).decode()
@app.route('/',methods=['GET'])
def idx(): return render_template_string(HTML)
@app.route('/predict',methods=['POST'])
def pred():
    if 'file' not in request.files:
        return '请先选择文件', 400
    f=request.files['file'].read()
    if not f:
        return '空文件，请重新上传', 400
    img=cv2.imdecode(np.frombuffer(f,np.uint8),1)
    if img is None:
        return '图像解码失败，请上传 JPG/PNG', 400
    masks,extra=detect_masks(img)
    ov,vis=overlay_and_contours(img,masks)
    s=stats(masks,extra)
    return render_template_string(HTML,res=s,imgs=[enc(img),enc(ov),enc(vis)])
if __name__=='__main__': app.run(port=5000,debug=True)
