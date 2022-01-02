#encoding=utf-8
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import timedelta
import os
import sys

"""Usage: predict.py [-m MODEL] [-s BS] [-d DECODE] [-b BEAM] [IMAGE ...]

-h, --help    show this
-m MODEL     model file [default: checkpoints/test68000.pkl]
-s BS       batch size [default: 256]
-d DECODE    decode method (greedy, beam_search or prefix_beam_search) [default: beam_search]
-b BEAM   beam size [default: 10]

"""
from docopt import docopt
import jittor as jt
from jittor import nn
from tqdm import tqdm
from config import common_config as config
from dataset import Synth90kDataset
from model import CRNN
from ctc_decoder import ctc_decode

jt.flags.use_cuda = jt.has_cuda

def predict(crnn, dataloader, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataloader), desc="Predict")

    all_preds = []
    with jt.no_grad():
        for data in dataloader:
            #device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            #images = data.to(device)
            images = data
           
            logits = crnn(images)
            log_probs = nn.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            all_preds += preds

            pbar.update(1)
        pbar.close()

    return all_preds


def show_result(paths, preds):
    print('\n===== result =====')
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
        print(f'{path} > {text}')


reload_checkpoint = "/root/group/crnn-jittor/checkpoints/test96000.pkl"
batch_size = 1
decode_method = "beam_search"
beam_size = 10

img_height = config['img_height']
img_width = config['img_width']

num_class = len(Synth90kDataset.LABEL2CHAR) + 1
crnn = CRNN(1, img_height, img_width, num_class,
            map_to_seq_hidden=config['map_to_seq_hidden'],
            rnn_hidden=config['rnn_hidden'],
            leaky_relu=config['leaky_relu'])
crnn.load_state_dict(jt.load(reload_checkpoint))



app = Flask(__name__)

# 输出
@app.route('/')
def hello_world():
    print("here")
    return 'Hello World!'

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

# 添加路由
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        # 通过file标签获取文件
        f = request.files['file']
        # print("fuck",sys.stdout)
        # return "fuck"
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "图片类型：png、PNG、jpg、JPG、bmp"})
        # 当前文件所在路径
        basepath = os.path.dirname(__file__)
        # 一定要先创建该文件夹，不然会提示没有该路径
        upload_path = "/root/group/crnn-jittor/src-jittor"+'/images/'+ secure_filename(f.filename)
        # 保存文件
        f.save(upload_path)
        
        # images = "/root/group/crnn-jittor/src-jittor/images/*.jpg"
        images = [upload_path]
        predict_loader = Synth90kDataset(paths=images,
                                    img_height=img_height, img_width=img_width,
                                    batch_size=batch_size,
                                    shuffle=False)

        preds = predict(crnn, predict_loader, Synth90kDataset.LABEL2CHAR,
                    decode_method=decode_method,
                    beam_size=beam_size)
        print('\n===== result =====')
        for path, pred in zip(images, preds):
            text = ''.join(pred)
            print(f'{path} > {text}')
        # show_result(images, preds)
        # 返回上传成功界面
        return text
    # 重新返回上传界面
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")
