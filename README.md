# ANN-2021-group-jittor-CRNN
2021秋季学期人工神经网络大作业

1.数据集准备
    在根目录下创建data文件夹，所有数据集的根目录即为data/

2.训练
    cd src-jittor && python3 train.py

3.评测
    cd src-jittor && python3 evaluate.py

    在config.py 配置checkpoints的文件路径

4.预测
    cd src-jittor

    mkdir demo && 将要预测的图片放进demo中

    python3 predict demo/*.jpg

5.线上demo
    采用flask框架
    
    cd src-jittor && python3 FlaskServer.py
