# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLOv10
if __name__ == '__main__':
    # model.load('yolov8n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLOv10(model=r'ultralytics/cfg/models/v10/yolov10n.yaml')
    model.train(data=r'data.yaml',
                imgsz=512,
                epochs=200,
                batch=4,
                workers=4,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
