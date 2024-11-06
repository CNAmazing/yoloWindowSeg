from ultralytics import YOLOv10
if __name__ == '__main__':
    # model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
    # or
    # wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
    model = YOLOv10(r'E:\hhhh\yolov10\runs\train\exp\weights\best.pt')

    # metrics=model.val(data=r'data.yaml', batch=256)
    metrics=model.val()
   


