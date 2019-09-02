import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
import cv2
import onnxruntime.backend as backend


# 图片预处理，转为numpy作为模型输入
def get_numpy_from_img(file): 
    nh = 224
    nw = 224

    # img = Image.open(file)
    # x = np.array(img, dtype='float32')
    # x = x.reshape(net.blobs['data'].data.shape)

    img = cv2.imread(file)
    
    # 裁剪中心部分
    h, w, _ = img.shape
    if h < w:
        off = int((w - h) / 2)
        img = img[:, off:off + h]
    else:
        off = int((h - w) / 2)
        img = img[off:off + h, :]
    img = cv2.resize(img, (nh, nw))

    
    # img = cv2.resize(img, (nh,nw))
    # cv2默认为 BGR顺序，而其他软件一般使用RGB，所以需要转换
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2默认为bgr顺序
    x = np.array(img, dtype=np.float32)

    # print(x)
    # x = np.reshape(x, (1,5,5,3))
    # 矩阵转置换，img读取后的格式为W*H*C 转为model输入格式 C*W*H
    x = np.transpose(x,(2,0,1))
    
    (a,b,c) = x.shape
    x = x.reshape(1, a, b, c)

    # mean操作
    x[0,0,:,:] -= 123.68
    x[0,1,:,:] -= 116.779
    x[0,2,:,:] -= 103.939

    # scale 操作  1/58.8 = 0.017
    x = x * 0.017

    file_name = file.split('.')[0]
    np.save(file_name, x)
        
    return x


if __name__ == "__main__":

    get_numpy_from_img("cat.jpg")
