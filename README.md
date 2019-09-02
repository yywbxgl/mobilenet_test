
训练好的模型获取地址,作者测试的准确率为 89.85%  
https://github.com/shicai/MobileNet-Caffe  


## 文件说明
- model 目录存放下载的训练模型以及转换后的模型
- weights 存放从模型中提取的参数
- signle_op_helper 用于单步operator操作验证
- run_accuracy.py 脚本用于准确率测试，


## 准确率测试 
测试数据：ILSVRC2012 测试集合 01-4999 共5000张图片
```
# v1  caffe run accuracy   01~4999 
total:  4999
top1 hit:  3302
top5 hit:  4340
top1_accuracy_rate:  0.6605321064212842
top5_accuracy_rate:  0.8681736347269454


# v1 caffe-to-onnx  onn run accuracy  01~4999  直接resize
total:  4999
top1 hit:  3302
top5 hit:  4340
top1_accuracy_rate:  0.6605321064212842
top5_accuracy_rate:  0.8681736347269454


# v1 caffe-to-onnx  onn run accuracy  01~4999  图片中心裁剪预处理
test dir:  /home/sunqiliang/share/img_data/img_org_01_4999/
total:  4999
top1 hit:  3414
top5 hit:  4434
top1_accuracy_rate:  0.6829365873174635
top5_accuracy_rate:  0.8869773954790958
```