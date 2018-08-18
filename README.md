# YOLO 人脸检测

基于YOLOv2的人脸检测之Keras 实现。

## 依赖项

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

WIDER FACE 数据集，32,203个图像并标记393,703张人脸；随机选择40％/ 10％/ 50％的数据作为训练，验证和测试集。

![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/wider_face_intro.jpg)

请按照[说明](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) 下载 WIDER_train.zip, WIDER_val.zip, WIDER_test.zip 以及 wider_face_split.zip 放入 data 目录。

```bash
$ wget http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip
```


## 用法

### 数据预处理
提取12880张训练图片，3226张验证图片和16097张测试图片：
```bash
$ python pre-process.py
```

### 训练
```bash
$ python train.py
```

如果想在培训期间进行可视化，请在终端中运行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
下载 [pre-trained model](https://github.com/foamliu/YOLO-Face-Detection/releases/download/v1.0/model.11-0.6262.hdf5) 放在 models 目录然后执行:

```bash
$ python demo.py
```

|1|2|3|4|
|---|---|---|---|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/0_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/5_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/10_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/15_out.jpg)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/1_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/6_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/11_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/16_out.jpg)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/2_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/7_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/12_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/17_out.jpg)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/3_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/8_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/13_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/18_out.jpg)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/4_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/9_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/14_out.jpg)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/19_out.jpg)|

### 数据增强

```bash
$ python augmentor.py
```
|before|after|
|---|---|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_before_0.png)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_after_0.png)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_before_1.png)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_after_1.png)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_before_2.png)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_after_2.png)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_before_3.png)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_after_3.png)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_before_4.png)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_after_4.png)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_before_5.png)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_after_5.png)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_before_6.png)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_after_6.png)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_before_7.png)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_after_7.png)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_before_8.png)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_after_8.png)|
|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_before_9.png)|![image](https://github.com/foamliu/YOLO-Face-Detection/raw/master/images/imgaug_after_9.png)|