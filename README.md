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