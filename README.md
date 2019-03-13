
***软院同学参考[这里](https://github.com/NJUOCR/universal_cnn/wiki/学院内网环境下的使用说明)下载***

---
# 依赖
| 依赖 | 安装示例 |
| --- | --- |
| cv2 | `pip install opencv-python` |
| tensorflow | `pip install tensorflow==1.11.0` |
| yaml | `pip install pyyaml` |
| flask | `pip install Flask` |
| progressbar | `pip install progressbar2` |

# TODO
- 添加文件上传功能

# 使用
OCR服务以web的方式对外提供接口。 推荐使用我们发布的docker镜像，~~点击下载(Not provide yet)~~


## 在docker中启动服务
**可以在app.py的末尾更改服务端口**

运行
``` shell
cd /usr/local/src/universal_cnn
python app.py
```

> 预训练模型已在docker镜像中配置好

## 使用OCR服务
服务接口为标准GET请求：
`http://[host]:[port]/?path=[your_image_path]`

`your_image_path`是需要做识别的图片路径，**务必确保它已经位于容器中，或已通过其他方式挂载进容器**

### 示例
使用`wget`调用服务
``` shell
wget -O out.txt http://[your_host]:[port]/?path=test_data/test0.png&remove_lines=1
```
识别结果会写入`out.txt`

> 我们在镜像中准备了一些测试用的图片，位置：`[project_root]/test_data`
# Trouble Shooting
## 1. 缺少动态链接库
### 1.1 Cannot open ** libSM.so.6 ** when import cv2:

``` shell
apt install -y libsm6 libxext6
```

*by [StackOverflow](https://stackoverflow.com/search?q=import+cv2+libXrender.so.1+)*

### 1.2 Cannot open ** libXrender.so.1 ** when import cv2:

``` shell
sudo apt-get install libxrender1
```
*by [StackOverflow](https://stackoverflow.com/questions/47113029/importerror-libsm-so-6-cannot-open-shared-object-file-no-such-file-or-directo)*
