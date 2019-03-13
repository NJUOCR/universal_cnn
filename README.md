
# Requirement
| dependency | instalation |
| --- | --- |
| cv2 | `pip install opencv-python` |
| tensorflow | `pip install tensorflow` |
| yaml | `pip install pyyaml` |
| flask | `pip install Flask` |
| progressbar | `pip install progressbar2` |

# 使用
OCR服务以web的方式对外提供接口。 推荐使用我们发布的docker镜像，~~点击下载(Not provide yet)~~

## 在docker中启动服务
**默认的web端口为 555**

运行
``` shell
cd /usr/local/src/universal_cnn
python app.py
```

> 预训练模型已在docker镜像中配置好

## 使用OCR服务
服务接口为标准GET请求：
`http://[host]:555/?path=[your_image_path]`

`your_image_path`是需要做识别的图片路径，**务必确保它已经位于容器中，或已通过其他方式挂载进容器**

> **不包含文件上传的功能**，我们不对“使用何种方式上传文件?”，“文件上传到哪里？”，“识别后是否删除文件？”等相关问题提供统一的解决方案
> 这些问题由使用者来解决

### 示例
使用`wget`调用服务
``` shell
wget http://[your_host]:555/?path=/usr/local/src/universal_cnn/test_data/input_data/img-0008.jpg&remove_lines=1 -O out.json
```
识别结果会写入`out.json`

> 我们在镜像中准备了一些测试用的图片，位置：`/usr/local/src/universal_cnn/test_data/input_data`
# Trouble Shooting
## 1. 缺少动态链接库
1.1 Cannot open ** libSM.so.6 ** when import cv2:

``` shell
apt install -y libsm6 libxext6
```

*by [StackOverflow](https://stackoverflow.com/search?q=import+cv2+libXrender.so.1+)*

1.2 Cannot open ** libXrender.so.1 ** when import cv2:

``` shell
sudo apt-get install libxrender1
```
*by [StackOverflow](https://stackoverflow.com/questions/47113029/importerror-libsm-so-6-cannot-open-shared-object-file-no-such-file-or-directo)*
