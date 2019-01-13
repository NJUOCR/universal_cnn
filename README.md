
# 使用
OCR服务以web的方式对外提供接口。 推荐使用我们发布的docker镜像，~~点击下载~~

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
wget http://[your_host]:555/?path=/usr/local/src/universal_cnn/test_data/input_data/img-0008.jpg -O out.json
```
识别结果会写入`out.json`

> 我们在镜像中准备了一些测试用的图片，位置：`/usr/local/src/universal_cnn/test_data/input_data`
