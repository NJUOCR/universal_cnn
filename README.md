# universal_cnn
---
layout: post
title: "OCR项目说明"
tag: OCR
---

# 目录总览

![](https://ws1.sinaimg.cn/large/e93305edgy1fy48upcjqhj204q09uq2x.jpg)

![](https://ws1.sinaimg.cn/large/e93305edgy1fy58atxwkwj20hp0k7dgf.jpg)



## configs

`infer.yaml:`预测时候的配置文件

`punctuation_letter_digit.yaml:`标点符号，数字，字母的配置文件

`single_char.yaml:`字的训练模型

## models 

相关模型

## processing

`rectification.py：`模型的纠正

`single_char_processing.py:`整个文件数据的处理

- 初始化-加载模型

  调用load函数，需要输入模型的宽w，高h，类书num_class，ckpt_dir：训练好的checkpoint

~~~python
    def __init__(self, charmap_path, aliasmap_path, ckpt_dir, h, w, num_class, batch_size):
        self.main = Main().load(w, h, num_class=num_class, ckpt_dir=ckpt_dir)
        self.data = SingleCharData(h, w).load_char_map(charmap_path).load_alias_map(aliasmap_path)
        self.batch_size = batch_size
~~~



- 图片二值化

~~~
auxiliary_img:辅助图片
~~~

![](https://ws1.sinaimg.cn/large/e93305edgy1fy5bj9az75j20oc0azq4v.jpg)

- 图片旋转矫正

- 水平投影 ——》 垂直投影切分

- 数据进行预测

~~~python
# 数据进行第一次的预测
# 将预测好的数据再放入data中
self.data.set_images(page.make_infer_input_1()).init_indices()
        results = self.main.infer(infer_data=self.data, batch_size=self.batch_size)
~~~

- 对于结果进行一个阈值判断，是否是半个字的，合并为一起

~~~python
page.set_result_1(results)
        page.filter_by_p(p_thresh=p_thresh)
        for line in page.get_lines(ignore_empty=True):
            line.mark_half()
            # line.calculate_meanline_regression()
            line.merge_components()
~~~

- 合并之后的文字进行，第二次预测

~~~python
 self.data.set_images(page.make_infer_input_2()).init_indices()
        results2 = self.main.infer(infer_data=self.data, batch_size=self.batch_size)
        page.set_result_2(results2)
        page.mark_char_location()

        rct.rectify_by_location(page.iterate(1))
        rct.rectify_5(page.iterate(5))
		
        if auxiliary_img is not None:
            # auxiliary_img: 作为路径； page.drawing_copy：要保存的图片
            uimg.save(auxiliary_img, page.drawing_copy)
~~~

- 获得预测结果json文件

~~~python
    # 对于输出的结果，进行一个格式化
    def get_json_result(self, page_path: str, p_thresh: float, auxiliary_img: str):
        page = self._process(page_path, p_thresh, auxiliary_img)
        return page.format_json(p_thresh=p_thresh)
~~~

- 获得预测结果txt文件

~~~python
    def get_text_result(self, page_path: str, p_thresh: float, auxiliary_img: str):
        page = self._process(page_path, p_thresh, auxiliary_img)
        return page.format_result(p_thresh=p_thresh)
~~~

### main调用

~~~python
path = "doc_imgs/2015南立刑初字第0001号_枉法裁判罪84页.pdf/img-0228.jpg"

    proc = Processor("/usr/local/src/data/stage2/all_4190/all_4190.json",
                     "/usr/local/src/data/stage2/all_4190/aliasmap.json",
                     '/usr/local/src/data/stage2/all_4190/ckpts',
                     64, 64, 4190, 64)
    res = proc.get_text_result(path, 0.9, '/usr/local/src/data/results/auxiliary.png')
    print(res)
~~~



## static/lib

前端？

## templates

页面？

## test

**测试文件**

`test_char:`在验证集上进行一个预测

## utils





## app.py



## args.py





## data.py



## main.py



# 使用

## 方法一

### 使用wget

- 命令行方式使用wget，下载服务器上的图片
- **前提要求**：1）图片必须在服务器上, 2）linux系统
- 举例如下：

~~~
wget “http://192.168.68.38:5678/?path=/usr/local/src/data/doc_imgs/2014东刑初字第0100号_诈骗罪208页.pdf/img-0008.jpg” -O test.json
或者
wget "http://nju-vm:5678/?path=/usr/local/src/data/doc_imgs/2014%E4%B8%9C%E5%88%91%E5%88%9D%E5%AD%97%E7%AC%AC0100%E5%8F%B7_%E8%AF%88%E9%AA%97%E7%BD%AA208%E9%A1%B5.pdf/img-0008.jpg" -O test.json
~~~

### 说明

`-O test.json`： 指定输出的文件名称，默认在当前路径下

- test.json，文件如下：

![](https://ws1.sinaimg.cn/large/e93305edgy1fy59851eyfj20j00fwad6.jpg)

- 命令行示例如下：

![](https://ws1.sinaimg.cn/large/e93305edgy1fy58wjcd3mj20kt0o5tf5.jpg)



## 方法二

### 浏览器直接访问

- 在浏览器搜索框中，输入要访问的地址
- **前提要求**：1）图片必须在服务器上, 2）linux系统
- 示例如下（nju-vm=192.168.68.38）

~~~python
http://nju-vm:5678/?path=/usr/local/src/data/doc_imgs/2014%E4%B8%9C%E5%88%91%E5%88%9D%E5%AD%97%E7%AC%AC0100%E5%8F%B7_%E8%AF%88%E9%AA%97%E7%BD%AA208%E9%A1%B5.pdf/img-0008.jpg
或者
http://192.168.68.38:5678/?path=/usr/local/src/data/doc_imgs/2014东刑初字第0100号_诈骗罪208页.pdf/img-0008.jpg
~~~

- 演示结果如下

![](https://ws1.sinaimg.cn/large/e93305edgy1fy59d6nqm3j211u0600ul.jpg)

## 方法三

### 代码调用

- **环境及编译器**

  **pycharm：**pycharm2017.1

  **python：**python3.6.2

  **tensorflow：**

- 

