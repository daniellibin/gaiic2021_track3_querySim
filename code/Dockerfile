# Base Images
## 从天池基础镜像构建(from的base img 根据自己的需要更换，建议使用天池open list镜像链接：https://tianchi.aliyun.com/forum/postDetail?postId=67720)
#FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.6-cuda10.1-py3
FROM registry.cn-shanghai.aliyuncs.com/xiaobu_match/match:cuda10.2base

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

##安装依赖包,pip包请在requirements.txt添加
#RUN apt-get update && apt-get install -y curl


#RUN   pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
#pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
#RUN    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==4.2.0
#RUN    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm
#RUN    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flask
#RUN    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas
#RUN    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple psutil
#RUN    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx
#RUN    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime-gpu==1.7.0
#RUN    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sklearn
#RUN    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime_tools
#RUN    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sympy
#RUN    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentencepiece


## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /


## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
