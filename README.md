# HUST-NLP

## 声明

    这里是华中科技大学计算机科学与技术学院本科2022级自然语言处理实验
    本人基于cuda11.0 pytorch1.7.1 vscode进行开发

## 基础环境准备

安装 Anaconda

   官方文档：<a href="https://docs.continuum.io/anaconda/install/">anaconda install</a>

搭建虚拟环境并安装 PyTorch

    # 创建虚拟环境
    conda create -n nlpcu python=3.7	# 创建名为 nlpcu 的虚拟环境

    conda activate nlpcu   # 激活虚拟环境
    conda deactivate       # 退出虚拟环境

    # 在nlpcu虚拟环境中安装 Pytorch 1.7.1 CPU 版本 （在HUST校园网环境下不需要换源）
    # CUDA 9.2
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=9.2 -c pytorch
    # CUDA 10.1
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
    # CUDA 10.2
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
    # CUDA 11.0
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
    # CPU Only
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch
    ```

根据各位的cuda版本进行选择

    pip install -r requirements.txt

## 运行方式



3. 训练

   ```shell
   # save 目录下存放了一个粗略训练过的模型，可先跳过训练过程直接进行推断
   
   # 数据准备，data 目录下运行
   python 0.split.py
   python 1.data_u_ner.py
   # 模型训练，项目根目录下运行
   # 若安装并配置了 GPU 相关运行环境可添加命令行参数 --cuda 来使用 GPU 训练
   python run.py
   ```

4. 推断

   ```shell
   python infer.py
   ```

   

## transformer环境准备

   ```shell
   conda create -n transformer python=3.10
   conda activate transformer
   pip install -r requirements.txt
   ```

   通过transformer的官方文档进行安装, 具体可参考：<a href=" 具体可参考：<a href="URL_ADDRESSuggingface.co/docs/transformers/index">transformer</a>
   