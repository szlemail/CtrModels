## 使用说明

### 遗传算法调参

1. 将训练数据 分成特征 和 标签 放在 data/fedata/目录下， 名字分别为train_x trian_y
2. 在optima/目录里 运行 python3 runGA.py -d fe -m lgb 会自动调参lgb
   python3 runGA.py -h  查看更多帮助

### 深度交叉网络 使用
3. models 目录下 Python3 dcn.py 可以对DeepCrossNet 调参。数据可以通过preprocessor/dcn_preprocessor 处理得到。

#### 特征提取示例
4. featureEngine 下有一些特征提取方法，仅做参考。


希望对您有点帮助。
有问题交流，可以在微信公众号 AIStuff123 机器学习干货 给我留言。

git 地址： https://github.com/szlemail/CtrModels
