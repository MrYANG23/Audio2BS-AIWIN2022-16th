数据处理
audioprocess.py 传入输入训练的audio的文件夹，和输出文件文件，得到每个音频对应的多个音频帧的mfcc特征
#audioprocess.py中 audioProcess为对训练和验证数据做处理 audioTestProcess为对测试数据做处理

dataset.py 数据加载和处理部分
model.py 网络模型部分
synthesis.py 对处理后的测试数据进行推理，得到每个音频帧mfcc的blendshape预测系数
submit.py 为对经过synthesis.py文件处理后的，每个音频帧的blendshape合并为整个音频段的blendshape
csv_ensemble.py为对多个模型结果做融合。



由于设备的资源问题，因此本方案只选用了小模型做训练对比，在CPU上跑，平均一个epoch耗时1-2分钟（后续可以使用预训练模型）



