数据处理
audioprocess.py 传入输入训练的audio的文件夹，和输出文件文件，得到每个音频对应的多个音频帧的mfcc特征
#audioprocess.py中 audioProcess为对训练和验证数据做处理 audioTestProcess为对测试数据做处理

dataset.py 数据加载和处理部分
model.py 网络模型部分
synthesis.py 对处理后的测试数据进行推理，得到每个音频帧mfcc的blendshape预测系数
submit.py 为对经过synthesis.py文件处理后的，每个音频帧的blendshape合并为整个音频段的blendshape
csv_ensemble.py为对多个模型结果做融合。

![image](https://user-images.githubusercontent.com/38352877/189347869-edf45dae-bc10-481c-8917-f70cfd0d8c54.png)
![image](https://user-images.githubusercontent.com/38352877/189347935-2bec5285-4af9-4e95-b287-b162c00deda0.png)

推理速度：以B榜总共74个数据为准，共计10分钟，600秒左右，再做了数据预处理后，总共推理时间为27秒，每秒推理22s的音频，1s音频耗时45ms，最终B榜测试，选用了BiLSTM的第40，48个epoch和LSTM的10，21epoch共4个模型预测的结果进行融合，最终B榜得分0.328

由于设备的资源问题，因此本方案只选用了小模型做训练对比，在CPU上跑，平均一个epoch耗时1-2分钟（后续可以使用预训练模型）

top20
![image](https://user-images.githubusercontent.com/38352877/189348113-14a1dd63-992d-43d5-93dd-70699800c609.png)

top10
![image](https://user-images.githubusercontent.com/38352877/189348137-16a2c840-e78c-4aab-92b8-feba5f6cd9df.png)



