import numpy as np
import matplotlib.pyplot as plt
 
data=[]
labels=[]
with open("mnist_train.csv") as f:
    for i in range(60000):
        line = f.readline()
        #data.append(line)
        transline = line.strip().split(',')
        labels.append(int(transline[0]))
        pixels=np.array(transline[1:],dtype=int)   #每个对应的元素被存储到了标签数组中，对应的像素没存储到了data
        data.append(pixels)


data_matrix=np.array(data)  #变成二维数组

# for i in range(data_matrix.shape[0]):
#     print(f"min={data_matrix[i].min()},{np.count_nonzero(data_matrix[i])}") #大括号读取

normal=data_matrix/255

train_data=normal[:50000]
test_data=normal[50000:]

# print("归一最大值",normal.max())
# print("归一非0",normal[0,normal[0]>0][0] if np.any(normal[0]>0) else"black")

#标记对应的标签

oh=np.zeros((60000,10))#注意内置括号
# print(oh[1][1])
for i in range (60000):
    oh[i][labels[i]]=1

train_labels=oh[:50000]
test_labels=oh[50000:]
# print(oh)

# print(train_data.shape,test_data.shape,train_labels.shape,test_labels.shape)

np.random.seed(42)       #这个模块算是一个帮助理解，是随机的预测
w1=np.random.randn(784,128)*0.01     #w1把128个特征映射到10个数字类别的权重
b1=np.zeros(128)

w2=np.random.randn(128,10)*0.01     #w2代表了对各种特征的一个判断，即从不同的方面来进行判断一个图像到底是什么数字，一开始的时候为随机值，依靠后续的梯度计算来调整权重
b2=np.zeros(10)
# print(w1.shape,b1.shape,w2.shape,b2.shape)

loss_history=[]
acc_history=[]

for epoch in range(100):
    z1=train_data@w1+b1        #z1是梯度，反馈该往什么方向，以什么速度调整
    a1=np.maximum(0,z1)    #记忆：预测的概率为a2,train_labels为对应的one-hot表
    z2=a1@w2+b2
    exp_z2=np.exp(z2)   #求所有的加起来的结果
    sum_per_row=np.sum(exp_z2,axis=1,keepdims=True)  #归结平均值
    a2=exp_z2/sum_per_row

    correct_probs=np.sum(a2*train_labels,axis=1)
    ave_loss=-np.mean(np.log(correct_probs))#交叉熵损失
    #print("平均损失：",ave_loss)
    predict=np.argmax(a2,axis=1)
    truth=np.argmax(train_labels,axis=1)
    acc=np.mean(predict==truth)#准确率，初次运行为0.1，因为就是基本随机的

    loss_history.append(ave_loss)
    acc_history.append(acc)

    print(f"Epoch{epoch+1:2d} ->损失：{ave_loss:.4f},准确率：{acc:.4f}")

    dz2=a2-train_labels  #对应为输出层的梯度dz2
    dw2=(a1.T@dz2)/len(train_data)
    db2=np.mean(dz2,axis=0)
    da1=dz2@w2.T  #继续回推梯度
    dz1=da1*(z1>0)
    dw1=(train_data.T@dz1)/len(train_data)
    db1=np.mean(dz1,axis=0)

    learning_rate=0.15
    w1=w1-learning_rate*dw1
    b1=b1-learning_rate*db1
    w2=w2-learning_rate*dw2
    b2=b2-learning_rate*db2

z1_test = test_data @ w1 + b1
a1_test = np.maximum(0, z1_test)
z2_test = a1_test @ w2 + b2
a2_test = np.exp(z2_test) / np.sum(np.exp(z2_test), axis=1,keepdims=True)
test_acc = np.mean(np.argmax(a2_test, axis=1) == np.argmax(test_labels,axis=1))
test_loss = -np.mean(np.log(np.sum(a2_test * test_labels, axis=1)))
print(f"\n测试集 → 损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")

#画图模块
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title('Loss')
plt.xlabel('Epoch')
plt.subplot(1, 2, 2)
plt.plot(acc_history)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()


num_examples = 0    #可以自行进行修改来判断是否增加数量
plt.figure(figsize=(12, 3))
for i in range(num_examples):
   # 取第 i 张测试图片
   image = test_data[i].reshape(28, 28)           # 784 → 28×28
   true_label = np.argmax(test_labels[i])         # 真实数字
   pred_label = np.argmax(a2_test[i])             # 预测数字
   plt.subplot(1, num_examples, i + 1)
   plt.imshow(image, cmap='gray')                 # 显示灰度图
   plt.title(f'real:{true_label} forcast:{pred_label}')
   plt.axis('off')
   plt.tight_layout()
   plt.show()

np.save('w1.npy',w1)
np.save('b1.npy',b1)
np.save('w2.npy',w2)
np.save('b2.npy',b2)

#手写板python 根目录/digit_recognizer.py

#训练一次产生的对应的变化
# z1=train_data@w1+b1
# a1=np.maximum(0,z1)
# z2=a1@w2+b2
# a2=np.exp(z2)/np.sum(np.exp(z2),axis=1,keepdims=True)
# loss_new=-np.mean(np.log(np.sum(a2*train_labels,axis=1)))
# acc_new=np.mean(np.argmax(a2,axis=1)==np.argmax(train_labels,axis=1))
# print(f"更新前损失：{ave_loss:.4f},准确率：{acc:.4f}")
# print(f"更新后损失：{loss_new:.4f},准确率：{acc_new:.4f}")

# print("da1",da1.shape)
# print('dz1',dz1.shape)
# print('dw1',dw1.shape)
# print('db1',db1.shape)


# print("dw2",dw2.shape)
# print("db2",db2.shape)

# print(acc)

# print(a2.size)
# print(a2[1])
# print('a2形状',a2.shape)
# print('前三张图片的概率和：',np.sum(a2[:3],axis=1))

# print(cel(a2,train_labels[0]))

# print('真实概率',train_labels[0])
# print('预测概率',a2)
# print('和',np.sum(a2))
# print('预测的数字',np.argmax(a2))
# print(train_labels[0][5])

