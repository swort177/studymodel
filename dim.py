b = torch.tensor([
    [
      [1, 1, 1],
      [1, 1, 1]
    ],
    [
      [2, 2, 2],
      [2, 2, 2]
    ],
    [
      [3, 3, 3],
      [3, 3, 3]
    ],
])

# b = torch.tensor([
#     [
#       [1, 1, 1],
#       [1, 1, 1]
#     ],
#     [
#       [2, 2, 2],
#       [2, 2, 2]
#     ],
#     [
#       [3, 3, 3],
#       [3, 3, 3]
#     ],
# ])


print(b.shape)
# torch.Size([3, 2, 3])
print(b[0])
# tensor([[1, 1, 1],
#     [1, 1, 1]])
print(b[1])
# tensor([[2, 2, 2],
#     [2, 2, 2]])
print(b[2])
# tensor([[3, 3, 3],
#     [3, 3, 3]])
dim0=torch.sum(b,dim=0)
print(dim0,dim0.shape)  
  
# 表示合并b[0] b[1] b[2]得到  
#  就是b[0]+b[1]+b[2]
# tensor([[6, 6, 6],
#     [6, 6, 6]]) 
# torch.Size([2, 3])
#从结果看就是第一维度去掉了 从 torch.Size([3, 2, 3])-----》torch.Size([（3丢掉了），2, 3])
#理解为 丢掉第一维的数据 怎么样丢掉第一维呢  想象在批次上叠加在一起就是了 b[0]+b[1]+b[2]
#在理解 这个数列表示 3批 2行3列的数据  丢掉第一维也就是去掉批次这个，那就需要转换数据，怎么样转换不损坏不丢失数据呢
#也就是所有数据保持不丢失的情况下 那就是叠加在一起呗 批次上叠加 在合并

dim1=torch.sum(b,dim=1)
print(dim1,dim1.shape)
# tensor([[2, 2, 2],
#     [4, 4, 4],
#     [6, 6, 6]]) 
# torch.Size([3, 3])
#从结果看就是第二维度去掉了 从 torch.Size([3, 2, 3])-----》torch.Size([3, （2丢掉了） 3])
#理解为 丢掉第二维的数据 怎么样丢掉第二维呢 变成[3,3]  想象成在行上叠加
# 比如[2,2,2]=[1, 1, 1]+[1, 1, 1] 
#在理解 这个数列表示 3批 2行3列的 数据  丢掉第二维也就是去掉2行这个，那就需要转换数据，怎么样转换不损坏不丢失数据呢
#也就是所有数据保持不丢失的情况下  也就是行挤压 在合并

dim2=torch.sum(b,dim=2)
print(dim2,dim2.shape)

# tensor([[3, 3],
#     [6, 6],
#     [9, 9]]) 
# torch.Size([3, 2])

#从结果看就是第三维度去掉了 从 torch.Size([3, 2, 3])-----》torch.Size([3, 2，（3丢掉了） ])
#理解为 丢掉第三维的数据  怎么样丢掉第二维呢 变成[3,2] 
#在理解 这个数列表示 3批 2行3列的 数据  丢掉第三维也就是去掉3列 这个，那就需要转换数据，怎么样转换不损坏不丢失数据呢
#也就是所有数据保持不丢失的情况下  也就是列挤压 在合并
