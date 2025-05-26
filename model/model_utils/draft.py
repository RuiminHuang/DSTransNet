import numpy as np


# 验证生成了numpy.ndarray，并且可以直接取第0个元素
inter = np.array([5])
print(inter)
print(type(inter))
print(type(inter[0]))


# 验证int加上numpy.ndarray会后，会直接变为numpy.ndarray
temp = 0
print(temp)
print(type(temp))
temp += inter
print(temp)
print(type(temp))


# 验证是直接加，而不是拼接
inter1 = np.array([6])
inter2 = np.array([7])
inter3 = np.array([8])
inter4 = np.array([9])
print(inter1 + inter2 + inter3 + inter4)


# 以下赋值方式不合理，也会存在警告
# area_inter_1 = np.zeros(2)
# area_inter_1[0] = inter1
# area_inter_1[1] = inter2
# print(area_inter_1)
# area_inter_2 = np.zeros(2)
# area_inter_2[0] = inter3
# area_inter_2[1] = inter4
# print(area_inter_2)
# 改成这样子就OK
area_inter_1 = np.zeros(2)
area_inter_1[0] = inter1[0]
area_inter_1[1] = inter2[0]
print(area_inter_1)
area_inter_2 = np.zeros(2)
area_inter_2[0] = inter3[0]
area_inter_2[1] = inter4[0]
print(area_inter_2)


# 验证可以apend
append = np.append(area_inter_1, area_inter_2)
print(append)#[6. 7. 8. 9.]