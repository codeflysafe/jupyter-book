#!/usr/bin/env python
# coding: utf-8

# ---
# title: linear transformer
# date: 2022-12-31 22:47:57
# mathjax: true
# tags:
#    - matrix
#    - math
#    - linear transformer
# ---

# ## Linear transformer 线性变换
# 
# 两个特点：
# - 直线保持不变
# - 原点保持不变
# 	
# 主要是保持网格线平行和等距分布

# 以 $p\in \mathbb{R}^2 = (x, y)^T$ 为例：
# $$
# \begin{equation}
# \begin{aligned}
# p&=x\vec{e_1}+y\vec{e_2} \\\\
#  &= \begin{bmatrix}
#          \vec{e_1} & \vec{e_2}
#       \end{bmatrix} \begin{bmatrix}
#          x \\\\ y
#       \end{bmatrix}\\\\
# p^′&=x(transformer_1 )+y(transformer_2 ) \\\\
# &= \begin{bmatrix}
#          \vec{transformer_1} & \vec{transformer_2}
#       \end{bmatrix} \begin{bmatrix}
#          x \\\\ y
#       \end{bmatrix}
# \end{aligned}
# \end{equation}
# $$
# 只需要求出变换后的 $transformer_1$ 和 $transformer_2$ 即可

# ### 几种常见的线性变换
# 对于常见的线性变换，只需要记住一点：只需要找出变换后的基坐标即可.
# $$
#   \vec{x} = x_1\vec{e_1} + \cdots + x_n\vec{e_n} = \begin{bmatrix}
#          x_1 \\\\ x_2 \\\\ \dots \\\\ x_n
#       \end{bmatrix}
# $$
# $\vec{x}$代表在基坐标$\mathbb{R}^{n}$上的坐标向量
# 则有:
# $$
# \begin{equation}
#    \label{eq_1}
#    \begin{aligned}
#       L(\vec{x}) &= x_iL(\vec{e_1}) + \cdots + x_nL(\vec{e_n})
#    \end{aligned}
# \end{equation}
# $$
# 
# 将$\eqref{eq_1}$ 带入到 $\eqref{eq_2}$ 中：
# $$
# \begin{equation}
#    \label{eq_2}
#    \begin{aligned}
#    A &= \begin{bmatrix}
#      a_{11} & a_{12} & \cdots &a_{1n} \\\\
#      a_{21} & a_{22} & \cdots &a_{2n} \\\\
#              & \ddots & \\\\
#      a_{n1} & a_{n2} & \cdots &a_{nn} 
#       \end{bmatrix} \\\\
#    L(\vec{x}) &= A\vec{x}  \\\\
#    &= \begin{bmatrix}
#      a_{11} & a_{12} & \cdots &a_{1n} \\\\
#      a_{21} & a_{22} & \cdots &a_{2n} \\\\
#              & \ddots & \\\\
#      a_{n1} & a_{n2} & \cdots &a_{nn} 
#       \end{bmatrix} \begin{bmatrix}
#          x_1 \\\\ x_2 \\\\ \dots \\\\ x_n
#       \end{bmatrix} \\\\
#    &= \begin{bmatrix}
#       L(e_1) & L(e_2) & \cdots & L(e_n)
#       \end{bmatrix} \begin{bmatrix}
#          x_1 \\\\ x_2 \\\\ \dots \\\\ x_n
#       \end{bmatrix}
#    \end{aligned}
# \end{equation}
# $$
# $A \in \mathbb{R}^{n \times n}$ 是线性变换， $x \in \mathbb{R}^{n}$, 则其变换前的基坐标为: 
# $$
# e = \begin{bmatrix}
#    \vec{e_1}& \vec{e_2}& \cdots & \vec{e_n}
# \end{bmatrix}
# $$
# 
# 则 A 如下所示：
# $$ 
# \begin{equation}
#    \label{eq_3}
#    A = \begin{bmatrix} L(e_1) & L(e_2) & \cdots & L(e_n) \end{bmatrix}
# \end{equation}
# $$

# #### $\mathbb{R}^{2}$ 逆旋转 $\theta$ 
# 旋转只改变角度，不改变大小, 根据 $\eqref{eq_3}$ 可得：
# $$
#  \begin{equation}
#      \label{eq:rotate}
#      \begin{aligned}
#          A = \begin{bmatrix} L(e_1) & L(e_2) \end{bmatrix}
#      \end{aligned}
#  \end{equation}
# $$
# 

# $$
# \begin{equation}
#  \begin{aligned}
#    \begin{bmatrix}
#    x^{'} \\  y^{'} \\ 1 
#    \end{bmatrix} &= \begin{bmatrix}
#    \cos{\theta} & -\sin{\theta} & 0 \\
#    \sin{\theta} & \cos{\theta} & 0 \\
#    0 & 0 & 1  
#    \end{bmatrix}\begin{bmatrix}
#    x \\  y \\ 1 
#    \end{bmatrix}
#  \end{aligned}
# \end{equation}
# $$

# ![image.png](attachment:image.png)

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import math
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
fig,ax=plt.subplots()
#绘制从[0,0]到[2,2]的向量
a=ax.quiver(0, 0, 2, 2, angles='xy', scale_units='xy', scale=1)
a=ax.quiver(0, 0, 2*math.sqrt(2) - 0.3, 0.3, angles='xy', scale_units='xy', scale=1)
plt.text(0.5,0.2, r'$\theta$')
plt.text(2,2, 'A')
plt.text(2*math.sqrt(2) - 0.3, 0.3, r'$A^{r}$')
ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
plt.show()


# #### 平移 
# 假设x、y方向平移的偏移量分别为 dx 和 dy， 则有：
# $$
# \begin{equation}
#    \begin{aligned}
#    x^{'} &= x + dx \\
#    y^{'} &= y + dy \\
#    \begin{bmatrix}
#    x^{'} \\  y^{'} \\ 1 
#    \end{bmatrix} &= \begin{bmatrix}
#    1 & 0 & dx \\
#    0 & 1 & dy \\
#    0 & 0 & 1  
#    \end{bmatrix}\begin{bmatrix}
#    x \\  y \\ 1 
#    \end{bmatrix}
#    \end{aligned}
# \end{equation}
# $$

# #### 缩放
# 
# 

# $$
# \begin{equation}
#  \begin{aligned}
#    \begin{bmatrix}
#    x^{'} \\  y^{'} \\ 1 
#    \end{bmatrix} &= \begin{bmatrix}
#    s & 0 & 0 \\
#    0 & t & 0 \\
#    0 & 0 & 1  
#    \end{bmatrix}\begin{bmatrix}
#    x \\  y \\ 1 
#    \end{bmatrix}
#  \end{aligned}
# \end{equation}
# $$

# #### 翻转
# 
# 

# x 轴对称
# $$
# \begin{equation}
#  \begin{aligned}
#    \begin{bmatrix}
#    x^{'} \\  y^{'} \\ 1 
#    \end{bmatrix} &= \begin{bmatrix}
#    1 & 0 & 0 \\
#    0 & -1 & 0 \\
#    0 & 0 & 1  
#    \end{bmatrix}\begin{bmatrix}
#    x \\  y \\ 1 
#    \end{bmatrix}
#  \end{aligned}
# \end{equation}
# $$

# y 轴对称
# $$
# \begin{equation}
#  \begin{aligned}
#    \begin{bmatrix}
#    x^{'} \\  y^{'} \\ 1 
#    \end{bmatrix} &= \begin{bmatrix}
#    -1 & 0 & 0 \\
#    0 & 1 & 0 \\
#    0 & 0 & 1  
#    \end{bmatrix}\begin{bmatrix}
#    x \\  y \\ 1 
#    \end{bmatrix}
#  \end{aligned}
# \end{equation}
# $$

# #### 错切

# 错切 y 方向
# 
# $$
# \begin{equation}
#  \begin{aligned}
#    \begin{bmatrix}
#    x^{'} \\  y^{'} \\ 1 
#    \end{bmatrix} &= \begin{bmatrix}
#    1 & 0 & 0 \\
#    s & 1 & 0 \\
#    0 & 0 & 1  
#    \end{bmatrix}\begin{bmatrix}
#    x \\  y \\ 1 
#    \end{bmatrix}
#  \end{aligned}
# \end{equation}
# $$

# 错切 x 方向
# $$
# \begin{equation}
#  \begin{aligned}
#    \begin{bmatrix}
#    x^{'} \\  y^{'} \\ 1 
#    \end{bmatrix} &= \begin{bmatrix}
#    1 & r & 0 \\
#    0 & 1 & 0 \\
#    0 & 0 & 1  
#    \end{bmatrix}\begin{bmatrix}
#    x \\  y \\ 1 
#    \end{bmatrix}
#  \end{aligned}
# \end{equation}
# $$

# ### 基变换
# 从一个坐标系的变换到另一个坐标系的变换
# 
# 一组基是 $U= [\vec{u_1} ,  \vec{u_2}]$  另一组为  $V= [\vec{v_1}, \vec{v_2}]$
# 在对应基下的坐标向量位 $(x, y)^T$ 和 $(a,b)^T$
# 则：
#    $$ 
#    \begin{equation}
#       \begin{aligned}
#       x\vec{u_1} +y\vec{u_2} &= a\vec{v_1}+b \vec{v_2} \\\\
#       \begin{bmatrix}
#          \vec{u_1} & \vec{u_2}
#       \end{bmatrix} \begin{bmatrix}
#          x \\\\ y
#       \end{bmatrix} &= \begin{bmatrix}
#          \vec{v_1} & \vec{v_2}
#       \end{bmatrix} \begin{bmatrix}
#          a \\\\ b
#       \end{bmatrix} \\\\ 
#       \begin{bmatrix}
#          x \\\\ y
#       \end{bmatrix} &= \begin{bmatrix}
#          \vec{u_1} & \vec{u_2}
#       \end{bmatrix}^{-1} \begin{bmatrix}
#          \vec{v_1} & \vec{v_2}
#       \end{bmatrix} \begin{bmatrix}
#          a \\\\ b
#       \end{bmatrix} \\\\
#        &= U^{-1}V  \begin{bmatrix}
#          a \\\\ b
#       \end{bmatrix} \\\\
#       &= S \begin{bmatrix}
#          a \\\\ b
#       \end{bmatrix} \\\\
#       \begin{bmatrix}
#          a \\\\ b
#       \end{bmatrix} &= \begin{bmatrix}
#          \vec{v_1} & \vec{v_2}
#       \end{bmatrix}^{-1} \begin{bmatrix}
#          \vec{u_1} & \vec{u_2}
#       \end{bmatrix} \begin{bmatrix}
#          x \\\\ y
#       \end{bmatrix} \\\\
#        &= V^{-1}U  \begin{bmatrix}
#          x \\\\ y
#       \end{bmatrix} \\\\
#       &= S^{-1} \begin{bmatrix}
#          x \\\\ y
#       \end{bmatrix} \\\\
#       \end{aligned}
#    \end{equation}
#    $$
#      
# S 称为 V →U  的转移矩阵 , 特殊情况当 U 为标准基时 S = V
