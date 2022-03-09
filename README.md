# Image-Enhancement
Some methods of image enhancement

## <font face=楷体>一、灰度世界算法
<font face=楷体 size=4>① 算法原理
<font face=楷体 size=4>灰度世界算法以灰度世界假设为基础，该假设认为：对于一幅有着大量色彩变化的图像，R,G,B三个分量的平均值趋于同一灰度值Gray。从物理意义上讲，灰色世界法假设自然界景物对于光线的平均反射的均值在总体上是个定值，这个定值近似地为“灰色”。颜色平衡算法将这一假设强制应用于待处理图像，可以从图像中消除环境光的影响，获得原始场景图像。

<font face=楷体 size=4>一般有两种方法确定Gray值

<font face=楷体 size=4>1) 使用固定值，对于8位的图像(0~255)通常取128作为灰度值

<font face=楷体 size=4>2) 计算增益系数,分别计算三通道的平均值avgR，avgG，avgB，则：

<font face=楷体 size=4>$$ Avg=(avgR + avgG + avgB) / 3 $$

<font face=楷体 size=4>$$kr=Avg/avgR  \\
kg=Avg/avgG \\ kb=Avg/avgB$$

<font face=楷体 size=4>利用计算出的增益系数，重新计算每个像素值，构成新的图片

<font face=楷体 size=4>② 算法优缺点
<font face=楷体 size=4>这种算法简单快速，但是当图像场景颜色并不丰富时，尤其出现大块单色物体时，该算法常会失效。

<font face=楷体 size=4>vegetable.png效果明显， sky.png效果不明显
![请添加图片描述](https://img-blog.csdnimg.cn/a2fe551d56a346d39f50832081f728bf.png)
![请添加图片描述](https://img-blog.csdnimg.cn/036a3682290c45f7a5512da9541d15ab.png)
## <font face=楷体>二、直方图均衡化
<font face=楷体 size=4>① 算法原理
<font face=楷体 size=4>直方图均衡化，一般可用于灰度图像的对比增强（如：人脸阴影部位增强）；

<font face=楷体 size=4>② 算法优缺点
<font face=楷体 size=4>如果直接对彩色图像R,G,B三通道分别均衡化后再合并，极容易出现颜色不均、失真等问题，所以，一般会将RGB图像转换到YCrCb空间，对Y通道进行均衡化（Y通道代表亮度成分)
![请添加图片描述](https://img-blog.csdnimg.cn/c351990175424a26b54df899a0a00d82.png)
## <font face=楷体>三、Retinex算法
① 算法原理
<font face=楷体 size=4>视网膜-大脑皮层（Retinex）理论认为世界是无色的，人眼看到的世界是光与物质相互作用的结果，也就是说，映射到人眼中的图像和光的长波（R）、中波（G）、短波（B）以及物体的反射性质有关

![请添加图片描述](https://img-blog.csdnimg.cn/a78a936b607a4d22b5a12a60ab844cd5.png?#pic_center)
$$I(x, y)=R(x, y)L(x,y)$$
<font face=楷体 size=4>其中I是人眼中看到的图像，$R$是物体的反射分量，$L$是环境光照射分量，$(x, y)$是二维图像对应的位置

<font face=楷体 size=4>它通过估算$L$来计算$R$，具体来说，$L$可以通过高斯模糊和I做卷积运算求得，用公式表示为：
$$log(R)=log(I)-log(L)\\L=F*I$$
<font face=楷体 size=4>其中$F$是高斯模糊的滤波器，$*$表示卷积运算
$$F=\frac{1}{\sqrt{2\pi}\sigma}exp(\frac{-r^2}{\sigma^2})$$
其中 $\sigma$ 称为高斯周围空间常数（Gaussian Surround Space Constant），也就是算法中所谓的尺度，对图像处理有比较大的影响，对于二维图像，$r^2$ 等于对应位置即：$x^2+y^2$，即一般认为光照分量是原图像经过高斯滤波后的结果

<font face=楷体 size=4>② 算法优缺点

<font face=楷体 size=4>Retinex算法，从SSR（单尺度Retinex）到MSR（多尺度Retinex）以及到最常用的MSRCR（带颜色恢复的多尺度Retinex）；其中色彩恢复主要目的是来调节由于图像局部区域对比度增强而导致颜色失真的缺陷
如果是灰度图像，只需要计算一次即可，如果是彩色图像，如RGB三通道，则每个通道均需要如上进行计算
<font face=楷体 size=4>先看一组公式：

<font face=楷体 size=4>RMSRCR(x,y)'=G⋅RMSRCR(x,y)+b

<font face=楷体 size=4>RMSRCR (x,y)=C(x,y)RMSR(x,y)

<font face=楷体 size=4>C(x,y)=f[I'(x,y)]=f[I(x,y)/∑I(x,y)]Ci(x,y)=f[Ii′(x,y)]=f[Ii(x,y)∑j=1NIj(x,y)] 

<font face=楷体 size=4>f[I'(x,y)]=βlog[αI'(x,y)]=β{log[αI'(x,y)]−log[∑I(x,y)]}

<font face=楷体 size=4>G表示增益Gain（一般取值：5）

<font face=楷体 size=4>b表示偏差Offset（一般取值：25）

<font face=楷体 size=4>I (x, y)表示某个通道的图像

<font face=楷体 size=4>C表示某个通道的彩色回复因子，用来调节3个通道颜色的比例；

<font face=楷体 size=4>f(·)表示颜色空间的映射函数；

<font face=楷体 size=4>β是增益常数（一般取值:46）；

<font face=楷体 size=4>α是受控制的非线性强度(一般取值：125)

<font face=楷体 size=4>MSRCR算法利用彩色恢复因子C，调节原始图像中3个颜色通道之间的比例关系，从而把相对较暗区域的信息凸显出来，达到了消除图像色彩失真的缺陷。 处理后的图像局部对比度提高，亮度与真实场景相似，在人们视觉感知下，图像显得更加逼真；但是MSRCR算法处理图像后，像素值一般会出现负值。所以从对数域r(x, y)转换为实数域R(x, y)后，需要通过改变增益Gain，偏差Offset对图像进行修正![请添加图片描述](https://img-blog.csdnimg.cn/fcc84ca3f185456e9fdb5e011441e06f.png)
## <font face=楷体>四、自动白平衡(AWB)
<font face=楷体 size=4>① 算法原理

 <font face=楷体 size=4>用一个简单的概念来解释什么是白平衡：假设，图像中R、G、B最高灰度值对应于图像中的白点，最低灰度值的对应于图像中最暗的点；其余像素点利用(ax+b)映射函数把彩色图像中R、G、B三个通道内的像素灰度值映射到[0.255]的范围内.

<font face=楷体 size=4>白平衡的本质是让白色的物体在任何颜色的光源下都显示为白色，这一点对人眼来说很容易办到，因为人眼有自适应的能力，只要光源的色彩不超出一定的限度，就可以自动还原白色。但相机就不同了，无论是图像传感器还是胶卷都会记录光源的颜色，白色的物体就会带上光源的颜色，白平衡所要做的就是把这个偏色去掉。

<font face=楷体 size=4> ② 算法优缺点</font>>

<font face=楷体 size=4>自动白平衡是一个很复杂的问题，目前还没有一个万能的方法可以解决所有场景的白平衡问题![请添加图片描述](https://img-blog.csdnimg.cn/34e96eb48ae04d92a3202ec9ee70febf.png)
## <font face=楷体>五、自动色彩均衡(ACE)
<font face=楷体 size=4>① 算法原理

<font face=楷体 size=4>ACE算法源自retinex算法，可以调整图像的对比度，实现人眼色彩恒常性和亮度恒常性，该算法考虑了图像中颜色和亮度的空间位置关系，进行局部特性的自适应滤波，实现具有局部和非线性特征的图像亮度与色彩调整和对比度调整，同时满足灰色世界理论假设和白色斑点假设。

<font face=楷体 size=4>第一步：对图像进行色彩／空域调整，完成图像的色差校正，得到空域重构图像；
$$R_c(p)=\sum_{j\in Subset,j\ne p}\frac{r(I_c(p)-I_c(j))}{d(p,j)}$$
<font face=楷体 size=4>式中，$Rc$ 是中间结果，$I_c(p)-I_c(j)$为两个不同点的亮度差，$d(p,j)$ 表示距离度量函数，$r(*)$为亮度表现函数，需是奇函数；这一步可以适应局部图像对比度，$r(*)$能够放大较小的差异，并丰富大的差异，根据局部内容扩展或者压缩动态范围。一般得，$r(*)$为：
$$r(x)=\begin{cases}
1&,x<-T \\
x/T&,-T\leqslant x\leqslant T \\
-1&,x>T\end{cases}$$
<font face=楷体 size=4>第二步：对校正后的图像进行动态扩展。ACE算法是对单一色道进行的，对于彩色图片需要对每一个色道分别处理
其中存在一种简单的线性扩展：

$R(x)=round[127.5+w*R_c(p)]$,其中，$w$表示线段$[(0,m_c),(255,M_c)]$的斜率，且有:
$M_c=min[R_c(p)]，M_c=max[R_c(p)]$

<font face=楷体 size=4>第三步：利用下面的公式将$R(x)$展到$[0,1]$之间，得到增强后的通道
$$L(x)=\frac{R(x)-minR}{maxR-minR}$$
<font face=楷体 size=4>②算法优缺点

<font face=楷体 size=4>ACE的增强效果普遍比retinex好。需要注意的是，ACE中当前像素是与整个图像的其他像素做差分比较，计算复杂度非常非常高，这也是限制它应用的最主要原因。

<font face=楷体 size=4>所以，一般算法中，会通过指定采样数来代替与整副图像的像素点信息进行差分计算,减少运算量，提高效率。
![请添加图片描述](https://img-blog.csdnimg.cn/de5137328ff247efaa5dcff09321e75e.png)
<font face=楷体 size=4>总结：查看各种传统算法的效果图，ACE自动色彩均衡算法具有比较好的普遍性和效果，当然，对于一些图片ACE也不能得到很好地效果
