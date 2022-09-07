#    一种差异化缩放方法

![图0](quote\图0.jpg "图0")
## 概述
本方法基于seam_carving的动态规划思想，同时引入视觉注意力机制，通过自顶向下和自底向上两个方向进行视觉显著性检测，大大拓宽了原算法的应用范围
## 问题分析  
参考论文中的方法在风景照片的缩放中取得了良好的效果，其基于一个假设：  __图片的高频部分保存着更多的更值得注意的信息，所以应该对低频区域进行删减或插入操作__   

但仔细推敲一下，我们可以发现两个不适用上述假设的反例：
- 有时候我们会用低频部分传递信息，比如图片中的文字和LOGO(图1) 
- 人们对低频区域的变换会感到强烈的违和感，如人脸(图2)  

对上述两类图片应用seam_carving算法，效果并不理想

![图1](quote\图1.jpg "图一")
<center>图1</center>  


![图2](quote\图2.jpg "图二")
<center>图2</center> 

## 解决方法
为了得到缩放质量更好的图片，我决定在计算图片的能量图之前先进行视觉显著性检测
- 自底向上的检测  
  借鉴了FT算法，针对图像中的色彩进行逐像素的显著性检测，有以下步骤
  - 对图像进行5*5的高斯平滑。
  - 转换颜色空间。RGB颜色空间转换为LAB颜色空间。
  - 计算整幅图片的l、a、b的平均值。
  - 按照算法中的公式，计算每个像素l、a、b值同图像三个l、a、b均值的欧氏距离。得到显著图
  - 归一化。图像中每个像素的显著值除以最大的那个显著值。得到最终的显著图。  
  
```
img = cv2.GaussianBlur(img,(5,5), 0)
gray_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
l_mean = np.mean(gray_lab[:,:,0])
a_mean = np.mean(gray_lab[:,:,1])
b_mean = np.mean(gray_lab[:,:,2])
lab = np.square(gray_lab- np.array([l_mean, a_mean, b_mean]))
lab = np.sum(lab,axis=2)
lab = lab/np.max(lab) 
```
  将显著图化为灰度图进行可视化，灰度越高的像素在人眼中越醒目（图3）  
  ![图3](quote\图3.jpg "图三")
  <center>图3</center>

- 自顶向下的检测  
  利用先验知识检测到图中人脸，将检测区域视为显著区域(图4)  
  PS：采用LBP算子检测人脸，基于统计方法，而非神经网络
  
```
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
frame = cv2.imread(r"D:\\seam-carving-master\\in\\images\\3.jpg")
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor = 1.15,#缩放比
    minNeighbors = 10,#敏感度
    minSize = (5,5),
)
print ("发现{0}个人脸!".format(len(faces)))
frame[:,:]=(0,0,0)
for(x,y,w,h) in faces:
    cv2.circle(frame, (int(x+w/2),int(y+h/2)), int(w/2), (255,255,255), -1)
    
    gray[y:y+h, x:x+w]=255
    face_area_draw = frame[y:y+h, x:x+w]
```
  ![图4](quote\图4.jpg "图四")
<center>图4</center>   

## 算法
在原算法的基础上，本方法将能量函数定义为  

$$
e(I)=e_{g} (I)+\alpha (\beta e_{s}(I)+\gamma e_{p}(I) ) 
$$

α为显著性指标，取值为[0,+∞]，β和γ表明是否采用自底向上或自顶向下的显著性检测，取值为0或1    

其中梯度能量函数：  

$$
e_{g} (I)=\left | \frac{\partial }{\partial x}I  \right | +\left | \frac{\partial }{\partial y}I  \right |   
$$

其中显著图：  
$$
e_{s}(I)=
\begin{bmatrix}
 S(p_{11})&S(p_{12})&··· \\
 S(p_{21})&S(p_{22})&··· \\
 ···&···&S(p_{mn})
\end{bmatrix}
$$
显著函数：   
$$
S(p_{ij})=\left | \left | I_{\mu }- I_{\omega hc }(p_{ij}) \right |  \right | 
$$
其中Iu为图像的平均特征，使用Lab颜色特征，后一项为像素p在高斯平滑后的Lab颜色特征,使用L2范数计算欧式距离


其它的步骤与Seam-carving算法思想大致相同，具体实现请见源码  
缩小操作：
- 对能量图进行动态规划，得到能量最小路线
- 删除能量最小路线上的所有像素
- 合并左右部分图片
- 如果达到指定宽高，保存并结束；否则返回步骤1   

放大操作：
- 计算宽度差detalx
- 对能量图进行动态规划，得到能量最小的前detalx条路线
- 对路线上的所有像素复制并插入图片
## 总结  

优点： 
- 缩放质量好 
- 拓宽了差异化缩放的应用范围  
- 算法简洁高效鲁棒性强  

不足： 
- 计算时间过长，缩小每个像素都需要重新计算能量图  
- 需要人为确定是否需要添加视觉显著性影响
  

待改进：  
- 可一次取能量函数最小的前几个缝隙进行删除，节约时间
- 可利用先验知识改进显著性检测方法
  - 如图片显著物体一般在中心，因此可在显著图中心叠加高斯核函数
  - 图像前景比背景更重要，因此可用像素显著值乘以像素深度的倒数
- 可先用神经网络对图片进行分类（风景，人物，LOGO等），再进行显著性检测

