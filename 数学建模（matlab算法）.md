# 统计

## 1 评价与决策

### 1.1 层次分析法

#### 1.1.1 层次分析法模型部分

> 确定评价指标、形成评价体系—————评价类问题
>
> （1）评价目标
>
> （2）几种方案可以达到目标
>
> （3）评价的准则或目标

示例：

> 小明要去旅游，可以选择的目的地有A、B、C三地，评价出最佳旅游地。
>
> （1）根据材料影响因素有景色、花费、居住、饮食、交通
>
> |    标度    |              含义              |
> | :--------: | :----------------------------: |
> |     1      |            同样重要            |
> |     3      |            稍微重要            |
> |     5      |            明显重要            |
> |     7      |            强烈重要            |
> |     9      |            极端重要            |
> | 2、4、6、8 |      上述量相邻判断的中值      |
> |    倒数    | A和B相比标度为3则B和A相比为1/3 |
>
> 得到5X5的矩阵(正互反矩阵或判断矩阵)，记为A，对应的元素为a~ij~.
>
> 这个方阵的特点：
>
> （1）a~ij~表示的意义，与指标j相比，i的重要程度。
>
> （2）a~ij~>0且满足a~ij~ X a~ij~=1
>
> 根据材料得到方案对应于准则的判断矩阵
>
> 一致矩阵：判断矩阵满足a~ij~ X a~jk~=a~ik~
>
> *引理：n阶正互反矩阵A为一致时当且仅当最大特征值*λ~max~=n。
>
> 非一致时，一定满足λ~max~>n。
>
> ![](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/1657896128926.png)
>
> ![](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/1657896211905.png)
>
> ![](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/1657896299613.png)
>
> ![](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/1657896452369.png)
>
> ![image-20220719212016991](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220719212016991.png)
>
> ![image-20220719212043114](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220719212043114.png)
>
>  ![image-20220719212241596](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220719212241596.png)
>
> ![image-20220719212300250](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220719212300250.png)
>
> ![image-20220719213343420](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220719213343420.png)
>
> ![image-20220719214131150](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220719214131150.png)
>
> 

#### 1.1.2 层次分析法代码部分

**基础学习代码：**

```matlab
%% 多行注释Ctrl+R，取消注释Ctrl+T
clear  %清除工作区所有变量
clc  %清除命令行窗口所有文本

%% 输出函数和输入函数(disp()和input())
%matlab中两个字符串的合并
%(1)strcat('字符串1','字符串2'……)
%(2)[‘字符串1’，‘字符串2’]
%数字转换为字符串：num2str()
str1="kmxy";
str2="xxgcxy";
x=100;
disp('x的取值为：')
num2str(x)
disp('shr1+str2的取值为：')
strcat(str1,str2)
disp('str1+str2x的取值为：')
['kmxy' 'xxgcxy']

%input输入函数
%一般将输入的数、向量、矩阵、字符串等值赋一个变量
%语句结束加分号会影响显示
A=input('请输入A：');
B=input('请输入B：')

%% sum函数
%向量无论行向量列向量sum求和都一样
%矩阵：按列求和sum(X,1)或sum(X)
%矩阵：按列求和sum(X,2)
%整个矩阵求和sum(X(:))

%% size()函数
%矩阵的形式返回
X=[1 2 3 4 5 6];
size(X)  %得到1 6
%可以用
[r,x]=size(X);
% r=size(X,1);
% x=size(X,2);

%% matlab中矩阵的运算
%MATLAB在矩阵运算中，“*”号和“/”号代表矩阵之间的乘法与除法（A/B=A*inv(B)）
A = [1 2;3 4];
B = [1 0;1 1];
A .* B
A ./ B
inv(B) %求B的逆矩阵
B * inv(B)
A * inv(B)
A / B
%对A的每一个元素进行平方".^2","A*A=A^2"

%% MATLAB中求特征值和特征向量eig(X)函数
A=[1 2 3;2 2 1;2 0 3];
%(1)E=eig(A):求矩阵A的全部特征值，构成向量E
E=eig(A)
%(2)[V,D]=eig(A):求矩阵A的全部特征值,构成对角矩阵D，并求A的特征向量构成V的列向量，
[V,D]=eig(A)

%% find函数的用法
% find函数，它可以用来返回向量或者矩阵中不为零的位置索引
X=[1 0 4 -3 0 0 8 6];
FIND_X=find(X);
disp(FIND_X) 


```

**模型代码：**

```matlab
%% 输入判断矩阵
clear;clc
disp('请输入判断矩阵：')
A = input('判断矩阵A=')

%% 方法1：算术平均法求权重
%将判断矩阵按列归一化
sum_A=sum(A); %每一列求和
[n,m]=size(A); %可以写成n=size(A,1)得到矩阵行列大小
SUM_A=repmat(sum_A,n,1);%sum_A形成一个n*1的矩阵
%可以用如下方法：
% SUM_A=[];
% for i = 0 : n
%     SUM_A = [SUM_A;sum_A];
% end

Stand_A=A./SUM_A; %直接将两个矩阵点除
%第二步：将归一化各行求和
%sum(Stand_A,2);
%将求和得到的向量中每一个元素除以n得到算术平均权重
disp('算术平均法求权重的结果为：')
disp(sum(Stand_A,2)/n)

%% 方法2：几何平均法求权重
%第一步：将A的元素按行相乘得到一个新的列向量
PROD_A = prod(A,2);  %prod和sum一样，一个乘一个加
%第二步：将新的列向量的每一分量开n次方
Prodct_A = PROD_A.^(1/n);
%第三步：对该列向量进行归一化即可以得到权重向量
disp('几何平均法求权重的结果为：')
disp(Prodct_A./sum(Prodct_A))

%% 方法3：特征值法求权重
%第一步：求出矩阵A的最大特征值以及其对应的特征向量
[V,D]=eig(A);
Max_eig = max(max(D));  %也可以Max_eig = max(D(:))
%D == Max_eig
[x,y]=find(D==Max_eig,1); %前一个不为零的元素位置
%第二步：对求出的特征向量进行归一化即可得到权重
disp('特征值法求权重的结果为：')
disp(V(:,y)/sum(V(:,y)))

%% 计算一致性比例CR
CI=(Max_eig-n) / (n-1);
RI=[0 0 0.52 0.89 1.12 1.26 1.36 1.41 1.46 1.49 1.52 1.56 1.54 1.56 1.58 1.59];
CR=CI/RI(n);
disp('一致性指标CI=');disp(CI);
disp('一致性比例标CR=');disp(CR);
if CR<0.01
    disp('因为CR<0.01，所以该判断矩阵A的一致性可以接受！');
else
    disp('注意：CR>0.01，所以该判断矩阵A需要修改！');
end

```

### 1.2 模糊综合评判

#### 1.2.1 模糊综合评判法模型部分

(1)模糊性与确定性：

确定性概念：

模糊性概念：

模糊集合的概念：

*模糊集可以记为A。 映射（函数）μ~A~(·) 或简记为A(·) 叫做模糊集A的[隶属函数](https://baike.baidu.com/item/隶属函数)。 对于每个*x*∈*U*，μ ~A~(x)叫做元素x对模糊集A的**[隶属度](https://baike.baidu.com/item/隶属度)**。*

模糊集的常用表示法有下述几种：

（1）解析法，也即给出隶属函数的具体表达式。

（2）Zadeh 记法，例如
$$
A=A(x~1~)/x~1~+A(x~2~)/x~2~+A(x~3~)/x~3~+A(x~4~)/x~4~
$$
$$
A=1/x~1~+0.5/x~2~+0.72/x~3~+0/x~4~
$$

 。分母是论域中的元素，分子是该元素对应的隶属度。有时候，若隶属度为0，该项可以忽略不写。

（3）序偶法，例如

$$A={(x~1~,1),(x~2~,0.5),(x~3~,0.75),(x~4~,0)}$$

 ，序偶对的前者是论域中的元素，后者是该元素对应的隶属度。

（4）向量法，在有限论域的场合，给论域中元素规定一个表达的顺序，那么可以将上述序偶法简写为隶属度的向量式，如*A*= (1,0.5,0.72,0) 。

设A，B是论域U的两个模糊集合，定义：

![image-20220725220840877](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220725220840877.png)

模糊矩阵：

![image-20220725221050938](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220725221050938.png)

![image-20220725221214512](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220725221214512.png)

![image-20220725221331412](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220725221331412.png)

![image-20220725221534675](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220725221534675.png)

![image-20220725221905982](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220725221905982.png)

![image-20220725221933102](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220725221933102.png)

![image-20220725222015459](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220725222015459.png)

![image-20220725222126083](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220725222126083.png)

![image-20220725222625397](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220725222625397-16587591873152.png)

### 1.3 主成分分析

#### 1.3.1 主成分分析模型部分

设有p项指标*X**1**,X**2**,…,X**p**，*以*x**1**,x**2**,…,x**n*表示的n个观测值，得到原始数据矩阵。

![image-20220801204047574](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220801204047574.png)

其中

![image-20220801204108613](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220801204108613.png)

设Fi表示第 i 个主成分，**i=1,2,…,p** ，则 p 项指标变量的主成份表示为：

![img](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/wps13.jpg) 

可以优化为

![img](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/wps15.jpg) 

其中 ![img](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/wps16.jpg) ，由于方差反映了数据的变异程度，上式的最优解就是一个主成份方向，主成份可以有多个，几何上两个主成份之间的 方向正交。在实际问题解决中根据主成分贡献率大小取主成分，一般情况下取主成分累积贡献率85%-100%的特征根所对应的主成分

根据上述主成分分析的基本原理，在本题中主成分分析计算步骤：

step1：建立预先处理后的宏观经济指标观测值矩阵

![image-20220801204212969](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220801204212969.png)

step2：对原来的p个观测指标进行标准化处理，避免指标变量在量纲上的影响

![img](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/wps21.jpg) 

其中![img](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/wps22.jpg)表示标准化后的数据，![img](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/wps23.jpg)和![img](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/wps24.jpg)分别表示第j个指标的样本数据均值和标准
差。

step3：根据标准化后的数据求出相关系数矩阵R，

![image-20220801204256141](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220801204256141.png)

因为短期贷款利率和定期存款利率相关系数矩阵值为0，因此剔除短期贷款利率和定期存款利率数据。

Step4：保留累计贡献率为 85%以上的特征值和对应的特征向量:

[mi1,mi2,…,mi20]  (i=1,2)

计算主成分荷载：

![img](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/wps31.jpg) 

其中![img](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/wps32.jpg)表示特征根，![img](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/wps33.jpg)表示特征向量。

对主成分荷载归一化。

则主成分可表示为：

![img](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/wps34.jpg) 



![image-20220801203718144](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220801203718144.png)

#### 1.3.2 主成分分析代码部分

**模型代码：**

```matlab
clear;  %清除变量
clc   %清除窗口
%% 输入样本观测值
data = input('请输入数据data=');

%% 1、标准化处理
X = zscore(data);  

%% 2、求相关系数
R = corrcoef(X);

%% 3、求X的特征值和特征向量
[coeff,D] = eig(R);
newD = sort(diag(D),'descend'); %从大到小排序特征值构成的向量
%newD=rot90(rot90(D));
coeff = fliplr(coeff);  %特征向量矩阵，正交矩阵，也是因子矩阵

%% 4、计算各主成分的贡献率和累积贡献率
explained = newD/sum(newD)*100;
Cumsun = cumsum(explained);

%% 5、计算个体得分
score = X*coeff;

%% 6、求累积贡献率超过85%
count=0;sumrate=0;
for k = 1:length(newD)
    sumrate = explained(k)+sumrate;
    count = count+1;
    if sumrate > 85
        break;
    end
end

%% 7、可拟化
subplot(1,2,1)
plot(newD,'k--o','LineWidth',1)
xlabel('newD Value'),ylabel('newD num')
title('newD-碎石图')
subplot(1,2,2)
plot(Cumsun,'ko--','LineWidth',1)
xlabel('Cumsun Value'),ylabel('Cumsun num')
title('Cumsum-累计贡献率')

%% 输出参数组合
disp(coeff)
disp(explained)
%% 求主成分Y于X的系数
for i = 1:count
    fprintf('主成分Y%d的系数为：',i)
    disp(coeff(:,i));
end

```

### 1.4 优劣解距离法（TOPSIS法）

#### 1.4.1 TOPSIS法模型部分

1、标准化处理

![image-20220802223115863](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220802223115863.png)

2、构造计算评分公式：
$$
(x-min)/(max-min)
$$
变形：
$$
(x-min)/(max-min)=(x-min)/(max-x)+(x-min)
$$
可以看作：x与最小的距离比x与最大的距离+x与最小的距离

![image-20220803092625419](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803092625419.png)

![image-20220803093737273](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803093737273.png)

最后归一化处理

![image-20220803093955463](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803093955463.png)

极小型指标转换为极大型指标：max-x

所有元素均为正数：1/x

![image-20220803094326816](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803094326816.png)

![image-20220803094556828](../../image-20220803094556828.png)

**示例：**

![image-20220803095110745](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803095110745.png)

#### 1.4.2 TOPSIS法模型拓展

![image-20220803100052576](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803100052576.png)

![image-20220803100109667](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803100109667.png)

![image-20220803100138983](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803100138983.png)



#### 1.4.3 TOPSIS法代码部分

**主程序：**

```matlab
clear; %清除变量
clc  %清空窗口

%% 1、导入数据
%X=input('请输入评价指标矩阵A=');
%X = xlsread('20条河流水质情况数据.xlsx');
%% 2、数据正向化处理
[n,m] = size(X);
disp(['共有' num2str(n) '个评价对象，' num2str(m) '个评价指标'])
%% 正向化数据处理函数输入数据
%如果是级小型则调用Min2Max(x)
%如果是中间型则调用Mid2Max(x,best)
%如果是中间型则调用Mid2Max(x,a,b)
%X(:,3) = Min2Max(X(:,3));
%X(:,2) = Mid2Max(X(:,2),7);
%X(:,4) = inter2Max(X(:,4),10,20);
disp('正向化后的矩阵X=')
disp(X)
%% 3、对正向化处理后的矩阵标准化
Z = X./repmat(sum(X.*X).^0.5,n,1);
disp('标准化矩阵Z=')
disp(Z)
%% 4、计算最大值的距离与最小值的距离
D_P = sum([(Z-repmat(max(Z),n,1)).^2],2).^0.5;  %与最大值的距离向量
D_N = sum([(Z-repmat(min(Z),n,1)).^2],2).^0.5;  %与最小值的距离向量
S = D_N./(D_P+D_N);  %归一化
disp('最后得分为：')
stand_S = S/sum(S);
disp(stand_S)
[sorted_S,Index] = sort(stand_S,'descend');
disp('    评分:     位置:')
disp([sorted_S,Index])
%% 4、可视化
bar(stand_S,'c')
xlabel('1-20分别对应A-T'),ylabel('评分值')
title('......')



```

**Min2Max()函数：**

```matlab
function [posit_x] = Min2Max(x)
    %极小型
    posit_x = max(x)-x;
end
```

**Mid2Max()函数:**

```matlab
function [posit_x] = Mid2Max(x,best)
    M = max(abs(x-best));
    posit_x = 1-abs(x-best)/M;
end
```

**inter2Max()函数:**

```matlab
%区间型
function [posit_x] = inter2Max(x,a,b)
    r_x = size(x,1);
    M = max([a-min(x),max(x)-b]);
    posit_x = zeros(r_x,1);
    for i=1:r_x
        if x(i)<a
            posit_x(i) = 1-(a-x(i))/M;
        elseif x(i)>b
            posit_x(i) = 1-(x(i)-b)/M;
        else
            posit_x(i) = 1;
        end
    end
end
```

### 1.5 秩和比综合评价

#### 1.5.1 秩和比综合评价法模型

第一步：编秩

![image-20220803211510959](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803211510959.png)

![image-20220803212036921](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803212036921.png)

![image-20220803212202202](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803212202202.png)

![image-20220803212317355](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803212317355.png)

![image-20220803212352079](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803212352079.png)

示例：

![image-20220803212554130](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803212554130.png)

![image-20220803212619628](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803212619628.png)

![image-20220803212847193](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803212847193.png)

![image-20220803212945091](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803212945091.png)

![image-20220803213054500](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803213054500.png)

![image-20220803213123204](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220803213123204.png)

### 1.6 因子分析

#### 1.6.1 因子分析模型部分

1.因子分析原理

1．因子分析原理分析

数学与统计 因子分析是将多个实测变量转换为少数几个不相关的综合指标的多元统计方法。它通过研究众多变量之间的内部依赖关系，探求观测数据中的基本结构，并用少数几个假想变量来表示其基本的数据结构。==假想变量是不可观测的潜在变量，称为因子。==

假定这p个有相关关系的随机变量含有m个彼此独立的因子，可表示为：

![image-20220806082318650](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806082318650.png)

或用矩阵表示为X=AF+ε。F1,F2,...,Fm称为==公共因子==，是不可观测的变量，它们的系数称为因子载荷，A称为==因子载荷矩阵==，它反映了公共因子对变量的重要程度，对解释公共因子具有重要的作用。ε是独有的特殊因子，是不能包含在公共因子的部分。

>![image-20220806082318650](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806082318650.png)
>
>需满足：
>
>m ≤ p,即公共因子数不超过原变量个数；
>
>公共因子之间互不相关，且每个Fi；方差为1，即F的协方差矩阵为单位矩阵I；
>
>公共因子和特殊因子之间彼此互不相关，即Cov(F,ε)=0; 特殊因子之间彼此互不相关，但方差不一定相同，记ε，的方差为：   
>
>​                                                         var(ε)= D=diag(σ2,σ2,...,σ2)
>
>理想的情况是，对于每个原始变量而言，其在因子载荷矩阵中，在一个公共因子上的载荷较大，在其他的因子上载荷较小。可以通过因子旋转方法调整因子载荷矩阵。

> 因子载荷阵的求解方法有极大似然估计法、主成分分析法、主因子法。这里仅介绍最为常用的主成分分析法，且不加证明地给出使用主成分分析法求解因子载荷阵的一般步骤：
>
> 1. 计算原始数据的协差阵Σ。
> 2. 计算协差阵Σ的特征根为λ1>λ2>·.·λp>0，相应的单位特征向量为u1，u2，···，up。
> 3. 利用Σ的特征根和特征向量计算因子载荷阵：
>
> ![image-20220806083621590](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806083621590.png)
>
> 由于因子分析的目的是减少变量个数，因此，因子数目m应小于原始变量个数p。所以在实际应用中，仅提取前m个特征根和对应的特征向量，构成仅包含m个因子的因子载荷阵：
>
> ![image-20220806083634944](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806083634944.png)

![image-20220806083717288](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806083717288.png)

>2．变量共同度
>
>设因子载荷矩阵为A，称第i行元素的平方和h~i~^2^＝Σa~ij~^2^      i=1,2,···,p
>
>为变量X~i~；的共同度。j=1由因子模型，知
>
>![image-20220806084504770](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806084504770.png)
>
>上式说明，变量X；的方差由两部分组成：
>
>第一部分为共同度h~i~^2^，它描述了全部公共因子对变量X~i~，的总方差所作的贡献，反映了变量X~i~；的方差中能够被全体因子解释的部分。
>
>第二部分为特殊因子ε~i~；对变量X~i~；的方差的贡献，也就是变量X~i~；的方差中没有被全体因子解释的部分。
>
>变量共同度越高，说明该因子分析模型的解释能力越高。

![image-20220806084548558](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806084548558.png)

![image-20220806084757491](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806084757491.png)

![image-20220806084849844](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806084849844.png)

![image-20220806084933057](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806084933057.png)

![image-20220806085129835](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806085129835-16597470915601.png)

![image-20220806085159024](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806085159024.png)

#### 1.6.2 因子分析代码部分

> matlab因子分析函数rotatefactors和factoran,
>
> lambda=factoran（X，m）：返回包含m个公共因子的因子模型的载荷阵lambda。 
>
> 输入参数X是n行d列的矩阵，每行对应一个观测，每列对应一个变量。
>
> m是一个正整数，表示模型中公共因子的个数。
>
>  输出参数lambda是一个d行m列的矩阵，第i行第j列元素表示第i个变量在第j个公共因子的载荷。
>
>  默认情况下，factoran函数调用用rotatefactors函数，并用＇varimax＇选项（rotatefactors函数的可用选项）来计算旋转后因子载荷阵的估计。
>
> ![image-20220806085900919](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806085900919.png)
>
> ![image-20220806090105831](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806090105831.png)
>
> ![image-20220806090157656](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806090157656.png)

**示例：**

![image-20220806090314693](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806090314693.png)

**样本数据**

![image-20220806090435178](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806090435178.png)

| 地区   | 人口密度（人/平方公里） | 人口均日生活用水量（升） | 供水普及率(%) | 燃气普及率 | 人均道路面积（平方米） | 排水管道 暗渠密度  （公里/ 平方公里） | 人均公园 绿地面积 （平方米） | 绿化覆盖率 （%） | 绿地率（%） |
| ------ | ----------------------- | ------------------------ | ------------- | ---------- | ---------------------- | ------------------------------------- | ---------------------------- | ---------------- | :---------: |
| 全国   | 4471                    | 82.81                    | 68.24         | 19.5       | 12.11                  | 3.57                                  | 1.08                         | 12.72            |    5.27     |
| 北京   | 6058                    | 67.56                    | 94.3          | 80.58      | 5.24                   | 4.69                                  | 0.44                         | 21.5             |    12.68    |
| 天津   | 3046                    | 84.03                    | 95.28         | 35.69      | 11.11                  | 2.03                                  | 0.02                         | 24.3             |    0.24     |
| 河北   | 4186                    | 63.9                     | 63.43         | 20.29      | 11.06                  | 1.51                                  | 0.4                          | 9.25             |    3.41     |
| 山西   | 4544                    | 63.7                     | 82.85         | 10.34      | 13                     | 3.46                                  | 1.1                          | 19.3             |    7.77     |
| 内蒙古 | 2871                    | 56.56                    | 50.36         | 10.66      | 10.29                  | 0.83                                  | 1.94                         | 7.64             |    3.45     |
| 辽宁   | 3782                    | 78.6                     | 49.19         | 10.23      | 14.33                  | 2.73                                  | 0.4                          | 9.27             |    2.02     |
| 吉林   | 3206                    | 75.36                    | 48.35         | 10.21      | 14.35                  | 1.34                                  | 0.35                         | 5.34             |    2.49     |
| 黑龙江 | 3119                    | 64.35                    | 73.76         | 10.86      | 21.48                  | 1.3                                   | 0.53                         | 5.64             |    2.48     |
| 上海   | 4345                    | 101.1                    | 99.02         | 99.02      | 20.22                  | 12.61                                 | 7.98                         | 37.25            |    28.29    |
| 江苏   | 5101                    | 111.81                   | 95.67         | 78.05      | 15.91                  | 7.57                                  | 4.32                         | 23.81            |    14.96    |
| 浙江   | 5431                    | 116.27                   | 80.36         | 44.19      | 14.15                  | 7.49                                  | 1.19                         | 12.18            |    6.76     |
| 安徽   | 4477                    | 94.63                    | 59.4          | 39.1       | 12.55                  | 4.9                                   | 3.55                         | 19.72            |    11.42    |
| 福建   | 6720                    | 108.1                    | 86.38         | 57.02      | 13.95                  | 7.23                                  | 7.21                         | 26.1             |    13.89    |
| 江西   | 4827                    | 106.6                    | 61.23         | 31.64      | 12.23                  | 6.25                                  | 1.24                         | 10.53            |    5.76     |
| 山东   | 4315                    | 71.92                    | 84.79         | 34.18      | 15.72                  | 6.51                                  | 1.4                          | 17.16            |    8.05     |
| 河南   | 5551                    | 73.84                    | 66.26         | 4.19       | 10.94                  | 4.32                                  | 1.01                         | 21.14            |    4.13     |
| 湖北   | 4125                    | 98.98                    | 76.98         | 27.28      | 10.1                   | 4.39                                  | 0.84                         | 10.88            |    5.25     |
| 湖南   | 4086                    | 102.59                   | 53.58         | 23.17      | 9.74                   | 3.4                                   | 0.48                         | 11.73            |    5.34     |
| 广东   | 3520                    | 151.43                   | 74.05         | 52.03      | 16.07                  | 9.04                                  | 1.49                         | 20.35            |     4.8     |

![image-20220806102814974](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806102814974.png)

![image-20220806104445559](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806104445559.png)

贡献情况F

![image-20220806111307905](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806111307905.png)

![image-20220806112604854](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806112604854.png)



## 2 预测与预报

### 2.1 灰色预测模型

#### 2.1.1 灰色预测模型原理

##### 2.1.1.1 灰色预测模型GM(1,1)

![image-20220809214524300](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809214524300.png)

![image-20220809214825690](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809214825690.png)

![image-20220809215032469](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809215032469.png)

![image-20220809215613051](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809215613051.png)

![image-20220809215922875](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809215922875.png)

![image-20220809220349047](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809220349047.png)

![image-20220809220934896](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809220934896.png)

![image-20220809221301306](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809221301306.png)

![image-20220810163400036](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810163400036.png)

![image-20220810163550846](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810163550846.png)

![image-20220810163938812](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810163938812.png)

![image-20220810164224645](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810164224645.png)

##### 2.1.1.2灰色预测模型GM(1,n)

![image-20220810165442334](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810165442334.png)

![image-20220810172450361](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810172450361.png)

##### 2.1.1.3灰色预测模型GM(2,1)

![image-20220810212317405](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810212317405.png)

![image-20220810212658542](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810212658542.png)

![image-20220810212717054](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810212717054.png)

![image-20220810213007124](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810213007124.png)

![image-20220810213219294](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810213219294.png)

![image-20220810213247346](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810213247346.png)

![image-20220810213543692](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220810213543692.png)

#### 2.1.2 灰色预测模型代码

```matlab
function gm11 = GM11_demo(X,td)
%Gm11_demo函数用来建立GM(1,1)模型，并预测
%输入X是非负的原始数据,t表示预测未来td数据
%输出gm11是一个结构体，输出各类参数

    %% 1、数据预处理
    n = length(X); %原始数据个数
    Ago = cumsum(X);  %一次累加
    Z = (Ago(1:n-1)+Ago(2:n))/2; %邻均值序列
    
    %% 2、构造B和Yn矩阵，并利用最小二乘求解a和u
    B = [-Z;ones(1,n-1)]';
    Yn = X(2:n)';
    sol = (B'*B)\(B'*Yn);  %最小二乘解
    a = sol(1);  %发展系数
    u = sol(2);  %灰色作用量
    
    %% 3、建立Gm(1,1)白化形式的一阶一元微分方程
    F = [X(1),(X(1)-u/a)*exp(-a*(1:n+td-1))+u/a];
    
    %% 4、数据的还原累减
    preData = [F(1),F(2:end)-F(1:end-1)];
    
    %% 5、可视化
    t = 1:n;
    plot(t,X,'ko--','MarkerFaceColor','k');%原始数据
    hold on
    grid on
    plot(t,preData(1:n),'b*-','LineWidth',1.5);
    plot(n:n+td,preData(n:n+td),'r*-','LineWidth',1.5);
    title('GM(1,1)——Original Vs Current And Futrue Predict')
    legend('Original Data','Predict Current Value',...
        'Predict Futrue Value','Location','best')
    legend('boxoff')
    xlabel('Time')
    ylabel('Value')
    
    %% 6、模型的检验
    Erorr = abs(X-preData(1:n)); %绝对残差
    RelE = Erorr./X; %相对误差
    RelMean = mean(RelE); %相对误差均值
    S1 = std(X,1); %原数据标准差
    S2 = std(Erorr(2:end),1); %误差的标准差
    C = S2/S1; %后验方差比
    P = sum(abs(Erorr-mean(Erorr))<0.6745*S1)/n;
    R_k= min(Erorr(2:end)+0.5*max(Erorr(2:end)))./...
        (Erorr(2:end)+0.5*max(Erorr(2:end)));
    R = sum(R_k/(n-1)); %关联度检验
    
    %% 7、组合参数输出
    gm11.Coeff_a = a;
    gm11.Coeff_u = u;
    gm11.Predict_Value = preData;
    gm11.Predict_Futrue_Value = preData(n+1:end);
    gm11.Abs_Erorr = Erorr;
    gm11.Rel_Erorr = RelE;
    gm11.Rel_Erorr_mean = RelMean;
    gm11.C = C;
    gm11.P = P;
    gm11.R = R;

end
```

```matlab
function gm1n = GM1N_demo(Y,X,X0,selcet)
%GM1N_demo函数用于灰色预测模型GM(1,n)，并预测
%输入Y是系统特征序列，X是相关因素，一行表示一个观测值
%输出结构体，用于输出参数

    %% 1、数据预处理
    n = length(Y); %原始数据个数
    Ago = cumsum(Y);  %一次累加
    if selcet == 1
        Z = (Ago(1:n-1)+Ago(2:n))/2; %邻均值序列
    else
        Z = (Ago(2:end)-Ago(1:end-1))./(log(Ago(2:end))-log(Ago(1:end-1)));
    end
    X1 = cumsum(X);
    
    %% 2、构造B和Yn矩阵，并利用最小二乘求解
    Yn = Y(2:end);
    B = [-Z,X1(2:end,:)];
    beta = (B'*B)\(B'*Yn); %最小二乘解
    a = beta(1);
    b = beta(2:end);
    
    %% 3、建立GM(1，n)的近似时间响应式
    F = zeros(n,1);
    F(1)=Y(1);
    for k = 2:n
        F(k) = (Y(1)-dot(b,X1(k,:))/a)*exp(-a*(k-1))+dot(b,X1(k,:))/a;
    end
     %数据还原
     if selcet ==1
         preData = [F(1);F(2:end)-F(1:end-1)]; 
    else
        %差分模拟
        diff_preData = zeros(n,1);
        diff_preData(1) = Y(1);
        for k = 2:n
            diff_preData(k) = -a*Z(k-1)+dot(b,X1(k,:));
        end
    end
    
    %% 4、数据预测值
    m = size(X0,1);
    Predict_Futrue_Value = zeros(m,1);
    for k = 1:m
        Predict_Futrue_Value(k) = (Y(1)-dot(b,X0(k,:))/a)*exp(-a*k)+dot(b,X0(k,:))/a;
    end
    %% 5、可视化
    t = 1:n;
    plot(t,Y,'ko--','MarkerFaceColor','k','LineWidth',1.5);%原始数据
    hold on;grid on;
    if selcet == 1
        plot(t,preData,'b*-','LineWidth',1.5);
        plot(m,Predict_Futrue_Value,'r*-','LineWidth',1.5);
    else
        plot(t,diff_preData,'b*-','LineWidth',1.5);
        plot(m,Predict_Futrue_Value,'r*-','LineWidth',1.5);
    end
    title('经典GM(1,1)——Original Vs Current And Futrue Predict')
    legend('Original Data','Predict Current Value',...
        'Predict Futrue Value','Location','best')
    legend('boxoff')
    xlabel('Time')
    ylabel('Value')
    
    %% 6、模型的检验
    if selcet == 1
        Erorr = abs(Y-preData); %绝对残差
    else
        Erorr = abs(Y-diff_preData); %绝对残差
    end
    RelE = Erorr./Y; %相对误差
    RelMean = mean(RelE); %相对误差均值
    S1 = std(Y,1); %原数据标准差
    S2 = std(Erorr(2:end),1); %误差的标准差
    C = S2/S1; %后验方差比
    P = 0; %小概率误差
    for i = 1:n
        if (abs(Erorr(i)-mean(Erorr(2:end)))<0.6745*S1)
            P = P+1;
        end
    end
    P = P/n;
    R_k= min(Erorr(2:end)+0.5*max(Erorr(2:end)))./...
        (Erorr(2:end)+0.5*max(Erorr(2:end)));
    R = sum(R_k)/(length(R_k)-1); %关联度检验
    
    %% 8、组合参数输出
    if selcet == 1
        gm1n.text = '经典GM(1,n)模型';
        gm1n.Predict_Value = preData;
    else
        gm1n.text2 = '差分模拟GM(1,n)模型';
        gm1n.Predict_Value = diff_preData;
    end 
    gm1n.Coeff_a = a;
    gm1n.Coeff_b = b;
    gm1n.Predict_Futrue_Value = Predict_Futrue_Value;
    gm1n.Abs_Erorr = Erorr;
    gm1n.Rel_Erorr = RelE;
    gm1n.Rel_Erorr_mean = RelMean;
    gm1n.C = C;
    gm1n.P = P;
    gm1n.R = R;
end
```

```matlab
function gmvh = GM_Verhulst_demo(X,pre_num)
%GMVerhulst_demo函数用于建立GM Verhulst模型，并进行预测

    %% 1、数据的预处理
    n = length(X);
    H = diff(X); %累减形式
    Z = (X(2:end)+X(1:end-1))/2; %对原数据构造邻均值序列
    B = [-Z;Z.^2]';
    Yn = H';
    
    %% 2、最小二乘法求解
    u = (B'*B)\(B'*Yn);
    a = u(1);
    b = u(2);
    
    %% 3、构造时间相应序列，累减还原
    F = [X(1),a*X(1)./(b*X(1)+(a-b*X(1))*exp(a*(1:n+pre_num-1)))]; 
    preData = F(1:n);
    FData = F(n+1:end); %%未来预测数据
    %% 5、可视化
    t = 1:n;
    plot(t,X,'ko--','MarkerFaceColor','k','LineWidth',1.5);%原始数据
    hold on;grid on;
    plot(t,preData,'b*-','LineWidth',1.5);
    plot(n+1:n+pre_num,FData,'r*-','LineWidth',1.5);
    title('GM(1,1)——Original Vs Current And Futrue Predict')
    legend('OriginalData','ForecastCurrentValue','ForecastFutrueValue','Location','best')
    legend('boxoff')
    xlabel('Time')
    ylabel('Value')
    
    %% 6、模型的检验
    Erorr = abs(X-preData); %绝对残差
    RelE = Erorr./X; %相对误差
    RelMean = mean(RelE); %相对误差均值
    S1 = std(X,1); %原数据标准差
    S2 = std(Erorr(2:end),1); %误差的标准差
    C = S2/S1; %后验方差比
    P = 0; %小概率误差
    for i = 1:n
        if (abs(Erorr(i)-mean(Erorr(2:end)))<0.6745*S1)
            P = P+1;
        end
    end
    P = P/n;
    R_k= (min(Erorr(2:end))+0.5*max(Erorr))./...
        (Erorr(2:end)+0.5*max(Erorr));
    R = sum(R_k)/(length(R_k)-1); %关联度检验
    
    %% 8、组合参数输出
    gmvh.Coeff_a = a;
    gmvh.Coeff_b = b;
    gmvh.Predict_Value = preData;
    gmvh.Predict_Futrue_Value = FData;
    gmvh.Abs_Erorr = Erorr;
    gmvh.Rel_Erorr = RelE;
    gmvh.Rel_Erorr_mean = RelMean;
    gmvh.C = C;
    gmvh.P = P;
    gmvh.R = R;
end
```



### 2.2 微分方程预测





### 2.3 回归分析预测

![image-20220812121557831](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812121557831.png)

![image-20220812121630312](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812121630312.png)

![image-20220812121818934](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812121818934.png)

![image-20220812121958918](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812121958918.png)

![image-20220812122105899](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812122105899.png)

![image-20220812122301570](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812122301570.png)

![image-20220812122334791](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812122334791.png)

![image-20220812122356793](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812122356793.png)

![image-20220812122430657](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812122430657.png)

![image-20220812122454523](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812122454523.png)

![image-20220812122510219](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812122510219.png)

![image-20220812122613459](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812122613459.png)

![image-20220812122709860](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812122709860.png)

![image-20220812122722064](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812122722064.png)

![image-20220812122858482](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812122858482.png)

![image-20220812190931371](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812190931371.png)

正态性检验：

![image-20220812191103583](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812191103583.png)

![image-20220812191325078](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812191325078.png)

![image-20220812191534663](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812191534663.png)

![image-20220812193304350](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812193304350.png)

![image-20220812193402301](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812193402301.png)





### 2.4 马尔科夫预测

#### 2.4.1 马尔科夫链

**随机过程**

![image-20220812101455746](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812101455746.png)

![image-20220812101554411](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812101554411.png)

后无效性：系统的概率与未来的概率无关

![image-20220812101911464](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812101911464.png)

![image-20220812102710493](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812102710493.png)

![image-20220812102845044](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812102845044.png)

![image-20220812103156485](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812103156485.png)

![image-20220812103208039](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812103208039.png)

![image-20220812103302336](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812103302336.png)

![image-20220812103352063](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812103352063.png)

![image-20220812103412202](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812103412202.png)

![image-20220812104531043](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812104531043.png)

![image-20220812104624158](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812104624158.png)

![image-20220812104740941](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812104740941.png)

![image-20220812105003081](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812105003081.png)

![image-20220812105023169](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812105023169.png)

![image-20220812105045884](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812105045884.png)

![image-20220812105123081](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812105123081.png)

![image-20220812105422863](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812105422863.png)

![image-20220812105448988](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812105448988.png)



### 2.5 时间序列预测

![image-20220812114554706](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812114554706.png)

![image-20220812114633393](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812114633393.png)

![image-20220812114905740](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812114905740.png)

![image-20220812114925568](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812114925568.png)

![image-20220812115156929](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812115156929.png)

![image-20220812115234271](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812115234271.png)

![image-20220812115447089](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812115447089.png)

![image-20220812115531418](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812115531418.png)

![image-20220812115556637](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812115556637.png)

![image-20220812115624536](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812115624536.png)

![image-20220812115754938](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812115754938.png)

![image-20220812115818454](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812115818454-16602767003871.png)

![image-20220812115844963](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812115844963.png)

![image-20220812115928558](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812115928558.png)

![image-20220812115952011](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812115952011.png)

![image-20220812120119726](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812120119726.png)

![image-20220812120207980](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812120207980.png)

![image-20220812120317023](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812120317023.png)

![image-20220812120336929](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812120336929.png)

![image-20220812120403294](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812120403294.png)

![image-20220812120505024](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812120505024.png)

![image-20220812120543423](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812120543423.png)

![image-20220812194711353](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812194711353.png)

![image-20220812194830956](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812194830956.png)

![image-20220812194902185](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812194902185.png)

![image-20220812195057192](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812195057192.png)

![image-20220812195246905](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812195246905.png)

![image-20220812195355316](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812195355316.png)

![image-20220812195454777](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812195454777.png)

![image-20220812195520207](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812195520207.png)

#### ARIMA预测模型

![image-20220812195742136](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812195742136.png)

### 2.6  移动平均与指数平滑法

#### 2.6.1 移动平均法原理

![image-20220812200207986](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812200207986.png)

![image-20220812203932925](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812203932925.png)

![image-20220812204105228](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812204105228.png)

![image-20220812204624101](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220812204624101.png)

#### 2.6.2 移动平均法代码

```matlab
function MA = MovAvg_method(X,N,preT)
%MovAvg_method实现一次移动平均和二次移动平均，并预测未来
%输入参数X是时间序列，N是移动平均步长，preT未来预测期数
%MA是结构体用于输出参数
    %% 1、数据预处理
    X = rmmissing(X); %删除序列中缺失值或NAN值
    n = length(X); %数据量
    m = length(N); %移动步长向量
    
    %% 2、一次移动平均
    Yhat1 = cell(1,m);
    Err1 = zeros(1,m);
    RelMeanErr1 = zeros(1,m);
    for i = 1:m  %针对每一个步长计算一次平均值
        for j = 1:n-N(i)+1  %移动平均循环计算下标索引
            Yhat1{i}(j) = mean(X(j:j+N(i)-1));  %一次平均移动
        end
        Err1(i) = sqrt(mean(Yhat1{i}-X(N(i):end)).^2); %均方根值
        RelMeanErr1(i) = mean((Yhat1{i}-X(N(i):end))./X(N(i):end));
    end
    
    %% 3、二次移动平均
    Yhat2 = cell(1,m);
    Err2 = zeros(1,m);
    RelMeanErr2 = zeros(1,m);
    at = cell(1,m);
    bt = cell(1,m);
    for i = 1:m  %针对每一个步长计算一次平均值
        ni = length(Yhat1{i});
        for j = 1:ni-N(i)+1  %移动平均循环计算下标索引
            Yhat2{i}(j) = mean(Yhat1{i}(j:j+N(i)-1));  %一次平均移动
            at{i}(j) = 2*Yhat1{i}(j+N(i)-1) - Yhat2{i}(j);
            bt{i}(j) = 2/(N(i)-1)*(Yhat1{i}(j+N(i)-1) - Yhat2{i}(j));
        end
        Err2(i) = sqrt(mean(Yhat2{i}-X(2*N(i)-1:end)).^2); %均方根值
        RelMeanErr2(i) = mean((Yhat2{i}-X(2*N(i)-1:end))./X(2*N(i)-1:end));
    end
    
    %% 4、未来期预测
    Ypre = cell(1,m);
    for i = 1:m  %针对移动步长计算一次移动平均
        for t = 1:preT  %每一次计算一个未来期值
            Ypre{i}(t) = at{i}(end)+bt{i}(end)*t;
        end
    end
    
    %% 5、可视化
    t =1:n;
    plot(t,X);
    hold on
    leg = cell(1,m+1);
    leg{1} = 'Original Data';
    for i = 1:m
        plot(N(i):n,Yhat1{i})
        leg{i+1} = strcat('MA_N =',num2str(N(i)));
    end
    legend(leg,'Location','best')
    legend('boxoff')
    xlabel('Time');ylabel('Value')
    title('一次移动平均法')
    
    t =1:n;
    plot(t,X);
    hold on
    leg = cell(1,m+1);
    leg{1} = 'Original Data';
    for i = 1:m
        plot(N(i):n,Yhat1{i})
        leg{i+1} = strcat('MA_N =',num2str(N(i)));
    end
    legend(leg,'Location','best')
    legend('boxoff')
    xlabel('Time');ylabel('Value')
    title('一次移动平均法')
    
    figure
    plot(t,X);
    hold on
    leg = cell(1,m+1);
    leg{1} = 'Original Data';
    for i = 1:m
        plot(2*N(i)-1:n,Yhat2{i})
        leg{i+1} = strcat('MA_N =',num2str(N(i)));
    end
    legend(leg,'Location','best')
    legend('boxon')
    xlabel('Time');ylabel('Value')
    title('二次移动平均法')
    %% 6、参数输出
    MA.Yhat1 = Yhat1;
    MA.Err1 = Err1;
    MA.RelMeanErr1 = RelMeanErr1;
    MA.Yhat2 = Yhat2;
    MA.at = at;
    MA.bt = bt;
    MA.Err2 = Err2;
    MA.RelMeanErr2 = RelMeanErr2;
    MA.Ypre = Ypre;
end
```



### 2.7 小波分析预测





### 2.8 神经网络预测





### 2.9 混沌序列预测





## 3 分类与判别

### 3.1 判别分析——距离判别

#### 3.1.1 马氏距离

**定义：**

![image-20220805195907041](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220805195907041.png)

![image-20220805200236542](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220805200236542.png)

![image-20220805200313365](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220805200313365.png)

#### 3.1.1 多个总体协方差矩阵相等检验

![image-20220805205800074](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220805205800074.png)

**代码：**

```matlab
function covE = COVA_equaldemo(G)
%COVA_equaldemo用来判别多个总体协方差矩阵是否相等
%G是元胞数组
    k = size(G,2);  %总体数，类别数
    n = zeros(1,k);
    for i = 1:k
        n(i) = size(G{i},1);  %每个总体样本数
    end
    sn = sum(n);  %总体样本量
    
    %% 基本统计的计算问题
    fn = size(G{1},2);   %指标数，对应公式 P
    def = fn*(fn+1)*(k-1)/2;   %自由度
    if length(unique(n)) == 1  %每个总体样本相等
        d = (2*fn^2+3*fn-1)*(k+1)/(6*(fn+1)*(sn-k));
    else %每个总体样本量不全等
        d = (2*fn^2+3*fn-1)/(6*(fn+1)*(k-1))*(sum(1./(n-1))-1/(sn-k));
    end
    Gm = cell(k,1);
    Gcov = cell(k,1);
    for i = 1:k
        Gm{i}=mean(G{i}); %每个总体的均值
        Gcov{i} = cov(G{i});  %每个总体的协方差矩阵
    end
    S = 0;
    sm = 0;
    M = 0;
    for i = 1:k
        S = S+(n(i)-1)*Gcov{i}/(sn-k);
        sm = sm + (n(i)-1)*log(det(Gcov{i}));
    end
    M = (sn-k)*log(det(S))-sm;
    T = (1-d)*M;  %检验方差矩阵相等统计量
    cv = chi2inv(0.95,def); %统计量临界值
    
    %%组合结构体输出
    covE.Class_num = k;
    covE.Sample_Num = n;
    covE.G = G;
    covE.Vars_NUm = fn;
    covE.Gm = Gm;
    covE.VGcov = Gcov;
    covE.S = S;
    
    if T < cv %协方差矩阵相等
       covE.equal = 1;
       covE.cov_equal = '协方差矩阵相等';
    else %协方差矩阵不相等
        covE.equal = 0;
       covE.cov_equal = '协方差矩阵不全相等';
    end
end
```

#### 3.1.2 多个总体的距离判别

![image-20220805222543199](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220805222543199.png)

**代码：**

```matlab
function [covE,dda] = DDA_dem(X,xtest)
    %X是总体，是一个元胞数组，每一个元胞数组对应一个总体，rtest是待判别样本
    covE = COVA_equaldemo(X);  %多个总体协方差矩阵判断
    Gn = covE.Class_num;  %总体数
    S = covE.S;  %所有总体样本的协方差
    mG = zeros(Gn,covE.Vars_Num);
    for i = 1:Gn
        mG(i,:) = covE.Gm{i}; %把所有总体的均值构成一个矩阵，每一行代表一个总体
    end
    sample_n =size(xtest,1); %待判样本数
    if covE.equal == 0  %多个总体协方差矩阵不全相等
        W = zeros(1,Gn);
        class = zeros(1,sample_n);
        for i = 1:sample_n
            for j = 1:Gn  %总体数
                W(j) = (xtest(i,:)-mG(j,:))*inv(S)*(xtest(i,:)-mG(j,:))';
            end
            [~,ind] = min(W); %求最小距离的索引
            class(i) = ind;  %判别样本的类别
            dda.DistenceG{i} = W;
        end
    elseif covE.equal == 1  %多个总体协方差矩阵相等
       class = zeros(1,sample_n);
        for i = 1:sample_n
            W = zeros(Gn,Gn);
            for j = 1:Gn
                for k = 1:Gn
                    W(j,k) = (xtest(i,:)-(mG(j,:)+mG(k,:))/2)*...
                        inv(S)*(mG(j,:)-mG(k,:))';
                    if W(j,k) < 0 %负值不考虑
                        sign = 0;
                        break;
                    else
                        sign = 1;
                    end
                end
                if sign == 1  %如果k循环完毕，意味着其他总体的距离及算出，且大于0
                   class(i) = j;
                   break;
                end
            end
            dda.DistenceG{i} = W;
        end
    end
    dda.Class = class;
end
```

最后的dda.Class类别值代表G1，G2...总体，既如果dda.Class=[1 3 2],则从上到下分别属于G1,G3,G2,

#### 3.1.3 判别准则的评价

![image-20220806143357433](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806143357433.png)

![image-20220806143425682](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806143425682.png)

**代码：**

```matlab
function daer = DAErrorRate(covE)
    %实现距离判别分析的回代误判率和交叉误判率
    %dear、covE输出输入结构体
    n = sum(covE.Sample_Num);  %总的样本量
    Gc = covE.Class_num; %总体数，类别数
    Gn = covE.Sample_Num; %每个总体的样本数，一个向量
    %% 1、回代误判
    k = 1;
    classReturn = zeros(n,1);
    for  i = 1:Gc
        Gi = covE.G{i};
        for j = 1:Gn(i)
            [~,dda] = DDA_dem(covE.G,Gi(j,:));
            classReturn(k) = dda.Class;  %所属类别
            k = k+1;
        end
    end
   %% 2、交叉误判
   k = 1;
   classCorss = zeros(n,1);
   for i = 1:Gc
       Gi = covE.G{i};  %取第i个总体
       for j = 1:Gn(i)
           covE.G{i} = Gi([1:i-1,i+1:Gn(i)],:);  %跳过第i个元素
           [~,dda] = DDA_dem(covE.G,Gi(j,:));
           classCorss(k) = dda.Class;  %所属类别
           k = k+1;
       end
   end
   %% 3、计算回代误判率和交叉误判率
   Gn= [0,covE.Sample_Num];
   resR = zeros(1,Gc);
   resC = zeros(1,Gc);
   for k = 1:Gc
       sind = sum(Gn(1:k))+1;       %类别中起始行
       eind = sum(Gn(2:k+1));  %类别中终止行
       resR(k) = sum(classReturn(sind:eind) == k);
       resC(k) = sum(classCorss(sind:eind) == k);
   end
   ErrorReturnRate = 1-sum(resR)/n;
   ErrorCorssRate = 1-sum(resC)/n;
   daer.classRtrun = classReturn;
   daer.ErrorReturnRate = ErrorReturnRate;
   daer.classGorss = classCorss;
   daer.ErrorCorssRate = ErrorCorssRate;
end
```

#### 3.1.4 matlab自带距离判别函数

![image-20220806154203496](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806154203496.png)

![image-20220806154329592](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806154329592.png)

![image-20220806154631304](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806154631304.png)

### 3.2 判别分析——贝叶斯判别

#### 3.2.1 贝叶斯判别模型

![image-20220806163008188](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806163008188.png)

![image-20220806163449928](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806163449928.png)

![image-20220806163834651](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806163834651.png)

![image-20220806164718781](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806164718781.png)

![image-20220806164904706](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806164904706.png)

![image-20220806164957997](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806164957997.png)

![image-20220806165105399](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806165105399.png)

![image-20220806181951313](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806181951313.png)

![image-20220806182051517](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806182051517.png)

#### 3.2.2 贝叶斯判别代码

```matlab
function covE = COVA_equaldemo(G)
%COVA_equaldemo用来判别多个总体协方差矩阵是否相等
%G是元胞数组
    k = size(G,2);  %总体数，类别数
    n = zeros(1,k);
    for i = 1:k
        n(i) = size(G{i},1);  %每个总体样本数
    end
    sn = sum(n);  %总体样本量
    
    %% 基本统计的计算问题
    fn = size(G{1},2);   %指标数，对应公式 P
    def = fn*(fn+1)*(k-1)/2;   %自由度
    if length(unique(n)) == 1  %每个总体样本相等
        d = (2*fn^2+3*fn-1)*(k+1)/(6*(fn+1)*(sn-k));
    else %每个总体样本量不全等
        d = (2*fn^2+3*fn-1)/(6*(fn+1)*(k-1))*(sum(1./(n-1))-1/(sn-k));
    end
    prior_p = n./sn; %先验概率
    Gm = cell(k,1);
    Gcov = cell(k,1);
    for i = 1:k
        Gm{i}=mean(G{i}); %每个总体的均值
        Gcov{i} = cov(G{i});  %每个总体的协方差矩阵
    end
    S = 0;
    sm = 0;
    M = 0;
    for i = 1:k
        S = S+(n(i)-1)*Gcov{i}/(sn-k);
        sm = sm + (n(i)-1)*log(det(Gcov{i}));
    end
    M = (sn-k)*log(det(S))-sm;
    T = (1-d)*M;  %检验方差矩阵相等统计量
    cv = chi2inv(0.95,def); %统计量临界值
    
    %%组合结构体输出
    covE.Class_num = k;
    covE.Sample_Num = n;
    covE.prior_p = prior_p;
    covE.G = G;
    covE.Vars_Num = fn;
    covE.Gm = Gm;
    covE.VGcov = Gcov;
    covE.S = S;
    
    if T < cv %协方差矩阵相等
       covE.equal = 1;
       covE.cov_equal = '协方差矩阵相等';
    else %协方差矩阵不相等
        covE.equal = 0;
       covE.cov_equal = '协方差矩阵不全相等';
    end
end
```



```matlab
function [covE,dda] = MBayes_DA(X,xtest)
    %MBayes_DA本也是判别法
    %X是总体，是一个元胞数组，每一个元胞数组对应一个总体，rtest是待判别样本
    covE = COVA_equaldemo(X);  %多个总体协方差矩阵判断
    Gn = covE.Class_num;  %总体数
    S = covE.S;  %所有总体样本的协方差
    p = covE.prior_p; %各个总体的先验概率
    mG = zeros(Gn,covE.Vars_Num);
    for i = 1:Gn
        mG(i,:) = covE.Gm{i}; %把所有总体的均值构成一个矩阵，每一行代表一个总体
    end
    sample_n =size(xtest,1); %待判样本数
    class = zeros(1,sample_n);
    DistanceG = cell(1,sample_n);
    %% 2、根据总体判别公式，计算贝叶斯函数并进行判别分析
    if covE.equal == 0  %多个总体协方差矩阵不全相等
        for i = 1:sample_n
            JW = zeros(Gn,1);
            for j = 1:Gn  %总体数
                Sj = covE.VGcov{j}; %取第j个总体协方差矩阵
                JW(j) = (xtest(i,:)-mG(j,:))*(Sj\(xtest(i,:)-mG(j,:))')+...
                    log(det(Sj))-2*log(p(j));
                [~,ind] = min(JW); %求最小距离的索引
                class(i) = ind;  %判别样本的类别
            end
            DistanceG{i} = JW;
        end
        %(2)各总体协方差相等
    elseif covE.equal == 1  %多个总体协方差矩阵相等
        JW = zeros(Gn,1);
        for i = 1:sample_n
            for j = 1:Gn
                %计算待判样本每两个总体之间的距离
                JW(j) = mG(j,:)*(S\xtest(i,:)')-1/2*mG(j,:)*(S\mG(j,:)')...
                    +log(p(j));
            end
             [~,ind] = min(JW); %求最小距离的索引
            class(i) = ind;  %判别样本的类别
            DistanceG{i} = JW;
        end
    end
    
    %% 3、输出结构体组合
    dda.DistanceG = DistanceG;
    dda.Class = class;
end
```

**误判**

```matlab
function daer = MBayes_DRule(covE)
    %实现距离判别分析的回代误判率和交叉误判率
    %dear、covE输出输入结构体
    n = sum(covE.Sample_Num);  %总的样本量
    Gc = covE.Class_num; %总体数，类别数
    Gn = covE.Sample_Num; %每个总体的样本数，一个向量
    %% 1、回代误判
    k = 1;
    classReturn = zeros(n,1);
    for  i = 1:Gc
        Gi = covE.G{i};
        for j = 1:Gn(i)
            [~,dda] = MBayes_DA(covE.G,Gi(j,:));
            classReturn(k) = dda.Class;  %所属类别
            k = k+1;
        end
    end
   %% 2、交叉误判
   k = 1;
   classCorss = zeros(n,1);
   for i = 1:Gc
       Gi = covE.G{i};  %取第i个总体
       for j = 1:Gn(i)
           covE.G{i} = Gi([1:i-1,i+1:Gn(i)],:);  %跳过第i个元素
           [~,dda] = MBayes_DA(covE.G,Gi(j,:));
           classCorss(k) = dda.Class;  %所属类别
           k = k+1;
       end
   end
   %% 3、计算回代误判率和交叉误判率
   Gn= [0,covE.Sample_Num];
   resR = zeros(1,Gc);
   resC = zeros(1,Gc);
   for k = 1:Gc
       sind = sum(Gn(1:k))+1;       %类别中起始行
       eind = sum(Gn(2:k+1));  %类别中终止行
       resR(k) = sum(classReturn(sind:eind) == k);
       resC(k) = sum(classCorss(sind:eind) == k);
   end
   ErrorReturnRate = 1-sum(resR)/n;
   ErrorCorssRate = 1-sum(resC)/n;
   daer.classRtrun = classReturn;
   daer.ErrorReturnRate = ErrorReturnRate;
   daer.classGorss = classCorss;
   daer.ErrorCorssRate = ErrorCorssRate;
end
```

**测试数据：**

|       |    G1    |      |      |
| :---: | :------: | :--: | :--: |
| -0.45 |  -0.41   | 1.09 | 0.45 |
| -0.56 |  -0.31   | 1.51 | 0.16 |
| 0.06  |   0.02   | 1.01 | 0.4  |
| -0.07 |  -0.09   | 1.45 | 0.26 |
| -0.1  |  -0.09   | 1.56 | 0.67 |
| -0.14 |  -0.07   | 0.71 | 0.28 |
| 0.04  |   0.01   | 1.5  | 0.71 |
| -0.06 |  -0.06   | 1.37 | 0.4  |
| -0.13 |  -0.14   | 1.42 | 0.44 |
|       |    G2    |      |      |
| 0.51  |   0.1    | 2.49 | 0.54 |
| 0.08  |   0.02   | 2.01 | 0.53 |
| 0.38  |   0.11   | 3.27 | 0.35 |
| 0.19  |   0.05   | 2.25 | 0.33 |
| 0.32  |   0.07   | 4.24 | 0.63 |
| 0.12  |   0.05   | 2.52 | 0.69 |
| -0.02 |   0.02   | 2.05 | 0.35 |
| 0.22  |   0.08   | 2.35 | 0.4  |
| 0.17  |   0.07   | 1.8  | 0.52 |
|       | 待判数据 |      |      |
| -0.23 |   -0.3   | 0.33 | 0.18 |
| 0.15  |   0.05   | 2.17 | 0.55 |
| -0.28 |  -0.23   | 1.19 | 0.66 |
| 0.48  |   0.09   | 1.24 | 0.18 |



#### 3.2.3 MATLAB自带函数

![image-20220806211252536](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220806211252536.png)

### 3.3 判别分析——Fisher判别分析

#### 3.3.1 Fisher判别分析原理

![image-20220807090926224](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807090926224.png)

![image-20220807091205101](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807091205101.png)

![image-20220807091333526](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807091333526.png)

![image-20220807091611139](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807091611139.png)

![image-20220807092525578](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807092525578.png)

![image-20220807092915543](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807092915543.png)

![image-20220807094003884](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807094003884.png)

![image-20220807093937683](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807093937683.png)

![image-20220807094701368](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807094701368.png)

![image-20220807094900236](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807094900236.png)

#### 3.3.2 Fisher判别分析代码部分

```matlab
function fdam = Fisher_DAW(data)
    %Fisher_DAW实现费希尔判别分析,针对二分类
    %输入参数data是样本数据、输出参数fdam是一个结构体
    
    %% 类别判断
    if length(unique(data(:,end))) ~= 2
        warning('Fisher_DAW函数仅适用于二分类！');
        fdam = [];
        return
    end
    
    %% 1、数据预处理
    sn = size(data,1);  %总的样本量
    %抽取80%的样本数据做训练样本，20%测验
    data = data(randperm(sn),:);  %使样本混乱，便于随机抽样
    n8 = round(sn*0.8);
    xtrain = data(1:n8,1:end-1);  %抽取80%训练样本
    trainLabel = data(1:n8,end); %对应训练样本的标签值
    xtest = data(n8+1:end,1:end-1); %提取20%的测试样本
    testLabel = data(n8+1:end,end); %对应测试样本的标签值
    
    %% 2、计算每个总体的各种参数（均值、类内离散度、类间离散度）
    cn = length(unique(data(:,end))); %得到类别数，即总体数
    Sinner = cell(cn,1);
    train = cell(cn,1);
    m = zeros(cn,size(xtrain,2));
    Ni = zeros(cn,1);
    for i = 1:cn
       %分离类内总体样本数据
       train{i} = xtrain(trainLabel == i,:);
       Ni(i)=size(train{i},1);  %每个总体的样本量
       m(i,:) = mean(train{i});  %类内均值向量
       Sinner{i} = (train{i}-m(i,:))'*(train{i}-m(i,:));  %类内离散度矩阵
    end
    Sinnersum = Sinner{1}+Sinner{2}; %总类内离散度矩阵
    %最佳投影方向向量
    W = (Sinnersum\(m(1,:)-m(2,:))')';
    %y0 = -0.5*W*(m(1,:)*m(2,:)); %阈值
    y0 = -W*(Ni(1)*m(1,:)+Ni(2)*m(2,:))'/sum(Ni);
    
    %% 3、20%的样本量
    nt = size(xtest,1);  %测试样本量
    newLabel = zeros(nt,1);  %测试样本类别标签
    preYVal = zeros(nt,1);
    for i = 1:nt
        y = W*xtest(i,:)' + y0; %投影后的值+阈值
        if y > 0
            newLabel(i) = 1;
        elseif y < 0
            newLabel(i) = 2;
        end
        preYVal(i) = y; %投影后的值+阈值
    end
     %% 4、构造混淆矩阵
     confusionMat = zeros(cn,cn);
     for i = 1:nt
         for j = 1:cn
             if newLabel(i) == 1 && testLabel(i) ==j
                 confusionMat(1,j) = confusionMat(1,j)+1;
             elseif newLabel(i) == 2 && testLabel(i) ==j
                 confusionMat(2,j) = confusionMat(2,j)+1;
             end
         end
     end
     
     %%参数输出
     fdam.M = m;
     fdam.Sinner = Sinner;
     fdam.Sinnersum = Sinnersum;
     fdam.G_Num = Ni;
     fdam.W = W;
     fdam.y0 = y0;
     fdam.prer_Val = preYVal;
     fdam.TrueAndPre_Label = [testLabel,newLabel];
     fdam.Test_Num = nt;
     fdam.Correct_Num = sum(diag(confusionMat));
     fdam.confusionMat = confusionMat;
     fdam.acc = sum(diag(confusionMat))/nt*100;
end
```

```matlab
function fmcw = Fisher_MultCW(data)
    %Fisher_DAW实现费希尔判别分析,针对二分类
    %输入参数data是样本数据、输出参数fdam是一个结构体

    %% 1、数据预处理
    sn = size(data,1);  %总的样本量
    %抽取80%的样本数据做训练样本，20%测验
    data = data(randperm(sn),:);  %使样本混乱，便于随机抽样
    n8 = round(sn*0.8);
    xtrain = data(1:n8,1:end-1);  %抽取80%训练样本
    trainLabel = data(1:n8,end); %对应训练样本的标签值
    xtest = data(n8+1:end,1:end-1); %提取20%的测试样本
    testLabel = data(n8+1:end,end); %对应测试样本的标签值
    
    %% 2、计算每个总体的各种参数（均值、类内离散度、类间离散度）
    cn = length(unique(data(:,end))); %得到类别数，即总体数
    Sinner = cell(cn,1);
    train = cell(cn,1);
    m = zeros(cn,size(xtrain,2));
    Ni = zeros(cn,1);
    vars = size(xtrain,2); 
    Sw = zeros(vars,vars);  %总类内离散度矩阵
    for i = 1:cn
       %分离类内总体样本数据
       train{i} = xtrain(trainLabel == i,:);
       Ni(i)=size(train{i},1);  %每个总体的样本量
       m(i,:) = mean(train{i});  %类内均值向量
       Sinner{i} = (train{i}-m(i,:))'*(train{i}-m(i,:));  %类内离散度矩阵
       Sw = Sw+Ni(i)*Sinner{i};
    end
    Sw = Sw/sn; %总类内离散度矩阵
    Sb = zeros(vars,vars);  %总类内间离散度矩阵
    M = mean(m);  %总类内均值
    for i = 1:cn
        Sb = Sb+Ni(i)*(M-m(i,:))'*(M-m(i,:));
    end
    Sb = Sb/sn; %总类间离散度矩阵
    
    %% 最大特征值对应的特征向量
    A = Sw\Sb;
    [V,D] = eig(A);
    [~,ind] = max(diag(D));
    W = V(:,ind)';  %最大特征值W
    
    %% 训练样本投影降维
    mY = zeros(1,cn);
    for i = 1:cn
        Y = W*train{i}'; %每个总体训练样本投影值
        mY(i) = mean(Y);  %每个总体投影后的类内中心
    end
    
    %% 20%样本进行判别
    nt = size(xtest,1); %测试样本量
    newLabel = zeros(nt,1);
    for i = 1:nt
        ytest = W*xtest(i,:)'; %每一次取一个测试样本
        y0 = mean(M);  %阈值的初始化
        for j = 1:cn
            if abs(ytest-mY(j)) < y0
                newLabel(i) = j;
                y0 = abs(ytest-mY(j));  %阈值更新
            end
        end
    end
    %% 构造混淆矩阵
    confusionMat = zeros(cn);
    for i = 1:nt
        c = testLabel(i);
        nl = newLabel(i);
        if c == nl  %判断正确
            confusionMat(c,c) = confusionMat(c,c)+1;
        else  %判断错误
            confusionMat(c,nl) = confusionMat(c,nl)+1;
        end
    end
    
    %%参数输出
    fmcw.G_NUm = Ni;
    fmcw.MeanInner = m;
    fmcw.MeanTotal = M;
    fmcw.Sinner = Sinner;
    fmcw.Sw_innerOut = Sw;
    fmcw.Sb_OutMat = Sb;
    fmcw.W = W;
    fmcw.mY = mY;
    fmcw.Test_Num = nt;
    fmcw.PerCorrect_Num = sum(diag(confusionMat));
    fmcw.TrueAndTest_Label = [testLabel,newLabel];
    fmcw.confusionMat = confusionMat;
    fmcw.predict_accuracy = sum(diag(confusionMat))/nt*100;
            
end
```

### 3.4 聚类分析——谱系聚类

#### 3.4.1 聚类常用距离

![image-20220807201008666](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807201008666.png)

![image-20220807203240290](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807203240290.png)

![image-20220807203608356](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807203608356.png)

![image-20220807203736589](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807203736589.png)

![image-20220807203856773](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807203856773.png)

![image-20220807204427185](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807204427185.png)

![image-20220807215436673](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220807215436673.png)

#### 3.4.2 类间距离

![image-20220808082559306](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220808082559306.png)

![image-20220808082922368](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220808082922368.png)

#### 3.4.3 谱系聚类

![image-20220808083307810](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220808083307810.png)

![image-20220808083515344](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220808083515344.png)

![image-20220808084019179](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220808084019179.png)

![image-20220808090017104](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220808090017104.png)

**示例：**

![image-20220808092931621](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220808092931621.png)



**代码：**

```matlab
clear;
clc
[data,text]=xlsread('测试数据人民生活水平.xlsx');
data = [data(1:end,1:4);data(1:14,7:10)];
text = [text(2:16,1);text(2:15,7)];
sale = zscore(data);
D = pdist(sale);
% 编号换成字符串
 z = linkage(D,'average');  %类平均距离
labels = text;
dendrogram(z,'labels',labels)
figure
dendrogram(z,'labels',labels,'Orientation','left','ColorThreshold',0.5*max(z(:,3)))
figure
dendrogram(z,'labels',labels,'Orientation','left','ColorThreshold','default')
T = cluster(z,5);
res = cell(29,2);
res(:,1) = labels;
res(:,2) = num2cell(T);
res = sortrows(res,2);
disp(res)
s1 = sale(T==1,:);
s2 = sale(T==2,:);
s3 = sale(T==3,:);
s4 = sale(T==4,:);
s5 = sale(T==5,:);
subplot(2,3,1)
[f,xi] = ksdensity(s1(:,4));
plot(xi,f);
title('房价指标--聚类1')

subplot(2,3,2)
[f,xi] = ksdensity(s2(:,4));
plot(xi,f);
title('房价指标--聚类2')

subplot(2,3,3)
[f,xi] = ksdensity(s3(:,4));
plot(xi,f);
title('房价指标--聚类3')

subplot(2,3,4)
[f,xi] = ksdensity(s4(:,4));
plot(xi,f);
title('房价指标--聚类4')

subplot(2,3,5)
[f,xi] = ksdensity(s5(:,4));
plot(xi,f);
title('房价指标--聚类5')

% z1 = linkage(D);%默认最短距离
% subplot(2,2,1)
% dendrogram(z1)
%
% z2 = linkage(D,'complete');  %默认最长距离
% subplot(2,2,2)
% dendrogram(z2)
% 
% z3 = linkage(D,'average');  %类平均距离
% subplot(2,2,3)
% dendrogram(z3)
% 
% z4 = linkage(D,'ward');  %离差平方和距离
% subplot(2,2,4)
% dendrogram(z4)
% 
% R = [cophenet(z1,D),cophenet(z2,D),cophenet(z3,D),cophenet(z4,D)];
```

### 3.5 聚类分析——K-均值聚类

#### 3.5.1 K-均值聚类模型

![image-20220809093458213](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809093458213.png)

![image-20220809093927044](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809093927044.png)

![image-20220809094133825](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809094133825.png)

![image-20220809094228498](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809094228498.png)

![image-20220809094304590](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809094304590.png)

![image-20220809094446212](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809094446212.png)

![image-20220809102716001](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809102716001.png)

![image-20220809094739651](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809094739651.png)

![image-20220809095955321](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809095955321.png)

**示例：**

![image-20220809102514151](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809102514151.png)

### 3.6 典型相关分析

#### 3.6.1典型相关分析模型

![image-20220809113814039](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809113814039.png)

![image-20220809113846199](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809113846199.png)

![image-20220809122306742](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809122306742.png)

![image-20220809122426828](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809122426828.png)

![image-20220809122930743](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809122930743.png)

![image-20220809123116881](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809123116881.png)

![image-20220809123456560](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809123456560.png)

![image-20220809123659362](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809123659362.png)

![image-20220809124132687](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809124132687.png)

![image-20220809125044563](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809125044563.png)

![image-20220809125202229](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809125202229.png)

![image-20220809125247166](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809125247166.png)

![image-20220809143407524](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809143407524.png)

#### 3.6.1 典型相关分析代码

```matlab
function cca = CCA_deno(X,Y)
%CCA_demo用于求解典型相关分析
%输入参数X,Y代表两组样本数据，具有相同的行数
%输出参数cca表示一个结构体，包含相关参数

    %% 1、数据预处理
    [n,p] = size(X);  %X的样本量及指标数
    q = size(Y,2); %Y的指标数
    data = [X,Y];  %合并样本X,Y
    dcov = corrcoef(data);  %相关系数矩阵
    cov11 = dcov(1:p,1:p);  %X的协方差矩阵
    cov22 = dcov(p+1:end,p+1:end);%Y的协方差矩阵
    cov12 = dcov(1:p,p+1:end); %X,Y的协方差矩阵
    cov21 = cov12';
    
    %% 2、构造A和B
    A = cov11\cov12*(cov22\cov21);
    B = cov22\cov21*(cov11\cov12);
    
    %% 3、求A和B的特征值和特征向量
    [VA,DA] = eig(A);
    [VB,DB] = eig(B);
    %特征值和特征向量排序
    [DA,indA] = sort(diag(DA),'descend');
    [DB,indB] = sort(diag(DB),'descend');
    VA = VA(:,indA);
    VB = VB(:,indB);
    A = VA*(VA'*cov11*VA)^(-1/2);
    B = VB*(VB'*cov22*VB)^(-1/2);
    
    %% 4、求典型相关系数
    if p>q
        r = sqrt(DB)';
    else
        r = sqrt(DA)';
    end
    U = zscore(X*A);%X得分情况
    V = zscore(Y*B); %Y得分情况
    
    %% 5、典型相关系数检验
    p = min([p,q]);
    D = r.^2;
    lambda = zeros(p,1);
    T = zeros(p,1);
    f = zeros(p,1);
    pChisq = zeros(p,1);
    for k = 1:p
        lambda(k) = prod(1-D(k:p));
        T(k) = -(n-(p+q+3)/2)*log(lambda(k));
        f(k) = (p-k+1)*(q-k+1); %自由度
        pChisq(k) = 1-chi2cdf(T(k),f(k)); %卡方检验
    end
        
    %% 6、参数输出
    cca.A = A;
    cca.B = B;
    cca.r = r;
    cca.U = U;
    cca.V = V;
    cca.pChisq = pChisq;
end
```

### 3.7 对应分析

#### 3.7.1 对应分析模型

![image-20220809151052091](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809151052091.png)

![image-20220809151855510](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809151855510.png)

![image-20220809152042563](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809152042563.png)

![image-20220809152306808](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809152306808.png)

![image-20220809152433495](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809152433495.png)

**示例：**

![image-20220809160848670](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809160848670.png)

![image-20220809161122454](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809161122454.png)





#### 3.7.2 对应分析代码

```matlab
function carq = CorrespA_RQ(data,label)
%CorrespA_RQ对应分析函数
%data是样本数据，label对于变量名称和行名称
%carq是输出参数结构体

    %% 1、数据预处理
    T = sum(sum(data));%计算总和
    P = data/T; %计算概率矩阵
    r = sum(P,2); %计算边缘分布，行和
    c = sum(P); %计算边缘分布，列和
    Z = (P-r*c)./sqrt((r*c));
    
    %% 2、求过渡矩阵Z的奇异值分解
    [u,D,v] = svd(Z,'econ');%奇异值分解
    G = u*sqrt(D);
    F = v*sqrt(D);
    
    %% 3、求解Z*Z惯量和累积贡献率
    lamda = diag(D).^2; %计算特征值
    con_rate = lamda/sum(lamda); %计算贡献率
    
    %% 4、卡方检验
    ksi2square = T*(lamda); %计算卡方统计量分解
    T_ksi2square = sum(ksi2square);
    def = (size(data,1)-1)*(size(data,2)-1); %自由度
    pChisq = 1-chi2cdf(T_ksi2square,def); %卡方检验概率值
    
    %% 5、对应分析可视化
    num = size(G,1); %样本点个数
    rang = minmax(G(:,[1,2])); %坐标的取值范围
    delta = (rang(:,2)-rang(:,1))/(5*num); %画图的标注位置调整
    ch = cellstr(label(1,2:size(label,2))); %对于列变量名称
    yb = cellstr(label(2:size(label,1),1))';%对于行样本名称
    h1 = plot(G(:,1),G(:,2),'b*','LineWidth',1.3);
    hold on
    text(G(:,1)-delta(1),G(:,2)-3*delta(2),yb)
    h2 = plot(F(:,1),F(:,2),'rH','LineWidth',1.3);
    text(F(:,1)+delta(1),F(:,2),ch)
    h = refline(0,0);h.Color = 'k';h.LineStyle = ':';
    h1m = minmax(h1.YData);h2m = minmax(h2.YData);
    mind = min(h1m(1),h2m(1));
    maxd = max(h1m(2),h2m(2));
    %添加垂直辅助线
    plot(zeros(1,10),linspace(mind-5*delta(2),maxd+5*delta(1),10),'k:');
    xtext = strcat('dimension1(',num2str(con_rate(1)*100),'%)');
    ytext = strcat('dimension2(',num2str(con_rate(2)*100),'%)');
    xlabel(xtext),ylabel(ytext)
    title('Correspondence analysls chart');
    
    rowclass = yb(G(:,1)>0); %提出第一类样本点
    coiclass = ch(F(:,1)>0); %提出第一类变量

    %% 5、参数输出
    carq.lamda = lamda;
    carq.con_rate = con_rate;
    carq.F = F;
    carq.G = G;
    carq.Total_Ksi2square = T_ksi2square;
    carq.pChisq = pChisq; 
    carq.FirstSample = rowclass;
    carq.FirstVar = coiclass;

end
```



## 4  关联与因果

### 4.1 灰色关联分析

#### 4.1.1 灰色关联分析理论

![image-20220809171306543](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809171306543.png)

![image-20220809171329980](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809171329980.png)

![image-20220809171352575](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809171352575.png)

![image-20220809171410793](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809171410793.png)

![image-20220809171509869](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809171509869.png)

*示例：*

![image-20220809171631745](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809171631745.png)







### 4.2 Spearman与Person相关分析

#### 4.2.1 Spearman与Person相关分析理论

> 1、两个变量是否独立
>
> 2、两个变量是否有共同趋势
>
> 3、一个变量的变化多大程度上由另一个变量的变量来解释

![image-20220809173904886](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809173904886.png)

![image-20220809174519782](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809174519782.png)

![image-20220809174702554](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809174702554.png)

![image-20220809184848126](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809184848126.png)

![image-20220809190606128](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809190606128.png)

**偏相关系数：**

![image-20220809203618694](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809203618694.png)

![image-20220809203637555](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220809203637555.png)



### 4.3 Copula相关









### 4.4 标准化回归分析











### 4.5 格兰杰因果检验





# 优化

## 1 优化与控制

### 1.1 线性规划、整数规划、0-1规划





### 1.2 非线性规划与智能优化算法





### 1.3 多目标规划与目标规划





### 1.4 动态规划





### 1.5 网络优化



# 插值算法

### 1.1 Lagrange插值

> 分别构造x~0~, x~1~ ,… x~n~,上的n次插值基函数l~0~(x),l~1~(x),l~n~(x),满足性质：
> $$
> l~i~(x~j~)=δ~ij~=1   ,i=j=0,1,2m,...,n
> $$
> 即![image-20220804204749995](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220804204749995.png)
>
> ![image-20220804204850441](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220804204850441.png)
>
> ![image-20220804204919518](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220804204919518.png)
>
> 

### 1.2 Newton插值

> ![image-20220804205110795](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220804205110795.png)
>
> ![image-20220804205140539](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220804205140539-16596175030263.png)
>
> ![image-20220804205205391](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220804205205391.png)
>
> ![image-20220804205225025](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220804205225025.png)
>
> ![image-20220804205247634](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220804205247634.png)
>
> ![image-20220804205348307](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220804205348307.png)
>
> ![image-20220804205359786](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220804205359786.png)
>
> 

### 1.3 Hermite插值

> ![image-20220804205707046](%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%EF%BC%88matlab%E7%AE%97%E6%B3%95%EF%BC%89.assets/image-20220804205707046.png)



