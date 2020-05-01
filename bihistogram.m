function Bi_HistogramEqualization()
[f,p]=uigetfile('*.*','选择图像文件');
if f
I=imread(strcat(p,f));
end
 
Ir=I(:,:,1);%提取红色分量
Ig=I(:,:,2);%提取绿色分量
Ib=I(:,:,3);%提取蓝色分量
I1=BBHE(Ir);
I2=BBHE(Ig);
I3=BBHE(Ib);
In=cat(3,I1,I2,I3);  %cat用L于构造多维数组


subplot(1,2,1);imshow(I);
title('A). Original image');  

subplot(1,2,2);imshow(mat2gray(In),[]);
title('B). BBHE');

end

function A=BBHE(I)
Xm=floor(mean2(I));  %求图像灰度均值Xm
Xmin=min(min(I));
Xmax=max(max(I));
[m,n]=size(I);  
Xl=zeros(1,Xm+1);       %记录图像在（Xmin,Xm)范围内的灰度值
Xu=zeros(1,256);       %记录图像在（Xm,Xmax)范围内的灰度值
nl=0;
nu=0;
for i=1:m
    for j=1:n
        if I(i,j)<Xm || I(i,j)==Xm      %统计≤平均值的各级灰度值数量及总数
            Xl(I(i,j)+1) = Xl(I(i,j)+1) + 1; %存在灰度值为0的情况，但矩阵下标不能为0，因此+1
            nl=nl+1;
        else                            %统计＞平均值的各级灰度值数量及总数
            Xu(I(i,j)+1) = Xu(I(i,j)+1) + 1;
            nu=nu+1;
        end
    end
end
X_(m+1)=Xm+1;
while(Xu(X_(m+1)+1))==0
    X_(m+1)=X_(m+1)+1;
end
 
Pl=Xl./nl;  %记录对应各级灰度值的概率密度
Pu=Xu./nu;
Cl=Xl; %累计密度函数
Cu=Xu;
Cl(1)=Pl(1);
Cu(X_(m+1))=Pu(X_(m+1));
for i=2:Xm+1
    Cl(i)=Pl(i) + Cl(i-1);
end
for i=X_(m+1)+1:256
    Cu(i)=Pu(i) + Cu(i-1);
end
%灰度转换函数
fl=Cl;fu=Cu;
for i=1:Xm
    fl(i)= Xmin + Cl(i)*(Xm-Xmin);   
end
for i=X_(m+1):256
    fu(i)= X_(m+1) + Cu(i)*(Xmax-X_(m+1));
end
%两个子图像合并
I_equal = I;
for i=1:m
    for j=1:n
        if I(i,j)<Xm || I(i,j)==Xm
           I_equal(i,j) = fl(I(i,j)+1);
        else
           I_equal(i,j) = fu(I(i,j)+1);
        end
    end
end
A=I_equal;
end