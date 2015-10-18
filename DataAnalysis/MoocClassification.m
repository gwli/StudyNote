clear all;
close all;
clc;
MoocData = importdata('MoocDataPart.txt');
NewMoocData2=zeros(size(MoocData));
for  i =1:size(MoocData,2)
    NewMoocData(:,i)=MoocData(:,i)-mean(MoocData(:,i));
    CovNewMoocData  =NewMoocData(:,i)'*NewMoocData(:,i);
    NewMoocData2(:,i) = NewMoocData(:,i)./sqrt(CovNewMoocData);
end
X= NewMoocData2;
K=2
[G,C] = kmeans(X,K);
clr = lines(K);
figure, hold on
scatter3(X(:,6), X(:,2), X(:,4), 36, clr(G,:), 'Marker','.')
scatter3(C(:,6), C(:,2), C(:,4), 100, clr, 'Marker','o', 'LineWidth',3)
hold off
view(3), axis vis3d, box on, rotate3d on
xlabel('x'), ylabel('y'), zlabel('z')
ylabel 'Petal Widths (cm)';
%%%%%%%%%%%%%%%%%%
plot(X(:,6), X(:,2), 36, clr(G,:), 'Marker','.')
plot(C(:,6), C(:,2), 100, clr, 'Marker','o', 'LineWidth',3)
%现在数据可以分类，并且是很多类。
%是可以分类，但是这些数据怎样进行分类。

