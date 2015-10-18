function [MSE] = MyBP(P,P_test,T,T_test)
% ����һ���µ�ǰ�������� 
net=newff(minmax(P),[6,1],{'tansig','purelin'},'traingdm');
% ����ѵ������ 
net.trainParam.show = 50; 
net.trainParam.lr = 0.05; 
net.trainParam.mc = 0.9; 
net.trainParam.epochs = 1000; 
net.trainParam.goal = 1e-3; 
% ���� TRAINGDM �㷨ѵ�� BP ���� 
[net,tr]=train(net,P,T); 
% �� BP ������з��� 
A = sim(net,P_test) 
% ���������� 
E = T_test - A 
MSE=mse(E) 
echo off 
