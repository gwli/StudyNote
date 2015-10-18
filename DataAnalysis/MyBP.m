function [MSE] = MyBP(P,P_test,T,T_test)
% 创建一个新的前向神经网络 
net=newff(minmax(P),[6,1],{'tansig','purelin'},'traingdm');
% 设置训练参数 
net.trainParam.show = 50; 
net.trainParam.lr = 0.05; 
net.trainParam.mc = 0.9; 
net.trainParam.epochs = 1000; 
net.trainParam.goal = 1e-3; 
% 调用 TRAINGDM 算法训练 BP 网络 
[net,tr]=train(net,P,T); 
% 对 BP 网络进行仿真 
A = sim(net,P_test) 
% 计算仿真误差 
E = T_test - A 
MSE=mse(E) 
echo off 
