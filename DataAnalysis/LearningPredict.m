clear all;; close all;
M=100;
N=8;
A=randi([0,100],M,N);
B=randi([100,200],M,N);
%%��ת�ƾ���P
Wopt0=pinv(A(1:0.8*M,:))*B(1:0.8*M,:);
%��һ�����ݣ� ���������ʲô������
c= randi([0,1],1,M);
%% Q�����ص���Ԫ���������ò�ƺ���Ԫû��ʲô��ϵ��
Q=1.2*N;
error = A(0.8*M+1:end,:)*Wopt0-B(0.8*M+1:end,:);
logist = sum(mean(1./(1+exp(-error))));
Wopt1=Wopt0;
f0=1;
k=0;
alpha =.008;
error=[];
while f0>10e-4 && k<50
        Wopt0=Wopt1;
        err = A(0.8*M+1:end,:)*Wopt0-B(0.8*M+1:end,:);
        Wopt1 = Wopt0-alpha.*diag(mean(err)).*ones(N,N);
        %���Ӧ����ô�ӣ�
        k=k+1;
        f0=norm(Wopt1-Wopt0);
        error=[error f0];
end
% plot(error)
% ��ô��ľ��������������
X=A+B;
W = pca(X,5);
%�ǲ�����Ҫ��һ��
% �������ģ����
P=X(1:0.8*M,:);
T=c(1:0.8*M);
P_test=X(1+0.8*M:end,:);
T_test=c(1+0.8*M:end);
MSE = MyBP(P.',P_test.', T,T_test);
