%%%%����ʱ���������������������Գ�Ȩ�أ�Ȼ����ݸ������������з����׼ȷ�ʲ��ԣ��õ����������Ը�����Ԥ�����
%%%%������֤�������룬ͨ��Ȩ�غ͸�����ˣ��õ����ĸ��ʾ���ͨ��ʱ��Ȩ��Ԥ��ĸ��ʡ���Ҫ����ͨ�������ķ��෽�����жԱȡ�
clear all;clc; close all;
InputDate=xlsread('8.xls','Sheet1');
Train_lab=InputDate(:,10);%����
DataLength=length(Train_lab);%����
train_m=InputDate(:,1:9);%��������
% train_m_data=tr(:,1:10);%��������
%ѵ���Ͳ����飬���ĸ����͵Ľ�����ȡ
a1=1;a2=1;a3=1;a4=1;
for i=1:DataLength
    %�������� ���±���д洢
    if Train_lab(i)==0 ;A1(a1,1)=i;a1=a1+1;
    elseif Train_lab(i)==1;A2(a2,1)=i;a2=a2+1;
    elseif Train_lab(i)==2;A3(a3,1)=i;a3=a3+1;
    elseif Train_lab(i)== 3 ;A4(a4,1)=i;a4=a4+1;
    end
end
% ѡ��ѵ�����Ͳ��Լ�
tr_data=[];Train_lab=[];te_data=[];te_label=[];%���ϸ�������۵����ݽ�����ղ�����
b1=uint32(a1)/2-1;
u=1;v=b1;
for c1=1:b1
    v=v+1;
    tr_data(u,:)=train_m(A1(c1),:);
    Train_lab(u,1)=Train_lab(A1(c1),1);
    te_data(u,:)=train_m(A1(v),:);
    te_label(u,1)=Train_lab(A1(v),1);
    %     last_data(u,:)=train_m(first(o-q+1),:);
    %     last_label(u,1)=Train_lab(first(o-q+1),1);
    u=u+1;
end
%�洢ѵ����һ��
train{1}=tr_data;
trainlab{1}=Train_lab;
%�洢���Ե�һ��
test{1}=te_data;
testlab{1}=te_label;

tr_data=[];Train_lab=[];te_data=[];te_label=[];
b2=uint32(a2)/2-1;
u=1;v=b2;
for c2=1:b2
    v=v+1;
    tr_data(u,:)=train_m(A2(c2),:);
    Train_lab(u,1)=Train_Lab(A2(c2),1);
    %
    te_data(u,:)=train_m(A2(v),:);
    te_label(u,1)=train_lab(A2(v),1);
    %     last_data(u,:)=train_m(second(m-r+1),:);
    %     last_label(u,1)=train_lab(second(m-r+1),1);
    u=u+1;
end
%�洢ѵ���ڶ���
train{2}=tr_data;
trainlab{2}=Train_lab;
%�洢���Եڶ���
test{2}=te_data;
testlab{2}=te_label;

tr_data=[];Train_lab=[];te_data=[];te_label=[];
b3=uint32(a3)/2-1;
u=1;v=b3;
for c3=1:b3
    v=v+1;
    tr_data(u,:)=train_m(A3(c3),:);
    Train_lab(u,1)=train_lab(A3(c3),1);
    %
    te_data(u,:)=train_m(A3(v),:);
    te_label(u,1)=train_lab(A3(v),1);
    %     last_data(u,:)=train_m(thrid(n-s+1),:);
    %     last_label(u,1)=train_lab(thrid(n-s+1),1);
    u=u+1;
end
%�洢ѵ��������
train{3}=tr_data;
trainlab{3}=Train_lab;
%�洢���Ե�����
test{3}=te_data;
testlab{3}=te_label;
tr_data=[];Train_lab=[];te_data=[];te_label=[];
b4=uint32(a4)/2-1;
u=1;v=b4;
for c4=1:b4
    v=v+1;
    tr_data(u,:)=train_m(A4(c4),:);
    Train_lab(u,1)=train_lab(A4(c4),1);
    %
    te_data(u,:)=train_m(A4(v),:);
    te_label(u,1)=train_lab(A4(v),1);
    %     last_data(u,:)=train_m(fore(p-t+1),:);
    %     last_label(u,1)=train_lab(fore(p-t+1),1);
    u=u+1;
end
%�洢ѵ��������
train{4}=tr_data;
trainlab{4}=Train_lab;
%�洢���Ե�����
test{4}=te_data;
testlab{4}=te_label;
    train_data=[];
    train_label=[];
    test_data=[];
    test_label=[];
%����������������
for d=1:length(train)
    train_data=[train_data;train{1,d}];
    train_label=[train_label;trainlab{1,d}];
    test_data=[test_data;test{1,d}];
    test_label=[test_label;testlab{1,d}];
end