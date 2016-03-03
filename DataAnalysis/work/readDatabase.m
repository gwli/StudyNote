clear all; clc; close all;
my_dir='G:\Client\facerecognition\GSRC_ECCV\FERET_80_80-ÈËÁ³Êý¾Ý¿â';
% !dir /s/b *.tif> filelist1.txt
fid=fopen('G:\Client\facerecognition\GSRC_ECCV\FERETfilelist.txt');
files = textscan(fid, '%s');
fclose(fid);
files = files{1};
I=length(files);
for i=1:I
    path = files{i,1};
    imageData(:,:,i) = imread(path);
end
Test_DAT=imageData(:,:,1:7:end);
imageData(:,:,1:7:end)=[];
Train_DAT =imageData;
save Test_DAT;
save Train_DAT;




