close all;
clear all;
addpath '../../../../../utilities/matlab/'

Isource = imread('data/I1.png');
Imodel = imread('output/Registration_I1_to_subject_0__t_9.png');
Itarget= imread('data/I2.png');
CP =  load('output/Registration_ControlPoints.txt');
MOM = readMomentaFile('output/Registration_Momentas.txt');

figure;
set(gcf,'OuterPosition',[-1500 1750 1500 500]);

subplot(1,3,1);
imagesc(Isource); colormap gray; hold on
quiver(CP(:,1),CP(:,2),MOM(1,:)',MOM(2,:)',0,'LineWidth',3);
title('Source image with momenta');

subplot(1,3,2);
imagesc(Imodel); colormap gray; hold on
title('Reconstructed image');

subplot(1,3,3);
imagesc(Itarget); colormap gray; hold on
title('Target image');

