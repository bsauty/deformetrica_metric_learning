close all;
clear all;
addpath '../../../../../utilities/matlab/'

[TemplatePts,~] = VTKPolyDataReader('data/sourcePoints.vtk');
[TargetPts,~] = VTKPolyDataReader('data/targetPoints.vtk');
[ModelPts,~] = VTKPolyDataReader('output/Registration_sourcePoints_to_subject_0__t_9.vtk');
CP =  load('output/Registration_ControlPoints.txt');
MOM = readMomentaFile('output/Registration_Momentas.txt');

figure;
set(gcf,'OuterPosition',[-1500 1750 1500 500]);

subplot(1,3,1);
scatter(TemplatePts(:,1), TemplatePts(:,2),200,'kx');hold on;
xlim([10,70]);
ylim([15,35]);
quiver(CP(1,1),CP(1,2),MOM(1,1)',MOM(2,1)',0,'LineWidth',3,'color','g');
quiver(CP(2,1),CP(2,2),MOM(1,2)',MOM(2,2)',0,'LineWidth',4,'color','r');
quiver(CP(3,1),CP(3,2),MOM(1,3)',MOM(2,3)',0,'LineWidth',1,'color','b');
grid on;
title('Source points with momenta');

subplot(1,3,2);
scatter(ModelPts(:,1), ModelPts(:,2),200,'kx');hold on;
xlim([10,70]);
ylim([15,35]);
grid on;
title('Reconstructed points');

subplot(1,3,3);
scatter(TargetPts(:,1), TargetPts(:,2),200,'kx');hold on;
xlim([10,70]);
ylim([15,35]);
grid on;
title('Target points');