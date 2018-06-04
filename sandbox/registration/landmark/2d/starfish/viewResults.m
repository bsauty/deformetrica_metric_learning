close all;
clear all;
addpath '../../../../../utilities/matlab/'

[TemplatePts,TemplateEdges] = VTKPolyDataReader('data/starfish_reference.vtk');
[TargetPts,TargetEdges] = VTKPolyDataReader('data/starfish_target.vtk');
[ModelPts,ModelEdges] = VTKPolyDataReader('output/Registration_starfish_reference_to_subject_0__t_9.vtk');
CP =  load('output/Registration_ControlPoints.txt');
MOM = readMomentaFile('output/Registration_Momentas.txt');

figure;
set(gcf,'OuterPosition',[-1500 1750 1500 500]);

subplot(1,3,1);
for k=1:size(TemplatePts,1)-1
    plot([TemplatePts(k,1),TemplatePts(k+1,1)],[TemplatePts(k,2),TemplatePts(k+1,2)],'-k','LineWidth',2);
    hold on;
end
plot(TemplatePts(size(TemplatePts,1),1),TemplatePts(size(TemplatePts,1),2),'-k','LineWidth',2);
quiver(CP(:,1),CP(:,2),MOM(1,:)',MOM(2,:)',0,'LineWidth',2,'color','b');
grid on;
title('Source shape with momenta');

subplot(1,3,2);
for k=1:size(ModelPts,1)-1
    plot([ModelPts(k,1),ModelPts(k+1,1)],[ModelPts(k,2),ModelPts(k+1,2)],'-k','LineWidth',2);
    hold on;
end
plot(ModelPts(size(ModelPts,1),1),ModelPts(size(ModelPts,1),2),'-k','LineWidth',2);
grid on;
title('Reconstructed shape');

subplot(1,3,3);
for k=1:size(TargetPts,1)-1
    plot([TargetPts(k,1),TargetPts(k+1,1)],[TargetPts(k,2),TargetPts(k+1,2)],'-k','LineWidth',2);
    hold on;
end
plot(TargetPts(size(TargetPts,1),1),TargetPts(size(TargetPts,1),2),'-k','LineWidth',2);
grid on;
title('Target shape');

