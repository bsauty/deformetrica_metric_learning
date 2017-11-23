close all;
clear all;
addpath '../../../../../utilities/matlab/'

name = {'australopithecus','habilis','erectus','sapiens'};

% load data
P = cell(1,5);
E = cell(1,5);
for s = 1:4
	[P{s},E{s}] = VTKPolyDataReader(['data/skull_' name{s} '.vtk']);
end

% load the initial model we provide as input of the atlas construction method	
[InitTemplatePts,InitTemplateEdges] = VTKPolyDataReader('data/template.vtk');

pos = [1 2 3 4];
figure;
for s=1:4
	subplot(2,4,pos(s));
	hold on
	for k=1:size(E{s},1)
		plot(P{s}(E{s}(k,:),1),P{s}(E{s}(k,:),2),'-b','LineWidth',2);
	end
	title([name{s},'   (t = ',num2str(s),')']);
	axis([-150 100 -100 120]);
end
subplot(2,4,5)
hold on
for k=1:size(InitTemplateEdges,1)
	plot(InitTemplatePts(InitTemplateEdges(k,:),1),InitTemplatePts(InitTemplateEdges(k,:),2),'-k','LineWidth',2);
end
axis([-150 100 -100 120]);
title('initial template model   (t = 0)');
set(gcf,'OuterPosition',[-1500 1200 1500 750]);
