clear all; 
close all;
addpath '../../../../../utilities/matlab/'

% load estimated template shape
[TemplatePts,TemplateEdges] = VTKPolyDataReader('output/Atlas_template.vtk');

% load optimal position of control points
CP = load('output/Atlas_ControlPoints.txt');

% load set of optimal momentum vectors
MOM = readMomentaFile('output/Atlas_Momentas.txt');

name = {'australopithecus','habilis','erectus','neandertalis','sapiens'};

% load data
P = cell(1,5);
E = cell(1,5);
for s = 1:5
	[P{s},E{s}] = VTKPolyDataReader(['data/skull_' name{s} '.vtk']);
end

% load the frames of the 5 template-to-subjects deformations
Pt = cell(1,10);
for t=1:10
	Pt{t} = cell(1,5);
	for s=1:5
		Pt{t}{s} = VTKPolyDataReader(['output/Atlas_template_to_subject_' num2str(s-1) '__t_' num2str(t-1) '.vtk']);
	end
end

% display final template
% figure;
% set(gcf,'OuterPosition',[200 1200 500 500]);
% for k=1:size(TemplateEdges,1)
%     hold on
% 	plot(TemplatePts(TemplateEdges(k,:),1),TemplatePts(TemplateEdges(k,:),2),'-b','LineWidth',3);
% end
% title('Estimated template');
% axis([-150 100 -100 120]);

% display template-to-subject deformations
color = {'g','c','m','k','y'};
pos = [3 4 1 2 6];
figure;
set(gcf,'OuterPosition',[-1500 1200 1500 1000]);
for t=10:10
	clf;
	for s=1:5
		subplot(2,3,pos(s));
		hold on
		for k=1:size(E{s},1)
			plot(P{s}(E{s}(k,:),1),P{s}(E{s}(k,:),2),'-r','LineWidth',3);
		end
		for k=1:size(TemplateEdges,1)
			plot(Pt{t}{s}(TemplateEdges(k,:),1),Pt{t}{s}(TemplateEdges(k,:),2),color{s},'LineWidth',3);
		end
		for k=1:size(TemplateEdges,1)
			plot(TemplatePts(TemplateEdges(k,:),1),TemplatePts(TemplateEdges(k,:),2),'-b','LineWidth',3*(t==1)+(t>1)/2);
		end
		title(name{s});
		axis([-150 100 -100 120]);
	end
	subplot(2,3,5)
	hold on
	for s=1:5
		quiver(CP(:,1),CP(:,2),MOM(1,:,s)',MOM(2,:,s)',0.5,color{s},'LineWidth',3);
	end
	for k=1:size(TemplateEdges,1)
		plot(TemplatePts(TemplateEdges(k,:),1),TemplatePts(TemplateEdges(k,:),2),'-b','LineWidth',3);
	end
	title('Estimated template and momentum vectors');
	axis([-150 100 -100 120]);
	pause(0.0001);
end

