close all;
clear all;
addpath '../../../../../utilities/matlab/'

% load optimal position of control points
CP = load('output/Regression_ControlPoints.txt');

% load set of optimal momentum vectors
MOM = load('output/Regression_InitialMomenta.txt');

% load input data
name = {'australopithecus','habilis','erectus','sapiens'};
in_P = cell(1,4);
in_E = cell(1,4);
for s = 1:4
	[in_P{s},in_E{s}] = VTKPolyDataReader(['data/skull_' name{s} '.vtk']);
end
[in_P_template,in_E_template] = VTKPolyDataReader('data/template.vtk');

% load output data
out_P_trajectory = cell(1,20);
out_E_trajectory = cell(1,20);
for t = 1:21
	[out_P_trajectory{t},out_E_trajectory{t}] = VTKPolyDataReader(['output/Regression_baseline_template_trajectory___t_' num2str(t-1) '.vtk']);
end
[out_P_template,out_E_template] = VTKPolyDataReader('output/Regression_template.vtk');

% display the shape temporal evolution
figure;
set(gcf,'OuterPosition',[-1500 1200 1500 1200]);
for t=1:20 
    subplot(4,5,t);
    
    s=(t-1)/20*5;
    if any(s == [1,2,3,4])
        for k=1:size(in_E{s},1)
            plot(in_P{s}(in_E{s}(k,:),1),in_P{s}(in_E{s}(k,:),2),'-b','LineWidth',2);
            hold on
        end
        title(name{s});
    else
        title(['time = ',num2str(s,'%.2f')]);
        hold on
    end
        
    for k=1:size(out_E_trajectory{t},1)
        plot(out_P_trajectory{t}(out_E_trajectory{t}(k,:),1),out_P_trajectory{t}(out_E_trajectory{t}(k,:),2),'-k','LineWidth',1);
        hold on
    end
    
    axis([-150 100 -100 120]);
    box on
	pause(0.01);
end

% display the estimated template
subplot(4,5,1) 
for k=1:size(out_E_template,1)
    plot(out_P_template(out_E_template(k,:),1),out_P_template(out_E_template(k,:),2),'-k','LineWidth',2);
    hold on
end
quiver(CP(:,1),CP(:,2),MOM(:,1)*3,MOM(:,2)*3,0.5,'r','LineWidth',0.5);
title('template');

figure;
set(gcf,'OuterPosition',[200 1200 500 500]);
for k=1:size(out_E_template,1)
    plot(in_P_template(in_E_template(k,:),1),in_P_template(in_E_template(k,:),2),'--b','LineWidth',0.5); hold on;
    plot(out_P_template(out_E_template(k,:),1),out_P_template(out_E_template(k,:),2),'-k','LineWidth',2);
    hold on
end
quiver(CP(:,1),CP(:,2),MOM(:,1)*5,MOM(:,2)*5,0.5,'r','LineWidth',1);
axis([-150 100 -100 120]);
title('Estimated template and momentum vectors');

% Alternative visualization
% figure;
% for t=0:12
%     [P,E] = VTKPolyDataReader(['output/Regression_baseline_template_trajectory___t_' num2str(t) '.vtk']);
%     clf;
%     for k=1:size(E,1)
%         plot(P(E(k,:),1),P(E(k,:),2),'-k','LineWidth',1);
%         hold on;
%     end
%     pause(0.2);
% end

