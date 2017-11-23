close all;
clear all;
addpath '../../../../../utilities/matlab/'

scale = 1;
indexes = 1:5;
str = 'mean';

% load the initial & final templates
Itemp_i = imread(['data/digit_2_', str, '.png']);
Itemp_f = imread(['output/Atlas_digit_2_', str, '.png']);

a = min(Itemp_i(:)); b = max(Itemp_i(:));
Itemp_i = (Itemp_i-a)*(255/double(b-a));
Itemp_f = (Itemp_f-a)*(255/double(b-a));

% load target data and results
Itarget = cell(1,5);
Iresult = cell(1,5);
for s = 1:5
	Itarget{s} = imread(['data/digit_2_sample_', num2str(indexes(s)), '.png']);
    Iresult{s} = imread(['output/Atlas_digit_2_', str, '_to_subject_', num2str(indexes(s)-1), '__t_9.png']);

    Iresult{s} = (Iresult{s}-a)*(255/double(b-a));
end

% load optimal position of control points
CP = load('output/Atlas_ControlPoints.txt');

% load set of optimal momentum vectors
MOM = readMomentaFile('output/Atlas_Momentas.txt');

% plot
color = {'g','c','m','r','y'};
figure;
subplot(2,6,1)
imshow(imresize(Itemp_i, scale));
title('Initial template');
subplot(2,6,7)
imshow(imresize(Itemp_f, scale)); hold on; 
for s=1:5
    quiver(CP(:,1),CP(:,2),MOM(1,:,s)',MOM(2,:,s)',0.8,color{s},'LineWidth',1.5);
end
title('Final template');

for s=1:5
	subplot(2,6,1+s);
    imshow(imresize(Itarget{s}, scale)); hold on;
    plot(1,1, 'color', color{s},'marker', 'o', 'markersize', 5, 'markerfacecolor', color{s});
    title(['Target ', num2str(indexes(s))]);
    
    subplot(2,6,7+s);
    imshow(imresize(Iresult{s}, scale)); hold on;
    plot(1,1, 'color', color{s},'marker', 'o', 'markersize', 5, 'markerfacecolor', color{s});
    title(['Result ', num2str(indexes(s))]);
end
set(gcf,'OuterPosition',[-1500 1750 1750 1000]);



