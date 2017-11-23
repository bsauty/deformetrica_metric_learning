close all;
clear all;
addpath '../../../../../utilities/matlab/'

% load the reference geodesic
Igeo = cell(1, 5);
Igeo{1} = imread('output/Reference_geodesicI1_t=0.png');
Igeo{2} = imread('output/Reference_geodesicI1_t=7.png');
Igeo{3} = imread('output/Reference_geodesicI1_t=15.png');
Igeo{4} = imread('output/Reference_geodesicI1_t=22.png');
Igeo{5} = imread('output/Reference_geodesicI1_t=29.png');

% load the parallel trajectory
Ipt = cell(1, 5);
Ipt{1} = imread('output/ParallelTransport_I1_t=0.png');
Ipt{2} = imread('output/ParallelTransport_I1_t=7.png');
Ipt{3} = imread('output/ParallelTransport_I1_t=15.png');
Ipt{4} = imread('output/ParallelTransport_I1_t=22.png');
Ipt{5} = imread('output/ParallelTransport_I1_t=29.png');

% plot
indexes = [0, 7, 15, 22, 29];

figure;
set(gcf,'OuterPosition',[-1500 1750 2000 800]);

for s=1:5
	subplot(2,5,s);
    imshow(Igeo{s}); hold on;
    title(['Reference geodesic (t = ', num2str(indexes(s)),')']);
    
    subplot(2,5,5+s);
    imshow(Ipt{s}); hold on;
    title(['Parallel trajectory (t = ', num2str(indexes(s)),')']);
end



