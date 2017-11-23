close all;
clear all;

scale = 1;
indexes = 1:5;

% load data
I = cell(1,5);
for s = 1:5
	I{s} = imread(['data/digit_2_sample_', num2str(indexes(s)), '.png']);
end

% load the initial model we provide as input of the atlas construction method	
Itemplate = imread('data/digit_2_mean.png');

pos = [3 4 1 2 6];
figure;
for s=1:5
	subplot(2,3,pos(s));
	imshow(imresize(I{s}, scale));
	title(['Sample ', num2str(indexes(s))]);
	%axis([-150 100 -100 120]);
end
subplot(2,3,5)
imshow(imresize(Itemplate, scale))
title('Initial template');
set(gcf,'OuterPosition',[-1500 1200 1500 1000]);
