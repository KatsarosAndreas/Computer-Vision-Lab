
% Paths for input images and the functions
photoDir = 'C:\Users\AndreasK\Documents\MATLAB\Exercise 1\Matlab_Ex1\photos\';
functionDir = 'C:\Users\AndreasK\Documents\MATLAB\Exercise 1\Matlab_Ex1\';
addpath(functionDir);

% Load images
woman = im2double(imread([photoDir, 'woman.png']));
hand = im2double(imread([photoDir, 'hand.png']));

% Ensure grayscale
if size(woman, 3) == 3
    woman = rgb2gray(woman);
end
if size(hand, 3) == 3
    hand = rgb2gray(hand);
end

% Resize images to match
[rows, cols] = size(hand);
woman = imresize(woman, [rows, cols]);

% Create mask according to PDF properties
m1 = zeros(rows, cols);
% Keep the same eye region coordinates for consistent positioning
eyeRegion = round([rows*0.3 rows*0.7 cols*0.3 cols*0.7]);
m1(eyeRegion(1):eyeRegion(2), eyeRegion(3):eyeRegion(4)) = 1;

% Number of levels
numLevels = 5;

% Generate pyramids as per PDF equations
Gm1 = genPyr(m1, 'gauss', numLevels);     % Equation (1) - This includes inherent Gaussian smoothing
LI1 = genPyr(woman, 'lap', numLevels);     % Equation (2) for woman
LI2 = genPyr(hand, 'lap', numLevels);      % Equation (2) for hand

% Create blended pyramid B as per Equations (3) and (4)
B = cell(1, numLevels);
for j = 1:numLevels
    % Get current level dimensions
    [levelRows, levelCols] = size(LI1{j});
    % Resize mask to match current level
    gj = imresize(Gm1{j}, [levelRows, levelCols]);
    % Blend according to equation (4)
    B{j} = gj .* LI1{j} + (1 - gj) .* LI2{j};
end

% Reconstruct using pyramid reconstruction
result = pyrReconstruct(B);

% Display
figure;
imshow(result, []);
title('Blended Result');