% Paths for input images and the functions
photoDir = 'C:\Users\AndreasK\Documents\MATLAB\Exercise 1\Matlab_Ex1\photos\';
functionDir = 'C:\Users\AndreasK\Documents\MATLAB\Exercise 1\Matlab_Ex1\';
addpath(functionDir);

% Load all images
P200 = im2double(imread([photoDir, 'P200.jpg']));
dog1 = im2double(imread([photoDir, 'dog1.jpg']));
dog2 = im2double(imread([photoDir, 'dog2.jpg']));
cat  = im2double(imread([photoDir, 'cat.jpg']));
bench = im2double(imread([photoDir, 'bench.jpg']));
myImg = im2double(imread([photoDir, 'my_img.jpg']));

% Define base size for the composition (take first two dims to avoid colormap error)
baseSize = size(P200);
baseSize = baseSize(1:2);

% Resize all images to match base size
dog1 = imresize(dog1, baseSize);
dog2 = imresize(dog2, baseSize);
cat  = imresize(cat, baseSize);
bench = imresize(bench, baseSize);
myImg = imresize(myImg, baseSize);

% Create masks that can overlap but will be normalized to sum to 1
m1 = zeros(baseSize);
m2 = zeros(baseSize);
m3 = zeros(baseSize);
m4 = zeros(baseSize);
m5 = zeros(baseSize);
m6 = zeros(baseSize);
% Define base size for the composition (take first two dims to avoid colormap error)
baseSize = size(P200);
baseSize = baseSize(1:2);

% Resize all images to match base size
dog1 = imresize(dog1, baseSize);
dog2 = imresize(dog2, baseSize);
cat  = imresize(cat, baseSize);
bench = imresize(bench, baseSize);
myImg = imresize(myImg, baseSize);

% Create masks that can overlap, but here we'll make them mostly non-overlapping
m1 = zeros(baseSize);
m2 = zeros(baseSize);
m3 = zeros(baseSize);
m4 = zeros(baseSize);
m5 = zeros(baseSize);
m6 = zeros(baseSize);

rows = baseSize(1);
cols = baseSize(2);

% Example: assign each image to a distinct horizontal band
% You can adjust these ranges to suit your aesthetic needs
m1(1 : floor(rows/6),              1:cols) = 1;        % top band for P200
m2(floor(rows/6)+1 : floor(2*rows/6), 1:cols) = 1;     % next band for dog1
m3(floor(2*rows/6)+1 : floor(3*rows/6), 1:cols) = 1;   % dog2
m4(floor(3*rows/6)+1 : floor(4*rows/6), 1:cols) = 1;   % cat
m5(floor(4*rows/6)+1 : floor(5*rows/6), 1:cols) = 1;   % bench
m6(floor(5*rows/6)+1 : rows,         1:cols) = 1;      % myImg (bottom band)

% Normalize masks to satisfy Property 1 (sum to 1 for every pixel)
maskSum = m1 + m2 + m3 + m4 + m5 + m6;
m1 = m1 ./ maskSum;
m2 = m2 ./ maskSum;
m3 = m3 ./ maskSum;
m4 = m4 ./ maskSum;
m5 = m5 ./ maskSum;
m6 = m6 ./ maskSum;

% Number of pyramid levels
numLevels = 5;

% Generate pyramids exactly as specified in the PDF
% Gaussian pyramids for masks
Gm1 = genPyr(m1, 'gauss', numLevels);
Gm2 = genPyr(m2, 'gauss', numLevels);
Gm3 = genPyr(m3, 'gauss', numLevels);
Gm4 = genPyr(m4, 'gauss', numLevels);
Gm5 = genPyr(m5, 'gauss', numLevels);
Gm6 = genPyr(m6, 'gauss', numLevels);

% Laplacian pyramids for images
LI1 = genPyr(P200, 'lap', numLevels);
LI2 = genPyr(dog1, 'lap', numLevels);
LI3 = genPyr(dog2, 'lap', numLevels);
LI4 = genPyr(cat, 'lap', numLevels);
LI5 = genPyr(bench, 'lap', numLevels);
LI6 = genPyr(myImg, 'lap', numLevels);

% Blend pyramids according to PDF equations
B = cell(1, numLevels);
for j = 1:numLevels
    [levelRows, levelCols, ~] = size(LI1{j});
    
    % Resize masks to match current level
    g1 = imresize(Gm1{j}, [levelRows, levelCols]);
    g2 = imresize(Gm2{j}, [levelRows, levelCols]);
    g3 = imresize(Gm3{j}, [levelRows, levelCols]);
    g4 = imresize(Gm4{j}, [levelRows, levelCols]);
    g5 = imresize(Gm5{j}, [levelRows, levelCols]);
    g6 = imresize(Gm6{j}, [levelRows, levelCols]);
    
    % Blend using equation (4) from PDF
    B{j} = g1 .* LI1{j} + g2 .* LI2{j} + g3 .* LI3{j} + g4 .* LI4{j} + g5 .* LI5{j} + g6 .* LI6{j};
end

% Reconstruct final image using equation from PDF
result = pyrReconstruct(B);

% Display result in color (no grayscale stretching)
figure;
imshow(result);
title('Final Composition');
