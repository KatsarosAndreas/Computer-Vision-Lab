% Paths for input images and the functions
photoDir = 'C:\Users\AndreasK\Documents\MATLAB\Exercise 1\Matlab_Ex1\photos\';
functionDir = 'C:\Users\AndreasK\Documents\MATLAB\Exercise 1\Matlab_Ex1\';
% Add the function directory to the MATLAB path
addpath(functionDir);

% Load the images
imageA = im2double(imread([photoDir, 'apple.jpg']));
imageB = im2double(imread([photoDir, 'orange.jpg']));

% Resize images to be of the same dimensions
[dimRows, dimCols, ~] = size(imageB);
imageA = imresize(imageA, [dimRows, dimCols]);

% Define the number of pyramid levels
pyrLevels = 5;

%% Generate Gaussian pyramids for both images
gaussA = genPyr(imageA, 'gauss', pyrLevels);
gaussB = genPyr(imageB, 'gauss', pyrLevels);

%% Generate Laplacian pyramids for both images
lapA = genPyr(imageA, 'lap', pyrLevels);
lapB = genPyr(imageB, 'lap', pyrLevels);

%% Visualization: Gaussian and Laplacian Pyramids
% Gaussian pyramid visualization for imageA
figure;
for idx = 1:pyrLevels
    subplot(2, pyrLevels, idx);
    imshow(gaussA{idx}, []);
    title(['Gaussian (Level ', num2str(idx), ')']);
    
    subplot(2, pyrLevels, idx + pyrLevels);
    imshow(lapA{idx} + 0.5, []); % Offset added for visualization
    title(['Laplacian (Level ', num2str(idx), ')']);
end
sgtitle('ImageA: Gaussian and Laplacian Pyramids');

% Gaussian pyramid visualization for imageB
figure;
for idx = 1:pyrLevels
    subplot(2, pyrLevels, idx);
    imshow(gaussB{idx}, []);
    title(['Gaussian - ImageB (Level ', num2str(idx), ')']);
    
    subplot(2, pyrLevels, idx + pyrLevels);
    imshow(lapB{idx} + 0.5, []); % Offset added for visualization
    title(['Laplacian - ImageB (Level ', num2str(idx), ')']);
end
sgtitle('ImageB: Gaussian and Laplacian Pyramids');

%% Create Gaussian mask
blendMask = zeros(dimRows, dimCols, 3);
blendMask(:, 1:floor(dimCols/2), :) = 1; % Left half for imageA, right half for imageB
gaussMask = genPyr(blendMask, 'gauss', pyrLevels);

%% Blending: 
% Blend using the Laplacian pyramids and Gaussian mask
blendPyr = cell(1, pyrLevels);
for idx = 1:pyrLevels
    % Resize the Gaussian mask to match the Laplacian pyramid level
    [levelRows, levelCols, ~] = size(lapA{idx});
    resizedBlendMask = imresize(gaussMask{idx}, [levelRows, levelCols]);
    
    % Perform blending for the current level
    blendPyr{idx} = resizedBlendMask .* lapA{idx} + (1 - resizedBlendMask) .* lapB{idx};
end

% Reconstruct the blended image
resultBlend = pyrReconstruct(blendPyr);

% Feathered blend: Blend directly without pyramids (shitty)
directBlend = blendMask .* imageA + (1 - blendMask) .* imageB;

%% Visualization: =
figure;
subplot(1, 2, 1);
imshow(resultBlend);
title('Good Blend (Pyramid)');

subplot(1, 2, 2);
imshow(directBlend);
title('Other Blend (Feathered with Line)');
