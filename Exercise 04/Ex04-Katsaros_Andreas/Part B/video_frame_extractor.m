% Read the RGB image
img = imread('peppers.png');

% Replicate the image to create a video of 30 frames
numFrames = 30;
videoFrames = repmat(img, 1, 1, 1, numFrames);

% Save the variable in the base workspace
assignin('base', 'videoFrames', videoFrames);
