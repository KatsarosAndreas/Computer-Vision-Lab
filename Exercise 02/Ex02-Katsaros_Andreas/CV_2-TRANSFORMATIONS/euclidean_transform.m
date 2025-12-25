% Read the image
img = imread('pudding.png');

% Get image dimensions
[rows, cols, ~] = size(img);

% Create reference object for the image
ref = imref2d(size(img));

% Number of frames for the animation
numFrames = 60;

% Initialize cell array to store transformed images
transformedImages = cell(1, numFrames);

% Parameters for periodic shearing
maxShear = 0.5;  % Maximum shearing amount
period = 2*pi;   % Full period of the sine wave

% Create periodic horizontal shearing animation with fixed base
for i = 1:numFrames
    % Calculate shearing factor using sine function for smooth periodicity
    % Only horizontal shearing (shearX), no vertical shearing
    t = (i-1)/numFrames;  % Normalized time from 0 to 1
    shearX = maxShear * sin(period * t);
    
    % Create affine transformation matrix for horizontal-only shearing
    % The matrix is designed to keep the base fixed by adjusting the translation
    tform = affine2d([1    shearX  0;
                      0    1       0;
                      0    0       1]);
    
    % Apply the transformation
    transformedImages{i} = imwarp(img, tform, 'OutputView', ref);
end

% Convert cell array to 4D array for video
videoArray = cat(4, transformedImages{:});

% Create and save the video file
v = VideoWriter('sheared_pudding.avi');  % You can change extension to .mp4 if needed
v.FrameRate = 30;  % 30 fps for smooth playback
open(v);
writeVideo(v, videoArray);
close(v);

% Display the animation (optional)
figure('Name', 'Horizontal Shearing Animation');
player = implay(videoArray, 30);

% Optionally add controls to the video player
player.Visual.setPropertyValue('Colormap', gray(256));
player.Visual.setPropertyValue('ColorbarVisible', false);

% Function to preview a single frame (useful for debugging)
function previewFrame(img, frameNum)
    figure;
    imshow(img);
    title(['Frame ' num2str(frameNum)]);
end