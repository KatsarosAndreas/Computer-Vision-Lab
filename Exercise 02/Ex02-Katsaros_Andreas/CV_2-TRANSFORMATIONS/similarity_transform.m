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

% Create periodic shearing animation
for i = 1:numFrames
    % Use complex shearing pattern
    t = (i-1)/numFrames;  % Time parameter from 0 to 1
    tform = createComplexShearing(t, 0.3);
    
    % Apply the transformation
    transformedImages{i} = imwarp(img, tform, 'OutputView', ref);
end

% Convert cell array to 4D array for implay
videoArray = cat(4, transformedImages{:});

% Create the video player
figure('Name', 'Pudding Shearing Animation');
player = implay(videoArray, 30); % 30 fps playback speed

% Optional: Save the animation as a video file
v = VideoWriter('pudding_shearing.avi');
v.FrameRate = 30;
open(v);
writeVideo(v, videoArray);
close(v);

% Function to create a more complex shearing pattern (optional)
function tform = createComplexShearing(t, amplitude)
    % t: time parameter (0 to 1)
    % amplitude: maximum shearing amount
    
    % Combine multiple sinusoidal components for more interesting motion
    shearX = amplitude * (sin(2*pi*t) + 0.5*sin(4*pi*t));
    shearY = amplitude * (cos(2*pi*t) + 0.5*cos(4*pi*t));
    
    % Create transformation matrix
    tform = affine2d([1    shearX  0;
                      shearY   1    0;
                      0       0     1]);
end