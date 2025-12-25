% Task 2: Linear Scaling and Placement of Image (Pudding)

% Clear workspace
clear; clc; close all;

% Load the image
input_image = imread('pudding.png');

% Create a blank canvas (black background)
canvas = uint8(zeros(500, 1500, 3)); % Canvas dimensions: 500x1500

% Define scaling factors and parameters
scaling_factors = [0.3, 0.5, 0.7, 0.9, 1.2];
start_x = 50; % Initial X position
spacing = 150; % Space between scaled images

% Initialize current position
current_x = start_x;

% Loop through scaling factors and compose the canvas
for scale = scaling_factors
    % Scale the image
    scaled_image = imresize(input_image, scale);
    
    % Get scaled image dimensions
    [rows, cols, ~] = size(scaled_image);
    
    % Calculate vertical position (centered)
    y_position = round((size(canvas, 1) - rows) / 2);
    
    % Place the scaled image on the canvas
    canvas(y_position:y_position+rows-1, current_x:current_x+cols-1, :) = scaled_image;
    
    % Update horizontal position for the next image
    current_x = current_x + cols + spacing;
end

% Display and save the final composite image
imshow(canvas);
imwrite(canvas, 'linear_scaled_puddings.png');
title('Linear Scaling of Pudding Images');
