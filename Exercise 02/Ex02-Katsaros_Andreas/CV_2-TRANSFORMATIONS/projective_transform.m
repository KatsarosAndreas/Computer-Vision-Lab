%% Task 5: Rotating Windmill Animation with Mask Integration


clear; clc; close all;

% Load images
rotor = imread('windmill.png');          % Windmill blades
mask = imread('windmill_mask.png');      % Mask for blades
bg = imread('windmill_back.jpeg');       % Background image

% Convert to double precision
rotor = im2double(rotor);
mask = im2double(rgb2gray(mask));
bg = im2double(bg);

% Invert the mask
mask = ~mask;


% Find bounding box of the mask
[r, c] = find(mask);
top = min(r); 
bottom = max(r);
left = min(c); 
right = max(c);

% Crop windmill and mask
mask_c = mask(top:bottom, left:right);
rotor_c = rotor(top:bottom, left:right, :);

% Resize for efficiency
new_w = round(size(mask_c, 2) / 2);
new_h = round(size(mask_c, 1) / 2);
mask_r = imresize(mask_c, [new_h, new_w]);
rotor_r = imresize(rotor_c, [new_h, new_w]);
bg_r = imresize(bg, 0.5);

% Background dimensions and offsets
[bg_h, bg_w, ~] = size(bg_r);
y_off = round((bg_h - new_h) / 2);
x_off = round((bg_w - new_w) / 2);


vid = VideoWriter('windmill_rot.avi');
vid.FrameRate = 30;
open(vid);


n_frames = 200;               % Total number of frames
a_step = -360 / n_frames;     % Rotation step per frame

for k = 1:n_frames
    % Current rotation angle
    angle = (k - 1) * a_step;

    % Rotate windmill and mask
    rotor_rot = imrotate(rotor_r, angle, 'bilinear', 'crop');
    mask_rot = imrotate(mask_r, angle, 'nearest', 'crop');

    % Overlay on background
    frame = bg_r; % Start with resized background
    roi = frame(y_off+1:y_off+new_h, x_off+1:x_off+new_w, :);

    % Blend using mask
    for ch = 1:3
        roi(:,:,ch) = rotor_rot(:,:,ch) .* mask_rot + roi(:,:,ch) .* (1 - mask_rot);
    end

    % Insert ROI back into the frame
    frame(y_off+1:y_off+new_h, x_off+1:x_off+new_w, :) = roi;

    % Write frame to video
    writeVideo(vid, im2uint8(frame));

    % Optional display for debugging
    imshow(frame);
    title(sprintf('Frame %d of %d', k, n_frames));
    pause(0.01);
end


close(vid);
disp('Video saved as windmill_rot.avi');
