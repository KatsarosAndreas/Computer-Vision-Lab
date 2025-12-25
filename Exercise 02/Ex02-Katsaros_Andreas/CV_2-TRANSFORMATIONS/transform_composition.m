    %% Task 6: Comparing Interpolation Methods for Rotating Windmill
    
    clear; clc; close all;
    
    % Load images
    rotor = imread('windmill.png');          % Windmill blades
    mask = imread('windmill_mask.png');      % Mask for blades
    bg = imread('windmill_back.jpeg');       % Background image
    
    % Convert to double precision
    rotor = im2double(rotor);
    mask = im2double(rgb2gray(mask));
    bg = im2double(bg);
    
    % Invert the mask to select the rotor region
    mask = ~mask;
    
    % Find bounding box of the mask
    [r, c] = find(mask);
    top = min(r); bottom = max(r);
    left = min(c); right = max(c);
    
    % Crop the rotor and mask
    mask_c = mask(top:bottom, left:right);
    rotor_c = rotor(top:bottom, left:right, :);
    
    % Resize for efficiency
    new_w = round(size(mask_c, 2) / 2);
    new_h = round(size(mask_c, 1) / 2);
    mask_r = imresize(mask_c, [new_h, new_w]);
    rotor_r = imresize(rotor_c, [new_h, new_w]);
    bg_r = imresize(bg, 0.5);
    
    [bg_h, bg_w, ~] = size(bg_r);
    y_off = round((bg_h - new_h) / 2);
    x_off = round((bg_w - new_w) / 2);
    
    % Define interpolation methods to compare
    interp_methods = {'nearest', 'bilinear', 'bicubic'};
    output_names = {'windmill_rot_nearest.avi', 'windmill_rot_linear.avi', 'windmill_rot_cubic.avi'};
    
    % Total number of frames and angle step
    n_frames = 200;
    a_step = -360 / n_frames;
    
    for m = 1:length(interp_methods)
        interp_method = interp_methods{m};
        vid_name = output_names{m};
        
        vid = VideoWriter(vid_name);
        vid.FrameRate = 30;
        open(vid);
        
        for k = 1:n_frames
            angle = (k - 1) * a_step;
            
            % Rotate using chosen interpolation for the rotor
            % Mask remains with nearest to preserve binary nature
            rotor_rot = imrotate(rotor_r, angle, interp_method, 'crop');
            mask_rot = imrotate(mask_r, angle, 'nearest', 'crop');
            
            % Overlay on background
            frame = bg_r;
            roi = frame(y_off+1:y_off+new_h, x_off+1:x_off+new_w, :);
    
            % Blend using mask
            for ch = 1:3
                roi(:,:,ch) = rotor_rot(:,:,ch) .* mask_rot + roi(:,:,ch) .* (1 - mask_rot);
            end
            
            frame(y_off+1:y_off+new_h, x_off+1:x_off+new_w, :) = roi;
    
            writeVideo(vid, im2uint8(frame));
            
            % Optional visualization
            % imshow(frame);
            % title(sprintf('%s: Frame %d of %d', interp_method, k, n_frames));
            % pause(0.01);
        end
        
        close(vid);
        fprintf('Video saved as %s using %s interpolation.\n', vid_name, interp_method);
    end
        