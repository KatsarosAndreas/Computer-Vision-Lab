function create_beach_ball_animation()
    % Load and prepare images
    ball = im2double(imread('ball.jpg'));
    mask = im2double(rgb2gray(imread('ball_mask.jpg')));
    beach = im2double(imread('beach.jpg'));
    
    % Invert and pad mask
    mask = ~mask;
    padSize = 50;
    ball = padarray(ball, [padSize padSize], 'replicate');
    mask = padarray(mask, [padSize padSize], 'replicate');
    
    % Resize images
    ball = imresize(ball, [120 120]);
    mask = imresize(mask, [120 120]);
    beach = imresize(beach, [600 1080]);
    [sceneH, sceneW, ~] = size(beach);
    
    % Store frames for implay
    frames = cell(1, 240);
    
    % Animation parameters
    posX = 120;
    posY = sceneH - 200;
    velY = -12;
    velX = 3;
    angle = 0;
    
    % Setup video
    v = VideoWriter('transf_beach.avi');
    v.FrameRate = 30;
    open(v);
    
    for i = 1:240
        % Update physics
        velY = velY + 0.3;  % Gravity
        posY = posY + velY;
        
        % Bounce check
        if posY > sceneH - 200
            posY = sceneH - 200;
            velY = -abs(velY) * 0.8;  % Bounce with energy loss
        end
        
        % Update position and rotation
        posX = posX + velX;
        angle = angle + 2.5;  % Constant rotation
        
        % Rotate ball and mask
        rotBall = imrotate(ball, angle, 'bilinear');
        rotMask = imrotate(mask, angle, 'bilinear');
        [rotH, rotW, ~] = size(rotBall);
        
        % Calculate ROI (Region of Interest)
        topY = max(1, round(posY - rotH/2));
        leftX = max(1, round(posX - rotW/2));
        botY = min(sceneH, topY + rotH - 1);
        rightX = min(sceneW, leftX + rotW - 1);
        
        % Crop and blend
        roiH = botY - topY + 1;
        roiW = rightX - leftX + 1;
        ballCrop = rotBall(1:roiH, 1:roiW, :);
        maskCrop = rotMask(1:roiH, 1:roiW);
        
        % Create frame and apply blending
        frame = beach;
        roi = frame(topY:botY, leftX:rightX, :);
        
        % Blend each channel
        for ch = 1:3
            roi(:,:,ch) = ballCrop(:,:,ch).*maskCrop + roi(:,:,ch).*(1-maskCrop);
        end
        
        % Update frame with blended region
        frame(topY:botY, leftX:rightX, :) = roi;
        
        % Save frame
        writeVideo(v, im2uint8(frame));
        frames{i} = im2uint8(frame);
    end
    
    % Close video writer
    close(v);
    
    % Display animation
    implay(cat(4, frames{:}), 30);
end