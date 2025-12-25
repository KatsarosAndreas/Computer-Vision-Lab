function create_receding_ball_animation()
   % Load and prepare images
   ball = im2double(imread('ball.jpg'));
   mask = im2double(rgb2gray(imread('ball_mask.jpg')));
   beach = im2double(imread('beach.jpg'));
   
   % Prepare images
   mask = ~mask;
   beach = imresize(beach, [600 1080]);
   [sceneH, sceneW, ~] = size(beach);
   
   % Video setup
   v = VideoWriter('transf_beach_receding.avi');
   v.FrameRate = 30;
   open(v);
   
   % Animation parameters
   totalFrames = 240;
   startX = sceneW * 0.25;
   endX = sceneW * 0.5;
   startY = sceneH - 100;
   endY = 100;
   startSize = 200;
   endSize = 5;
   
   % Calculate increments
   deltaX = (endX - startX) / totalFrames;
   deltaY = (endY - startY) / totalFrames;
   deltaSize = (endSize - startSize) / totalFrames;
   
   for i = 1:totalFrames
       % Update position and size
       posX = startX + deltaX * i;
       posY = startY + deltaY * i;
       currentSize = max(1, startSize + deltaSize * i);
       
       % Resize current frame elements
       ballResized = imresize(ball, [round(currentSize) round(currentSize)]);
       maskResized = imresize(mask, [round(currentSize) round(currentSize)]);
       
       % Rotate
       angle = i * 1.5;
       ballRotated = imrotate(ballResized, angle, 'bilinear');
       maskRotated = imrotate(maskResized, angle, 'bilinear');
       
       % Calculate ROI
       [rotH, rotW, ~] = size(ballRotated);
       topY = max(1, round(posY - rotH/2));
       leftX = max(1, round(posX - rotW/2));
       botY = min(sceneH, topY + rotH - 1);
       rightX = min(sceneW, leftX + rotW - 1);
       
       % Crop and blend
       roiH = botY - topY + 1;
       roiW = rightX - leftX + 1;
       ballCrop = ballRotated(1:roiH, 1:roiW, :);
       maskCrop = maskRotated(1:roiH, 1:roiW);
       
       frame = beach;
       roi = frame(topY:botY, leftX:rightX, :);
       
       % Blend channels
       for ch = 1:3
           roi(:,:,ch) = ballCrop(:,:,ch).*maskCrop + roi(:,:,ch).*(1-maskCrop);
       end
       
       frame(topY:botY, leftX:rightX, :) = roi;
       writeVideo(v, im2uint8(frame));
   end
   
   close(v);
   implay('transf_beach_receding.avi');
end