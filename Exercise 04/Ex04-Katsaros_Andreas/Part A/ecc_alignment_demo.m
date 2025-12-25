
clearvars; close all;

% Configuration parameters
gapL = 20;          % Frame gap in low resolution sequence
gapH = 40;          % Frame gap in high resolution sequence
iterationCount = 15; % Number of iterations for alignment

% --- Low resolution sequence (64x64) ---
disp('Testing low resolution sequence...');

% Read initial frames from low-res video
inVid = VideoReader('video1_low.avi');
refFrame = readFrame(inVid);
refFrame = 255 * ( double(refFrame) - double(min(refFrame(:))) ) / ...
                ( double(max(refFrame(:))) - double(min(refFrame(:))) );

% Advance video by gapL frames
for idx = 1:gapL
    if hasFrame(inVid)
        movFrame = readFrame(inVid);
    end
end
movFrame = 255 * ( double(movFrame) - double(min(movFrame(:))) ) / ...
                ( double(max(movFrame(:))) - double(min(movFrame(:))) );

% Initialize affine warp matrix
warpMat = zeros(2,3);
warpMat(1,1) = 1; 
warpMat(2,2) = 1;

% Run ECC-LK alignment on low resolution data
[eccLo, lkLo, errLo, corrLo, errlkLo] = ecc_lk_alignment(movFrame, refFrame, 1, iterationCount, 'affine', warpMat);

% --- High resolution sequence (256x256) ---
disp('Testing high resolution sequence...');

% Read initial frames from high-res video
inVid = VideoReader('video1_high.avi');
refFrame = readFrame(inVid);
refFrame = 255 * ( double(refFrame) - double(min(refFrame(:))) ) / ...
                ( double(max(refFrame(:))) - double(min(refFrame(:))) );

% Advance video by gapH frames
for idx = 1:gapH
    if hasFrame(inVid)
        movFrame = readFrame(inVid);
    end
end
movFrame = 255 * ( double(movFrame) - double(min(movFrame(:))) ) / ...
                ( double(max(movFrame(:))) - double(min(movFrame(:))) );

% Re-initialize affine warp matrix
warpMat = zeros(2,3);
warpMat(1,1) = 1; 
warpMat(2,2) = 1;

% Run ECC-LK alignment on high resolution data
[eccHi, lkHi, errHi, corrHi, errlkHi] = ecc_lk_alignment(movFrame, refFrame, 1, iterationCount, 'affine', warpMat);

% --- Visualization of results ---
figure('Name', 'Performance Comparison');

% PSNR comparison
subplot(2,2,1);
hold on;
plot(20 * log10(255 ./ errLo), 'b-', 'DisplayName', 'Low Res ECC');
plot(20 * log10(255 ./ errlkLo), 'b--', 'DisplayName', 'Low Res LK');
plot(20 * log10(255 ./ errHi), 'r-', 'DisplayName', 'High Res ECC');
plot(20 * log10(255 ./ errlkHi), 'r--', 'DisplayName', 'High Res LK');
title('PSNR Comparison');
xlabel('Iterations'); ylabel('PSNR (dB)');
legend('Location', 'best'); grid on;

% Correlation comparison
subplot(2,2,2);
hold on;
plot(corrLo, 'b-', 'DisplayName', 'Low Res ECC');
plot(corrHi, 'r-', 'DisplayName', 'High Res ECC');
title('Correlation Coefficients');
xlabel('Iterations'); ylabel('Correlation');
legend('Location', 'best'); grid on;

% Display template and final warped for low resolution
subplot(2,2,3);
montage({uint8(refFrame), uint8(eccLo(end,end).image)}, 'Size', [1 2]);
title('Low Resolution: Template and Final Alignment');

% Display template and final warped for high resolution
subplot(2,2,4);
montage({uint8(refFrame), uint8(eccHi(end,end).image)}, 'Size', [1 2]);
title('High Resolution: Template and Final Alignment');

% --- Numerical results ---
disp('Results Summary:');

disp('Low Resolution:');
fprintf('Final PSNR (ECC): %.2f dB\n', 20 * log10(255 / errLo(end)));
fprintf('Final PSNR (LK): %.2f dB\n', 20 * log10(255 / errlkLo(end)));
fprintf('Final Correlation: %.4f\n', corrLo(end));

disp('High Resolution:');
fprintf('Final PSNR (ECC): %.2f dB\n', 20 * log10(255 / errHi(end)));
fprintf('Final PSNR (LK): %.2f dB\n', 20 * log10(255 / errlkHi(end)));
fprintf('Final Correlation: %.4f\n', corrHi(end));
