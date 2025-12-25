% Script for testing ECC and LK image alignment algorithms
clearvars; close all;  % Clear variables and close all figures

% 1. Load video and select frames
video_input = VideoReader('video1_high.avi');
frame_template = readFrame(video_input);
frame_target = readFrame(video_input);

% Normalize the frames to the range [0, 255]
frame_template = 255 * (double(frame_template) - double(min(frame_template(:)))) / (double(max(frame_template(:))) - double(min(frame_template(:))));
frame_target = 255 * (double(frame_target) - double(min(frame_target(:)))) / (double(max(frame_target(:))) - double(min(frame_target(:))));

% Initialize affine warp matrix (identity transformation)
affine_warp = zeros(2,3);
affine_warp(1,1) = 1;
affine_warp(2,2) = 1;

% Experiment 1: Single-level alignment
fprintf('Experiment 1: Single-level alignment\n');
figure('Name', 'Experiment 1');
[result_exp1, lk_result_exp1, mse_exp1, rho_exp1, mse_lk_exp1] = ecc_lk_alignment(frame_target, frame_template, 1, 5, 'affine', affine_warp);

% Experiment 2: Multi-level alignment using pyramids
fprintf('Experiment 2: Multi-level alignment using pyramids\n');
figure('Name', 'Experiment 2');
[result_exp2, lk_result_exp2, mse_exp2, rho_exp2, mse_lk_exp2] = ecc_lk_alignment(frame_target, frame_template, 2, 5, 'affine', affine_warp);

% Experiment 3: Increased iterations for alignment
fprintf('Experiment 3: Increased iterations\n');
figure('Name', 'Experiment 3');
[result_exp3, lk_result_exp3, mse_exp3, rho_exp3, mse_lk_exp3] = ecc_lk_alignment(frame_target, frame_template, 1, 10, 'affine', affine_warp);

% Experiment 4: Skipping frames for greater variation
video_input = VideoReader('video1_low.avi');
frame_template = readFrame(video_input);
for frame_skip = 1:3
    frame_target = readFrame(video_input);
end
frame_target = 255 * (double(frame_target) - double(min(frame_target(:)))) / (double(max(frame_target(:))) - double(min(frame_target(:))));

fprintf('Experiment 4: Skipping frames\n');
figure('Name', 'Experiment 4');
[result_exp4, lk_result_exp4, mse_exp4, rho_exp4, mse_lk_exp4] = ecc_lk_alignment(frame_target, frame_template, 1, 8, 'affine', affine_warp);

% Plot results: PSNR comparison
figure('Name', 'PSNR Comparison Across Experiments');
plot(10 * log10((255^2)./mse_exp1), 'b-', 'DisplayName', 'ECC Exp 1'); hold on;
plot(10 * log10((255^2)./mse_exp2), 'r-', 'DisplayName', 'ECC Exp 2');
plot(10 * log10((255^2)./mse_exp3), 'g-', 'DisplayName', 'ECC Exp 3');
plot(10 * log10((255^2)./mse_exp4), 'k-', 'DisplayName', 'ECC Exp 4');
plot(10 * log10((255^2)./mse_lk_exp1), 'b--', 'DisplayName', 'LK Exp 1');
plot(10 * log10((255^2)./mse_lk_exp2), 'r--', 'DisplayName', 'LK Exp 2');
plot(10 * log10((255^2)./mse_lk_exp3), 'g--', 'DisplayName', 'LK Exp 3');
plot(10 * log10((255^2)./mse_lk_exp4), 'k--', 'DisplayName', 'LK Exp 4');
title('PSNR Comparison');
xlabel('Iterations'); ylabel('PSNR (dB)');
legend('Location', 'Best'); grid on;

% Plot results: Correlation coefficients
figure('Name', 'Correlation Coefficient Comparison');
plot(rho_exp1, 'b-', 'DisplayName', 'ECC Exp 1'); hold on;
plot(rho_exp2, 'r-', 'DisplayName', 'ECC Exp 2');
plot(rho_exp3, 'g-', 'DisplayName', 'ECC Exp 3');
plot(rho_exp4, 'k-', 'DisplayName', 'ECC Exp 4');
title('Correlation Coefficients');
xlabel('Iterations'); ylabel('Correlation Coefficient');
legend('Location', 'Best'); grid on;

% Display final results
fprintf('Final Results:\n');
fprintf('PSNR (dB) - ECC: %.2f, %.2f, %.2f, %.2f\n', 10*log10((255^2)./[mse_exp1(end) mse_exp2(end) mse_exp3(end) mse_exp4(end)]));
fprintf('PSNR (dB) - LK: %.2f, %.2f, %.2f, %.2f\n', 10*log10((255^2)./[mse_lk_exp1(end) mse_lk_exp2(end) mse_lk_exp3(end) mse_lk_exp4(end)]));
fprintf('Correlation Coefficients - ECC: %.4f, %.4f, %.4f, %.4f\n', rho_exp1(end), rho_exp2(end), rho_exp3(end), rho_exp4(end));
fprintf('Correlation Coefficients - LK: %.4f, %.4f, %.4f, %.4f\n', lk_result_exp1(end).rho, lk_result_exp2(end).rho, lk_result_exp3(end).rho, lk_result_exp4(end).rho);