%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% match_SIFT_RANSAC_minimal.m
% Απαιτεί: Computer Vision Toolbox (detectSIFTFeatures, extractFeatures κ.λπ.)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% 1. Φόρτωση & Προεπεξεργασία Εικόνων

imgA = imread('fruit.jpg');
imgB = imread('fruit_rotated.png');

% Προαιρετικά, μετατροπή σε grayscale (αν οι εικόνες είναι έγχρωμες).
if size(imgA,3) == 3
    grayA = rgb2gray(imgA);
else
    grayA = imgA;
end

if size(imgB,3) == 3
    grayB = rgb2gray(imgB);
else
    grayB = imgB;
end

%Κλιμάκωση σε κοινή ανάλυση, π.χ. 256×256:
grayA = im2double(imresize(grayA, [256, 256]));
grayB = im2double(imresize(grayB, [256, 256]));

%% 2. Εντοπισμός Σημείων & Εξαγωγή Περιγραφέων (SIFT)
keyptsA = detectSIFTFeatures(grayA);
keyptsB = detectSIFTFeatures(grayB);

[descsA, validA] = extractFeatures(grayA, keyptsA);
[descsB, validB] = extractFeatures(grayB, keyptsB);

fprintf('Βρέθηκαν %d χαρακτηριστικά στην εικόνα A.\n', size(descsA,1));
fprintf('Βρέθηκαν %d χαρακτηριστικά στην εικόνα B.\n', size(descsB,1));

%% 3. Αντιστοίχιση των Descriptors (Ratio Test κ.λπ.)
matchesIdx = matchFeatures(descsA, descsB, ...
                           'MaxRatio', 0.8, ...
                           'Unique', true);

matchedA = validA(matchesIdx(:,1));
matchedB = validB(matchesIdx(:,2));

fprintf('Αρχικός αριθμός αντιστοιχιών: %d\n', numel(matchedA));

%% 4. Απόρριψη Λανθασμένων Αντιστοιχίσεων (RANSAC)
% Εκτίμηση projective transform (ομογραφίας) με RANSAC, π.χ. 'MaxDistance' = 3.
[tform, inlierMask] = estimateGeometricTransform2D(...
    matchedA, matchedB, 'projective', 'MaxDistance', 3);

inliersA = matchedA(inlierMask);
inliersB = matchedB(inlierMask);

fprintf('Inliers μετά το RANSAC: %d\n', numel(inliersA));

%% 5. Οπτικοποίηση Αντιστοιχιών (Inliers)
figure;
showMatchedFeatures(grayA, grayB, inliersA, inliersB, 'montage');
title('Αντιστοιχίσεις SIFT με RANSAC (Inliers/Outliers Removed)');
