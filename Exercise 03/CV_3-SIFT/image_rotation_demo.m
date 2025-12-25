% Φόρτωση cameraman
original = imread('fruit.jpg');

% Περιστροφή κατά 10 μοίρες
rotated = imrotate(original, 10, 'crop'); % 'crop' ή 'bicubic' κλπ.

% Αποθήκευση
imwrite(rotated, 'fruit_rotated.png');
