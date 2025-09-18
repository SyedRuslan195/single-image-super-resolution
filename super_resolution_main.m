close all;
clear all;

% Parameters
patch_size = 11; % Size of LR patch
trainingOverlap = 3;
overlap = 9; % Overlap between patches
scale = 2; % Scaling factor from LR to HR
lambda = 0.5e-7; % Regularization parameter for KRR
sigma = 10; % Kernel width for Gaussian kernel
enhancement_factor = 0.25; % Edge enhancement factor (0 for no enhancement)
num_similar_patches = 5; % Number of similar patches to find
use_self_similarity = false; % Toggle self-similarity use

% Load and resize images
disp('Loading and Resizing Images');
B = imread("Original-peppers-image.png");
C = imread("Cheetah.png");
D = im2double(imresize(C, [256, 256], 'bicubic'));
A = im2double(imresize(B, [512, 512], 'bicubic')); % Convert to double
X = imresize(A, [256, 256], 'bicubic'); % Low-resolution image
Y = A; % High-resolution ground truth

% Extract patches
disp('Extracting Patches');
%training
[lr_patches, lr_positions] = extract_patches(X, patch_size, trainingOverlap);
[hr_patches, hr_positions] = extract_patches(Y, patch_size * scale, trainingOverlap * scale);
%testing
m = mean(D,'all');
s = std(D,[0],'all');
[lr_patches2, lr_positions2] = extract_patches(D, patch_size, overlap);

% Super-Resolution without Self-Similarity
%training
disp('Training KRR Model without Self-Similarity');
model_no_sim = train_krr_model(lr_patches, hr_patches, lambda, sigma);
%testing
disp('Predicting High-Resolution Patches without Self-Similarity');
predicted_hr_patches_no_sim = predict_hr_patches(lr_patches2, model_no_sim);
disp('Combining Patches into High-Resolution Image without Self-Similarity');
hr_img_no_sim = combine_patches(predicted_hr_patches_no_sim, lr_positions2, size(D), patch_size, scale);

% Apply Edge Enhancement (if enhancement_factor > 0)
if enhancement_factor > 0
    disp('Enhancing Edges without Self-Similarity');
    enhanced_img_no_sim = edge_enhancement(hr_img_no_sim, enhancement_factor);
else
    disp('Skipping Edge Enhancement without Self-Similarity');
    enhanced_img_no_sim = hr_img_no_sim;
end

% Super-Resolution with Self-Similarity
if use_self_similarity
    disp('Training KRR Model with Self-Similarity');
    %training
    similar_hr_patches = zeros(size(hr_patches));
    
    for i = 1:size(lr_patches, 1)
        % Find similar patches for each LR patch
        similar_indices = find_similar_patches(lr_patches(i, :), lr_patches, num_similar_patches, 'euclidean');
        
        % Average the corresponding HR patches of the similar LR patches
        similar_hr_patches(i, :) = mean(hr_patches(similar_indices, :), 1);
    end
    
    model_with_sim = train_krr_model(lr_patches, similar_hr_patches, lambda, sigma);
    disp('Predicting High-Resolution Patches with Self-Similarity');
    %testing
    predicted_hr_patches_with_sim = predict_hr_patches(lr_patches2, model_with_sim);
    disp('Combining Patches into High-Resolution Image with Self-Similarity');
    hr_img_with_sim = combine_patches(predicted_hr_patches_with_sim, lr_positions2, size(D), patch_size, scale);

    % Apply Edge Enhancement (if enhancement_factor > 0)
    if enhancement_factor > 0
        disp('Enhancing Edges with Self-Similarity');
        enhanced_img_with_sim = edge_enhancement(hr_img_with_sim, 1.3);
    else
        disp('Skipping Edge Enhancement with Self-Similarity');
        enhanced_img_with_sim = hr_img_with_sim;
    end
end

% Display Results

% Original Image
figure;
imshow(C);
title('Original Image');

figure;
subplot(1, 2, 1);
imshow(D);
title('Low-Resolution Image');
% Bicubic Upsampled Image
bicubic_img = imresize(D, [512, 512], 'bicubic');
subplot(1, 2, 2);
imshow(bicubic_img);
title('Bicubic Upsampled Image');

% Super-Resolution without Self-Similarity
figure;
subplot(1, 2, 1);
imshow(s*((hr_img_no_sim-mean(hr_img_no_sim,'all','omitnan'))/std(hr_img_no_sim,[0],'all','omitnan'))+m);
title('Super-resolution, No Post-Processing');

subplot(1, 2, 2);
imshow(s*((enhanced_img_no_sim-mean(enhanced_img_no_sim,'all','omitnan'))/std(enhanced_img_no_sim,[0],'all','omitnan'))+m);
%imshow(enhanced_img_no_sim);
title('Super-resolution, Enhanced');
if use_self_similarity
    % Super-Resolution with Self-Similarity
    figure;
    subplot(1, 2, 1);
    imshow(s*((hr_img_with_sim-mean(hr_img_with_sim,'all','omitnan'))/std(hr_img_with_sim,[0],'all','omitnan'))+m);
    title('SR with Self-Similarity, No Post-Processing');
    
    subplot(1, 2, 2);
    imshow(s*((enhanced_img_with_sim-mean(enhanced_img_with_sim,'all','omitnan'))/std(enhanced_img_with_sim,[0],'all','omitnan'))+m);
    title('SR with Self-Similarity, Enhanced');
end