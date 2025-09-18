function hr_patches = predict_hr_patches(lr_patches, model)
    % Inputs:
    %   lr_patches - low-resolution patches to predict HR (M x d)
    %   model - trained KRR model structure
    % Output:
    %   hr_patches - predicted high-resolution patches (M x d')

    % Extract model parameters
    alpha = model.alpha;
    lr_train = model.lr_patches;
    % Normalize LR patches (zero mean, unit variance)
    mu_lr = mean(lr_patches, 1);
    std_lr = std(lr_patches, [], 1) + 1e-8; % avoid division by zero
    lr_patches = (lr_patches - mu_lr) ./ std_lr;
    sigma = model.sigma;
    
    % Compute the kernel matrix between input and training patches
    K = exp(-pdist2(lr_patches, lr_train).^2 / (2 * sigma^2));
    
    % Predict the HR patches
    hr_patches = K * alpha;
    hr_patches = hr_patches;



end
