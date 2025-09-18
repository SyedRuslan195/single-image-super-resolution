function model = train_krr_model(lr_patches, hr_patches, lambda, sigma)
    % Inputs:
    %   lr_patches - low-resolution patches (N x d)
    %   hr_patches - high-resolution patches (N x d')
    %   lambda - regularization parameter
    %   sigma - parameter for the Gaussian kernel
    % Output:
    %   model - trained KRR model structure
    % Compute pairwise squared distances
    D = pdist2(lr_patches, lr_patches, 'euclidean');
    
    % Take median of non-zero distances
    sigma = sigma.*median(D(D > 0));

    % Convert to double precision
    lr_patches = double(lr_patches);
    hr_patches = double(hr_patches);
    % Normalize LR patches (zero mean, unit variance)
    mu_lr = mean(lr_patches, 1);
    std_lr = std(lr_patches, [], 1) + 1e-8; % avoid division by zero
    lr_patches = (lr_patches - mu_lr) ./ std_lr;
    
    % Normalize HR patches
    mu_hr = mean(hr_patches, 1);
    std_hr = std(hr_patches, [], 1) + 1e-8;
    hr_patches = (hr_patches - mu_hr) ./ std_hr;


    N = size(lr_patches, 1);
    
    % Compute the kernel matrix
    K = exp(-pdist2(lr_patches, lr_patches).^2 / (2 * sigma^2));
    
    % Regularize the kernel matrix
    K = K + lambda * eye(N);
    
    % Compute the alpha coefficients
    alpha = K \ hr_patches;

    % Store the model
    model.alpha = alpha;
    model.lr_patches = lr_patches;
    model.sigma = sigma;
    model.mu_lr = mu_lr;
    model.std_lr = std_lr;
    model.mu_hr = mu_hr;
    model.std_hr = std_hr;


end
