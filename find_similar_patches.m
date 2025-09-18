function similar_patches = find_similar_patches(patch, patches, k, distance_metric)
    % Inputs:
    %   patch - the reference patch (1 x d)
    %   patches - all available patches (N x d)
    %   k - number of similar patches to find
    %   distance_metric - 'euclidean' or 'ssim'
    % Output:
    %   similar_patches - indices of the k most similar patches

    if nargin < 4
        distance_metric = 'euclidean';
    end

    if strcmp(distance_metric, 'euclidean')
        distances = sum((patches - patch).^2, 2);
    elseif strcmp(distance_metric, 'ssim')
        distances = arrayfun(@(x) -ssim(reshape(patch, sqrt(numel(patch)), sqrt(numel(patch))), reshape(patches(x, :), sqrt(numel(patch)), sqrt(numel(patch)))),1:size(patches, 1));
    else
        error('Unsupported distance metric');
    end

    [~, idx] = sort(distances);
    similar_patches = idx(1:k);
end
