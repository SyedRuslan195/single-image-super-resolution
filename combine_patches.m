function hr_img = combine_patches(hr_patches, positions, img_size, patch_size, scale)
    % Inputs:
    %   hr_patches - high-resolution patches (M x d')
    %   positions - top-left corner positions of each patch (M x 2)
    %   img_size - size of the original LR image (should be in the form [rows, cols, channels])
    %   patch_size - size of each patch
    %   overlap - overlap between patches
    %   scale - scaling factor from LR to HR
    % Output:
    %   hr_img - reconstructed high-resolution image

    rows = img_size(1);
    cols = img_size(2);
    if length(img_size) == 3
        channels = img_size(3);
    else
        channels = 1; % Assuming grayscale if no third dimension
    end

    hr_rows = rows * scale;
    hr_cols = cols * scale;
    hr_img = zeros(hr_rows, hr_cols, channels);
    weight = zeros(hr_rows, hr_cols, channels);

    num_patches = size(hr_patches, 1);

    for i = 1:num_patches
        row = positions(i, 1) * scale - scale + 1;
        col = positions(i, 2) * scale - scale + 1;
        patch = reshape(hr_patches(i, :), [patch_size * scale, patch_size * scale, channels]);

        hr_img(row:row + patch_size * scale - 1, col:col + patch_size * scale - 1, :) = hr_img(row:row + patch_size * scale - 1, col:col + patch_size * scale - 1, :) + patch;

        weight(row:row + patch_size * scale - 1, col:col + patch_size * scale - 1, :) = weight(row:row + patch_size * scale - 1, col:col + patch_size * scale - 1, :) + 1;
    end

    % Normalize the aggregated patches
    hr_img = hr_img ./ weight;
end
