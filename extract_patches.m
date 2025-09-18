function [patches, positions] = extract_patches(img, patch_size, overlap)
    % Extract overlapping patches from the image
    % Inputs:
    %   img - the input image
    %   patch_size - size of each patch (e.g., 7 for a 7x7 patch)
    %   overlap - overlap between patches
    % Outputs:
    %   patches - matrix of patches where each row is a patch
    %   positions - position of the top-left corner of each patch

    [rows, cols, channels] = size(img);
    step = (patch_size-overlap);
    % Calculate the number of patches along rows and columns
    num_patches_x = ceil((rows - patch_size) / step) + 1;
    num_patches_y = ceil((cols - patch_size) / step) + 1;
    
    % Preallocate the matrices for patches and positions
    patches = zeros(num_patches_x * num_patches_y, patch_size * patch_size * channels, 'like', img);
    positions = zeros(num_patches_x * num_patches_y, 2);
    
    % Initialize the index for storing patches and positions
    patch_idx = 1;
    
    % Loop through the image with the specified overlap
    for i = 1:step:(rows - patch_size + 1)
        for j = 1:step:(cols - patch_size + 1)
            % Extract the patch
            patch = img(i:i + patch_size - 1, j:j + patch_size - 1, :);
            
            % Store the patch in the preallocated matrix
            patches(patch_idx, :) = patch(:)';
            
            % Store the position in the preallocated matrix
            positions(patch_idx, :) = [i, j];
            
            % Increment the index
            patch_idx = patch_idx + 1;
        end
    end
    
    % Remove any unused preallocated space if the loops didn't fill the matrices completely
    patches = patches(1:patch_idx - 1, :);
    positions = positions(1:patch_idx - 1, :);
end
