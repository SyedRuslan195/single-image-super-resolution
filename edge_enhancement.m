function enhanced_img = edge_enhancement(img, enhancement_factor)
    if enhancement_factor == 0
        enhanced_img = img;
        return;
    end

    % Define Laplacian kernel for edge detection
    laplacian_kernel = [0 -1 0; -1 4 -1; 0 -1 0];
    
    % Initialize the enhanced image with the original image
    enhanced_img = img;

    % Loop over each color channel
    for i = 1:size(img, 3)
        % Extract the channel
        channel = img(:,:,i);
        
        % Detect edges using Laplacian filter
        edges = conv2(channel, laplacian_kernel, 'same');
        
        % Enhance the image by adding a fraction of the edges
        enhanced_channel = channel + enhancement_factor * edges;
        
        % Ensure the enhanced channel stays within valid bounds [0, 1]
        %enhanced_channel = min(max(enhanced_channel, 0), 1);
        
        % Store the enhanced channel back
        enhanced_img(:,:,i) = enhanced_channel;
    end
end
