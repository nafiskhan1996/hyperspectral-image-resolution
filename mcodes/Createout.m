% Parent folder containing subdirectories with image data
parentFolder = 'entire/complete_ms_data'; % Change this to your parent folder path

% List all subdirectories in the parent folder
subdirectories = dir(parentFolder);

% Exclude the '.' and '..' entries
subdirectories = subdirectories(arrayfun(@(x) x.name(1), subdirectories) ~= '.');

destFolder = 'output8';
excludedSubstrings = {'fake_and_real_lemon_slices_msimageStack.mat', 'fake_and_real_tomatoes_msimageStack.mat', 'pompoms_msimageStack.mat', 'stuffed_toys_msimageStack.mat', 'cd_msimageStack.mat', 'hairs_msimageStack.mat', 'fake_and_real_beers_msimageStack.mat', 'thread_spools_msimageStack.mat', 'sponges_msimageStack.mat', 'oil_painting_msimageStack.mat', 'feathers_msimageStack.mat', 'chart_and_stuffed_toy_msimageStack.mat', 'jelly_beans_msimageStack.mat', 'paints_msimageStack.mat', 'glass_tiles_msimageStack.mat', 'cloth_msimageStack.mat', 'beads_msimageStack.mat', 'watercolors_msimageStack.mat'};

% Loop through each subdirectory (child folder)
for subdirIndex = 1:numel(subdirectories)
    % Get the current subdirectory (child folder)
    currentSubdir = subdirectories(subdirIndex).name;
    if ~any(contains(currentSubdir, {'desktop.ini'}))
        % Define the source folder for PNG files in the current subdirectory
        sourceFolder = fullfile(parentFolder, currentSubdir, currentSubdir);
    
        % List all PNG files in the source folder
        pngFiles = dir(fullfile(sourceFolder, 'imageStack.mat'));
    
        fileNames = {pngFiles.name};
        factor = 0.125;
        img_size = 512;
        bands = 31;
        gt = zeros(numel(fileNames), img_size, img_size, bands);
        ms = zeros(numel(fileNames), img_size * factor, img_size * factor, bands);
        ms_bicubic = zeros(numel(fileNames), img_size, img_size, bands);
        
        cd(sourceFolder);
        for i = 1:numel(fileNames)
            load(fileNames{i}, 'imageStack');
            imageStack = mat2gray(imageStack);
            img_ms = single(imresize(imageStack, factor));
            gt(i, :, :, :) = imageStack;
            ms(i, :, :, :) = img_ms;
            ms_bicubic(i, :, :, :) = single(imresize(img_ms, 1 / factor));
        end
        cd ../../../../;
        gt = single(gt);
        ms = single(ms);
        ms_bicubic = single(ms_bicubic);
        
        % Save the processed data in the current subdirectory
        save(fullfile(destFolder, [currentSubdir, '_test.mat']), 'gt', 'ms', 'ms_bicubic');
    end
end
