% Parent folder containing subdirectories with image data
parentFolder = 'entire/complete_ms_data/watercolors_ms'; % Change this to your parent folder path

% List all subdirectories in the parent folder
subdirectories = dir(parentFolder);

destFolder = 'entire/stack';

% Exclude the '.' and '..' entries
subdirectories = subdirectories(arrayfun(@(x) x.name(1), subdirectories) ~= '.');

% Loop through each subdirectory (child folder)
for subdirIndex = 1:numel(subdirectories)
    % Initialize an empty 3D array to store the images
    imageStack = zeros(512, 512, 0); % Initialize as an empty array

    % Get the current subdirectory (child folder)
    currentSubdir = subdirectories(subdirIndex).name;

    % Specify the folder containing PNG images in the current subdirectory
    imageFolder = fullfile(parentFolder, currentSubdir);

    % List all PNG image files in the current subdirectory
    imageFiles = dir(fullfile(imageFolder, '*.png')); % Modify the file extension as needed

    % Loop through each image and read them into the array
    for i = 1:numel(imageFiles)
        % Read the PNG image
        imagePath = fullfile(imageFolder, imageFiles(i).name);
        [~,~,image] = imread(imagePath);


        % Concatenate the image along the third dimension
        imageStack = cat(3, imageStack, image);
    end
    name = [currentSubdir, 'imageStack.mat'];
    % Save the 3D array as a .mat file in the current subdirectory
    save(fullfile(destFolder, name), 'imageStack');
end
