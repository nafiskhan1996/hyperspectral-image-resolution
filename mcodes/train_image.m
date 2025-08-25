% Parent folder containing subdirectories with image data
parentFolder = 'Mat/entire/stack'; % Change this to your parent folder path
%cd('Mat/entire/stack');
stackfiles = dir(fullfile(parentFolder, '*.mat'));
excludedSubstrings = {'fake_and_real_lemon_slices_msimageStack.mat', 'fake_and_real_tomatoes_msimageStack.mat', 'pompoms_msimageStack.mat', 'stuffed_toys_msimageStack.mat', 'cd_msimageStack.mat', 'hairs_msimageStack.mat', 'fake_and_real_beers_msimageStack.mat', 'thread_spools_msimageStack.mat', 'sponges_msimageStack.mat', 'oil_painting_msimageStack.mat', 'feathers_msimageStack.mat', 'chart_and_stuffed_toy_msimageStack.mat', 'jelly_beans_msimageStack.mat', 'paints_msimageStack.mat', 'glass_tiles_msimageStack.mat', 'cloth_msimageStack.mat', 'beads_msimageStack.mat', 'watercolors_msimageStack.mat'};

for subdirIndex = 1:numel(stackfiles)
    filename = stackfiles(subdirIndex).name;
    containsExcluded = any(contains(filename, excludedSubstrings));
    if ~containsExcluded
        factor = 0.125;
        img_size = 512;
        bands = 31;
        load(filename, 'imageStack');
        crop_image(imageStack, 64, 32, 0.125, filename);
    end
end
%% Step 4: Please manually remove 10% of the samples to the folder of evals
%% Step 4: Automatically move 10% of the training samples to the "evals" folder
trainDir = '../Cave_x8/trains/train';  % Directory containing training samples
evalsDir = '../Cave_x8/evals/evals';  % Directory to store evaluation samples
evalPercentage = 0.10;  % Percentage of samples to move to "evals" folder

% List all files in the training directory
trainFiles = dir(fullfile(trainDir, '*.mat'));

% Calculate the number of samples to move to "evals"
numSamplesToMove = round(evalPercentage * numel(trainFiles));

% Randomly select samples to move
randomIndices = randperm(numel(trainFiles), numSamplesToMove);

% Create the "evals" folder if it doesn't exist
if ~exist(evalsDir, 'dir')
    mkdir(evalsDir);
end

% Move selected samples to the "evals" folder
for i = 1:numel(randomIndices)
    idx = randomIndices(i);
    sourceFile = fullfile(trainDir, trainFiles(idx).name);
    destinationFile = fullfile(evalsDir, trainFiles(idx).name);
    movefile(sourceFile, destinationFile);
end

%% Display a message to confirm the move
fprintf('Moved %d samples to the "evals" folder.\n', numSamplesToMove);