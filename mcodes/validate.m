%% Step 4: Please manually remove 10% of the samples to the folder of evals
%% Step 4: Automatically move 10% of the training samples to the "evals" folder
trainDir = '../Cave_x4/trains/train';  % Directory containing training samples
evalsDir = '../Cave_x4/evals/evals';  % Directory to store evaluation samples
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