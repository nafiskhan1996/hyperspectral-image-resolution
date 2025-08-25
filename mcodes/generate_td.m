fileFolder=fullfile('test8_n');
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name};
factor = 0.125;
img_size = 512;
bands = 31; 
gt = zeros(numel(fileNames),img_size,img_size,bands);
ms = zeros(numel(fileNames),img_size*factor,img_size*factor,bands);
ms_bicubic = zeros(numel(fileNames),img_size,img_size,bands);
f = '';
cd(fileFolder);
excludedSubstrings = {'fake_and_real_lemon_slices_ms_test.mat', 'fake_and_real_tomatoes_ms_test.mat', 'pompoms_ms_test.mat', 'stuffed_toys_ms_test.mat', 'cd_ms_test.mat', 'hairs_ms_test.mat', 'fake_and_real_beers_ms_test.mat', 'thread_spools_ms_test.mat', 'sponges_ms_test.mat', 'oil_painting_ms_test.mat', 'feathers_ms_test.mat', 'chart_and_stuffed_toy_ms_test.mat', 'jelly_beans_ms_test.mat', 'paints_ms_test.mat', 'glass_tiles_ms_test.mat', 'cloth_ms_test.mat', 'beads_ms_test.mat', 'watercolors_ms_test.mat'};
for i = 1:numel(fileNames)
    loaded_data = load(fileNames{i});
    gt(i,:,:,:) = loaded_data.gt;
    ms(i,:,:,:) = loaded_data.ms;
    ms_bicubic(i,:,:,:) = loaded_data.ms_bicubic;
end
gt = single(gt);
ms = single(ms);
ms_bicubic = single(ms_bicubic);
save('Cave_test.mat','gt','ms','ms_bicubic');