clear;
filepath = 'data/train';
target_file = 'ex15.mat';
load(fullfile(filepath, target_file));

segment_length = 100;


%% pad the input & output data
input = Xin;
output = Yout;
num_time_windows = size(input,1);
pad_amount = segment_length - mod(num_time_windows, segment_length);
input = padarray(input, pad_amount, 0, 'post');
output = padarray(output, pad_amount, 0, 'post');


%% save every segment in its own file
num_segments = ceil(size(input,1) / segment_length);
for i = 1:num_segments
    inds = (i-1)*segment_length+1:i*segment_length;
    Xin = input(inds, :);
    Yout = output(inds, :);
    filename = sprintf('ex15_%02d.mat', i);
    save_path = fullfile(filepath, filename);
    save(save_path, 'Xin', 'Yout');
end
