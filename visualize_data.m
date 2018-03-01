train_data_dir = 'data/train';
save_dir = fullfile(train_data_dir, 'inspection');
file_extension = '*.mat';

listing = dir(fullfile(train_data_dir, file_extension));
figure;
for i = 1:length(listing)
    tmp = strsplit(listing(i).name, '.');
    filename = tmp{1};
    % check to see if inspection has been done on this data before
    save_filepath = fullfile(save_dir, filename);
    if length(dir(strcat(save_filepath, '*'))) > 0
        continue;
    end

    %% load the data
    filepath = fullfile(train_data_dir, filename);
    data = load(filepath);
    inputs = data.X;
    outputs = data.Yout;

    %% check for long sustained note errors
    % assumes that the data has 3 dimensions (segment #, time window #, pitch encoding/audio)
    inputs_all = reshape(permute(inputs, [3,2,1]), size(inputs,3), [])';
    outputs_all = reshape(permute(outputs, [3,2,1]), size(outputs,3), [])';
    subplot(1,2,1); image(inputs_all, 'cdatamapping', 'scaled');
    title('Input');
    xlabel('Pitch encoding');
    ylabel('Time window #');
    grid on;
    subplot(1,2,2); image(outputs_all, 'cdatamapping', 'scaled');
    title('Output');
    xlabel('Sample #');
    ylabel('Time window #');
    grid on;
    % save figure
    exportFig(gcf, save_filepath, '-transparent');

    %% check for time window offset errors
    % assumes that any offset errors can be seen within the first 5 segments
    for segment_num = 1:5
        % assumes that the data has 3 dimensions (segment #, time window #, pitch encoding/audio)
        segment_input = squeeze(inputs(segment_num,:,:));
        segment_output = squeeze(outputs(segment_num,:,:));
        % create the figure for inspection
        subplot(1,2,1); image(segment_input, 'cdatamapping', 'scaled');
        % title(sprintf('%s input', filename), 'interpreter', 'none');
        title('Input');
        xlabel('Pitch encoding');
        ylabel('Time window #');
        grid on;
        subplot(1,2,2); image(segment_output, 'cdatamapping', 'scaled');
        % title(sprintf('%s output', filename), 'interpreter', 'none');
        title('Output');
        xlabel('Sample #');
        ylabel('Time window #');
        grid on;
        % save figure
        save_filename_segment = sprintf('%s segment%03d', filename, segment_num);
        save_filepath_segment = fullfile(save_dir, save_filename_segment);
        exportFig(gcf, save_filepath_segment, '-transparent');
    end
end
