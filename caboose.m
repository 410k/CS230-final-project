% script to resample midi files to different time window durations

% setup
time_window_duration = 0.005;           % seconds
filename_extension = sprintf('_dt%02d', time_window_duration*1e3);
save_dir = filename_extension;
mkdir(save_dir);

midi_dir = fullfile('TPD', 'classical');
midi_listing = dir(fullfile(midi_dir, '*.mid'));

keyboard_lowest_midi_note = 21;
num_keyboard_keys = 88;

tic;
for n = 1:length(midi_listing)
    % load the standard midi file
    midi_filename = midi_listing(n).name;
    fprintf('processing %s\n', midi_filename);
    midi_filepath = fullfile(midi_dir, midi_filename);
    midi = readmidi(midi_filepath);

    % parse the midi and output a matrix indicating when each note is on
    [notes, end_time] = midiInfo(midi, 0, 1);
    [piano_notes_on, time_steps, midi_notes] = piano_roll(notes, 0, time_window_duration);
    keyboard_keys = midi_notes - keyboard_lowest_midi_note + 1;
    % piano_notes_on: (midi_notes, time_steps)
    % time_steps: seconds
    % keyboard_keys: array of keyboard keys played

    % find note start indices (articulated)
    articulate = conv2(piano_notes_on, [1,-1]);
    articulate = articulate(:,1:end-1);
    articulate( articulate > 0 ) = 1;
    articulate( articulate <= 0 ) = 0;
    sustain = zeros(size(piano_notes_on));
    sustain( piano_notes_on > 0 ) = 1;
    
    % create the input matrix [time_windows, pitch_encoding (articulate & sustain)*88 keys on piano]
    Xin = zeros(size(sustain,2), num_keyboard_keys*2);
    for i = 1:num_keyboard_keys
        sustain_index = i*2;
        articulate_index = sustain_index-1;
        keyboard_key_index = find(ismember(keyboard_keys, i));
        if isempty(keyboard_key_index)
            continue;
        end
        Xin(:, articulate_index) = articulate(keyboard_key_index,:)';   % input articulation into odd columns
        Xin(:, sustain_index) = sustain(keyboard_key_index,:)';         % input sustain into even columns
    end

    % save the files
    save_filename = sprintf('%s%s.mat', midi_filename(1:end-4), filename_extension);
    save_filepath = fullfile(save_dir, save_filename);
    save(save_filepath, 'Xin');
end
toc;
