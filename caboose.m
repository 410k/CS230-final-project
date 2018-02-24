% data read in
tic
d = dir('/Users/granty/Documents/CS230/mp3/');
for ii = 4:length(d)
    
    %midi = readmidi('/Users/granty/Documents/CS230/midi/classical/Adagio in B flat.mid');
     midi = readmidi(['/Users/granty/Documents/CS230/midi/classical/' d(ii).name(1:end-4) '.mid']);
    %[aud,fs] = audioread('/Users/granty/Documents/CS230/mp3/Adagio in B flat.mp3');
    [aud,fs] = audioread(['/Users/granty/Documents/CS230/mp3/' d(ii).name ]);
    aud = decimate(aud,4)';
    fs = fs/4;
    [Notes, endtime] = midiInfo(midi,1,1);
    dt = 0.01; %seconds
    
    % PR(midi note, time), t = time index (s), nn = midi note index
    [PR,t,nn] = piano_roll(Notes,0,dt);
    
    % find note start indices (Articulated)
    attack = conv2(PR,[1,-1]);
    attack = attack(:,1:end-1);
    attack( attack > 0 ) = 1;
    attack( attack <= 0 ) = 0;
    sustain = zeros(size(PR));
    sustain( PR > 0 ) = 1;
    
    % input matrix ( sample, time, features)
    time = 1; % time duration in seconds
    N = 1/dt; % number of data points in time
 
    % Reshape audio segments
    % crop audio to same size as Xin
    Na = fs/time; % number of audio samples in time duration
    num_samples = ceil(length(aud)/Na);
    y = zeros(num_samples*Na,1);
    y(1:length(aud)) = aud;
    Yout = reshape(y,Na,num_samples)'; %output matrix
    
    time_audio = length(y)/fs;
    
    % input matrix (time, pitch (keys 1-88 on the piano)
    Xin = zeros(size(sustain,2),176);
    for i = 1:length(nn)
        Xin(:,(nn(i)-20)*2-1) = attack(i,:)'; % input articulation into odd columns
        Xin(:,(nn(i)-20)*2) = sustain(i,:)'; % input sustain into even columns
    end
    
    
    % crop Xin
    X = zeros(num_samples*N,176); % zero pad last sample
    X(1:size(Xin,1),:) = Xin;
    X = permute(reshape(X',176,N,num_samples),[3,2,1]);
%     

    
%     test = pianoRoll2matrix(PR,dt,nn);
%     y = midi2audio(matrix2midi(test),fs);
%     sound(y,fs)
%     
%     test = Yout';
%     sound(test(:),fs)
    save(['/Users/granty/Data/CS230_train/keras_ex' num2str(ii) '.mat'],'X','Yout');
    
end
toc
