%% visualize the outputs
m = reshape(permute(model_output,[3,2,1]), [], 1);
t = reshape(permute(true_output,[3,2,1]), [], 1);
figure; hold on;
plot(m);
plot(t);
legend('True audio', 'Predicted audio');
xlabel('Sample #');
ylabel('Amplitude');


%% fourier transform the audio
fs = 44100;
sampling_period = 1/fs;
window_length = length(t) * sampling_period;
freq_extent = fs;
freq_interval = 1/window_length;
freqs = (-fs/2:freq_interval:fs/2-freq_interval)';

M = fftshift(fft(fftshift(m)));


%% bandpass filter the audio
freq_cutoff_low = 25;       % Hz
freq_cutoff_high = 4200;    % Hz
unwanted_freq_inds = abs(freqs) < freq_cutoff_low | freq_cutoff_high < abs(freqs);
M2 = M;
M2(unwanted_freq_inds) = 0;

figure; 
subplot(2,1,1); plot(freqs, log(abs(M)));
ylabel('Log(abs(amplitude))');
grid on;
subplot(2,1,2); plot(freqs, log(abs(M2)));
xlim([freqs(1), freqs(end)]);
xlabel('Frequency (Hz)');
ylabel('Log(abs(amplitude))');
grid on;


%% fourier transform back to time domain
m2 = real(ifftshift(ifft(ifftshift(M2))));


%% listen to the audio
% sound(m, fs);
sound(m2, fs);
 