m = reshape(permute(model_output,[3,2,1]), [], 1);
t = reshape(permute(true_output,[3,2,1]), [], 1);
figure; hold on;
plot(m);
plot(t);
legend('True audio', 'Predicted audio');

fs = 44100;
% sound(m, fs);


sampling_period = 1/fs;
window_length = length(t) * sampling_period;
freq_extent = fs;
freq_interval = 1/window_length;
freqs = (-fs/2:freq_interval:fs/2-freq_interval)';

M = fftshift(fft(fftshift(m)));

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

m2 = real(ifftshift(ifft(ifftshift(M2))));
sound(m2, fs);
 