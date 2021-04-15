# %%
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
# %%
voice_path = './train_sample'


def look_data():
    print('音频文件夹个数：{}'.format(len(os.listdir(voice_path))))
    voice_total = 0
    single_label = {}
    for ind, label_name in enumerate(os.listdir(voice_path)):
        file_path = voice_path+'/'+label_name
        single_num = len(os.listdir(file_path))
        single_label[label_name] = single_num
        voice_total += single_num
    print('音频文件的总量：{}'.format(voice_total))
    print(f'{"序号":<5}{"类别":<15}{"数量":<10}{"占比"}')
    for ind, (key, value) in enumerate(single_label.items()):
        print(f'{ind:<5}{key:<20}{value:<10}{value / voice_total:.2%}')


# %%
look_data()
# %%
data1, sampling_rate1 = librosa.load('./train_sample/aloe/24EJ22XBZ5.wav')
# %%
data2, sampling_rate2 = librosa.load('./train_sample/burger/0WF1KDZVPZ.wav')
# %%
plt.figure(figsize=(14, 5))
librosa.display.waveplot(data1, sr=sampling_rate1)
# %%
plt.figure(figsize=(14, 5))
librosa.display.waveplot(data2, sr=sampling_rate2)
# %%
plt.figure(figsize=(20, 10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(data1)), ref=np.max)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram of aloe')
# %%
plt.figure(figsize=(20, 10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(data2)), ref=np.max)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram of burger')
# %%
