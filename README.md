# Voice Spectrogram
声谱图

此次学习如何利用librosa提取声音的声谱图
声谱图（spectrogram）是声音或其他信号的频率随时间变化时的频谱（spectrum）的一种直观表示

当数据以三维图形表示时，可称其为瀑布图（waterfalls）

读取数据采用`librosa.load()`返回读取后的强度序列，也可以返回采样率，采样率参数`sr`默认为22050 Sa/s

通过短时傅里叶变换，
The STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows.

调用`librosa.stft(data)`，便对`data`做短时傅里叶处理，其中`n_fft`默认为2048，`window="hann"`. The number of rows in the STFT matrix is (1 + n_fft/2).

采用`librosa.amplitude_to_db(data,ref=np.max)`，将振幅化成以dB为单位的形式

the amplitude ``abs(S)`` is scaled relative to ``ref``: ``20 * log10(data / ref)``.

在二维数组中，第一个轴是频率，第二个轴是时间。我们使用librosa.display.specshow来显示声谱图

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

data, sampling_rate = librosa.load('./train_sample/aloe/24EJ22XBZ5.wav')

plt.figure(figsize=(14, 5))
librosa.display.waveplot(data, sr=sampling_rate)

plt.figure(figsize=(20, 10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
```
![2.png](https://github.com/Cocytus-Leon/FoodVoiceRecognition_2/blob/main/1.png)
![2.png](https://github.com/Cocytus-Leon/FoodVoiceRecognition_2/blob/main/2.png)
