import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
def npytomp4(npy):
    spec = npy.numpy() #tensor를 numpy로 변환
    print("spec", spec.shape)
    sr = 22050
    res = librosa.feature.inverse.mel_to_audio(spec, 
                                            sr=sr, 
                                            n_fft=512, 
                                            hop_length=256,
                                            win_length=None, 
                                            window='hann', 
                                            center=True, 
                                            pad_mode='reflect', 
                                            power=2.0, n_iter=256)
    import soundfile as sf
    sf.write("C:/Users/ICT/Desktop/youda/AI_TeamRepo/test.wav", res, sr)
# specs = np.load('C:/Users/ICT/Desktop/youda/AI_TeamRepo/testDatasetv8/angry/0.npy')
# specs = np.load(f"C:/Users/ICT/Desktop/youda/AI_TeamRepo/test/test_2.npy")
# spec = specs.astype(np.float64)
# print("spec: ", spec.shape)

