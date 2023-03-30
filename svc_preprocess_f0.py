import os
import numpy as np
import torch
from scipy.io import wavfile
from crepe.model import CrepeInfer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file = 'crepe/assets/full.pth'
crepeInfer = CrepeInfer(file, device)

if __name__ == "__main__":
    os.makedirs("filelists", exist_ok=True)
    files = open("./filelists/train.txt", "w", encoding="utf-8")

    rootPath = "./data_svc/waves/"
    outPath = "./data_svc/pitch/"
    os.makedirs(outPath, exist_ok=True)

    for file in os.listdir(f"./{rootPath}"):
        if file.endswith(".wav"):
            file = file[:-4]
            wav_path = f"./{rootPath}//{file}.wav"
            featur_pit = crepeInfer.compute_f0(wav_path)

            np.save(
                f"{outPath}//{file}.nsf",
                featur_pit,
                allow_pickle=False,
            )

            path_spk = "./data_svc/lora_speaker.npy"
            path_wave = f"./data_svc/waves//{file}.wav"
            path_pitch = f"./data_svc/pitch//{file}.nsf.npy"
            path_whisper = f"./data_svc/whisper//{file}.ppg.npy"
            print(
                f"{path_wave}|{path_pitch}|{path_whisper}|{path_spk}",
                file=files,
            )

    files.close()
