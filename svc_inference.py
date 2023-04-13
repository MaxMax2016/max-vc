import os
import torch
import argparse
import numpy as np
import librosa

from scipy.io.wavfile import write
from omegaconf import OmegaConf
from model.generator import Generator
from crepe.model import CrepeInfer


def load_svc_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["model_g"])
    return model


ppg_path = "svc_tmp.ppg.npy"


def main(args):
    os.system(f"python svc_inference_ppg.py -w {args.wave} -p {ppg_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crepeModel = 'crepe/assets/full.pth'
    crepeInfer = CrepeInfer(crepeModel, device)
    audio, _ = librosa.load(args.wave, sr=16000)
    pit = crepeInfer.compute_f0(audio)
    del crepeInfer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hp = OmegaConf.load(args.config)
    model = Generator(hp)
    load_svc_model(args.model, model)

    ppg = np.load(ppg_path)
    pos = [1, 2]
    pos = np.tile(pos, ppg.shape[0])
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    ppg = torch.FloatTensor(ppg)

    pit = torch.FloatTensor(pit)
    pos = torch.LongTensor(pos)

    spk = np.load(args.spk)
    spk = torch.FloatTensor(spk)

    len_pit = pit.size()[0]
    len_ppg = ppg.size()[0]
    len_min = min(len_pit, len_ppg)
    pit = pit[:len_min, :]
    ppg = ppg[:len_min, :]
    pos = pos[:len_min]

    model.eval(inference=True)
    model.to(device)
    with torch.no_grad():
        spk = spk.unsqueeze(0).to(device)
        ppg = ppg.unsqueeze(0).to(device)
        pos = pos.unsqueeze(0).to(device)
        pit = pit.unsqueeze(0).to(device)
        audio = model.inference(spk, ppg, pos, pit)
        audio = audio.cpu().detach().numpy()

    write("svc_out.wav", hp.audio.sampling_rate, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="path of model for evaluation")
    parser.add_argument('-w', '--wave', type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument('-s', '--spk', type=str, required=True,
                        help="Path of speaker.")
    args = parser.parse_args()

    main(args)
