import pickle
import torch
from scipy.io import wavfile
import argparse
import numpy as np

from model_gated import PedalNet
import os


def save(name, data,sr=4410):
    wavfile.write(name, sr, data.flatten().astype(np.float32))


@torch.no_grad()
def test(args):
    model = PedalNet.load_from_checkpoint(args.model)
    model.eval()

    if args.data == "Default":
        data_path = os.path.dirname(args.model) + "/data.pickle"
    else:
        data_path = args.data
    data = pickle.load(open(data_path, "rb"))

    x_test = data["x_test"]
    prev_sample = np.concatenate((np.zeros_like(x_test[0:1]), x_test[:-1]), axis=0)
    pad_x_test = np.concatenate((prev_sample, x_test), axis=2)

    y_pred = []
    for x in np.array_split(pad_x_test, 10):
        y_pred.append(model(torch.from_numpy(x)).numpy())

    y_pred = np.concatenate(y_pred)
    y_pred = y_pred[:, :, -x_test.shape[2] :]

    save(os.path.dirname(args.model) + "/y_pred.wav", y_pred,args.sr)
    if args.store_all:
        save(os.path.dirname(args.model) + "/x_test.wav", data["x_test"] * data["std"] + data["mean"],args.sr)
        save(os.path.dirname(args.model) + "/y_test.wav", data["y_test"],args.sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pedalnet/pedalnet.ckpt") #model log directory (with best.pth)
    parser.add_argument("--data", default="Default")
    parser.add_argument("--sr",type=int,default=44100) #sampling rate
    parser.add_argument("--store-all",type=bool,default=True,help="If true store all waveform, else only y_pred")



    args = parser.parse_args()
    test(args)
