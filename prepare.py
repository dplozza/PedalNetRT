import argparse
import pickle
from scipy.io import wavfile
import numpy as np
import os


def prepare(args):
    in_rate, in_data = wavfile.read(args.in_file)
    out_rate, out_data = wavfile.read(args.out_file)
    assert in_rate == out_rate, "in_file and out_file must have same sample rate"

    #NORMALIZE in range -1,1  depending on format!!!
    if in_data.max()>2**8:
        in_data = in_data*(2**-15)
        out_data = out_data*(2**-15)

    sample_size = int(in_rate * args.sample_time)
    length = len(in_data) - len(in_data) % sample_size

    in_data = in_data[:length]
    out_data = out_data[:length]

    if args.skip_silence > 0:
        #idea: we want to skip samples that have at least skip_silence of samples of silence in both in and out, between the sample and the closest NOnzero sample
        #implement taht using box filter

        in_data_nonzero = np.abs(in_data) > args.sil_treshold
        out_data_nonzero = np.abs(out_data) > args.sil_treshold

        io_data_nonzero = in_data_nonzero | out_data_nonzero

        #kernel = np.ones(smooth) / smooth
        #y_pred_qat_filt = np.convolve(y_pred_qat_filt.flatten(),kernel,mode="same")
        skip_silence = args.skip_silence*2
        kernel = np.ones(skip_silence) / skip_silence
        io_data_nonzero_filt = np.convolve(io_data_nonzero,kernel,mode='same')
        io_data_nonzero_filt = io_data_nonzero_filt>0

        in_data_skipz = in_data[io_data_nonzero_filt]
        out_data_skipz = out_data[io_data_nonzero_filt]

        print("Lenght before skipz:",in_data.size)
        print("Lenght after skipz:",in_data_skipz.size)
        print("Percentage of original dataset:",np.around((in_data_skipz.size/in_data.size)*100,2),"%")
        
        in_data = in_data_skipz
        out_data = out_data_skipz
        length = len(in_data) - len(in_data) % sample_size


    x = in_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)
    y = out_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)

    split = lambda d: np.split(d, [int(len(d) * 0.6), int(len(d) * 0.8)])

    d = {}
    d["x_train"], d["x_valid"], d["x_test"] = split(x)
    d["y_train"], d["y_valid"], d["y_test"] = split(y)
    d["mean"], d["std"] = d["x_train"].mean(), d["x_train"].std()


    #skip samples where x is silent
    if args.skip_silence_samples:
        is_zero = np.sum(np.abs(d["x_train"]),axis=2).flatten() < args.sil_treshold*sample_size
        d["x_train"] = np.delete(d["x_train"],is_zero,0)
        d["y_train"] = np.delete(d["y_train"],is_zero,0)

        is_zero = np.sum(np.abs(d["x_valid"]),axis=2).flatten() < args.sil_treshold*sample_size
        d["x_valid"] = np.delete(d["x_valid"],is_zero,0)
        d["y_valid"] = np.delete(d["y_valid"],is_zero,0)



    #forces max test samples (for debug)
    if args.max_test_samples is not None:
        d["x_test"] = d["x_test"][:args.max_test_samples,:,:]
        d["y_test"] = d["y_test"][:args.max_test_samples,:,:]

    #log
    print("Prepared data")
    print("Train shape:",d["x_train"].shape)
    print("Validation shape:",d["x_valid"].shape)
    print("Test shape:",d["x_test"].shape)

    # standardize
    for key in "x_train", "x_valid", "x_test":
        d[key] = (d[key] - d["mean"]) / d["std"]

    if not os.path.exists(os.path.dirname(args.model)):
        os.makedirs(os.path.dirname(args.model))

    pickle.dump(d, open(os.path.dirname(args.model) + "/"+args.name+".pickle", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")

    parser.add_argument("--model", type=str, default="models/pedalnet/pedalnet.ckpt")
    parser.add_argument("--name", type=str, default="data")
    parser.add_argument("--sample_time", type=float, default=100e-3)
    parser.add_argument("--max_test_samples", type=int, default=None, help="forces max test samples, debug")
    parser.add_argument("--skip_silence_samples",action='store_true',default=False,help="Skip train and validation samples where sum(abs(x)) is less then len(x)*sil_treshold"),
    parser.add_argument("--skip_silence",type=int,default=0,help="Skip portion of dataset where both x and y train are zero and 'distant' skip_silence samples from any nonzero value ")
    parser.add_argument("--sil_treshold",type=float,default=5e-3,help="Silence treshold, range 0,1")
    args = parser.parse_args()
    prepare(args)
