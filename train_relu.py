import pytorch_lightning as pl

import argparse
import sys
import time

#from model_relu import PedalNet
from prepare import prepare

# python train2.py data/ts9_test1_in_FP32.wav data/ts9_test1_out_FP32.wav --cpu --max_epochs 1

def gen_timestamp_name() -> str:
    """generate a timestap to use as filename"""
    secondsSinceEpoch = time.time() # Get the seconds since epoch 
    timeObj = time.localtime(secondsSinceEpoch) # Convert seconds since epoch to struct_time
    name = '%04d.%02d.%02d-%02d%02d%02d' % (
    timeObj.tm_year, timeObj.tm_mon, timeObj.tm_mday, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
    return name

def main(args):
    """
    This trains the PedalNet model to match the output data from the input data.

    When you resume training from an existing model, you can override hparams such as
        max_epochs, batch_size, or learning_rate. Note that changing num_channels,
        dilation_depth, num_repeat, or kernel_size will change the shape of the WaveNet
        model and is not advised.

    """

    if args.model_type=="model_relu":
        from model_relu import PedalNet
    elif args.model_type=="model_relu_skipless":
        from model_relu_skipless import PedalNet
    elif args.model_type=="model":
        from model import PedalNet
    else:
        print("Invalid model type")
 

    prepare(args)
    print(vars(args))
    model = PedalNet(vars(args))
    version = gen_timestamp_name()
    version += "_"+args.model.split('/')[-1].split('.')[-2]
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name="",version=version)
    trainer = pl.Trainer(
        resume_from_checkpoint=args.model if args.resume else None,
        gpus=None if args.cpu or args.tpu_cores else args.gpus,
        tpu_cores=args.tpu_cores,
        log_every_n_steps=50,#100,
        logger=logger,
        max_epochs=args.max_epochs,
    )

    trainer.fit(model)
    trainer.save_checkpoint(args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", nargs="?", default="data/in.wav")
    parser.add_argument("out_file", nargs="?", default="data/out.wav")
    parser.add_argument("--sample_time", type=float, default=100e-3)

    parser.add_argument("--in_bit_depth", type=str, default="None")#input quantization

    parser.add_argument("--num_channels", type=int, default=12)
    parser.add_argument("--dilation_depth", type=int, default=10)
    parser.add_argument("--num_repeat", type=int, default=1)
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-3)

    parser.add_argument("--max_epochs", type=int, default=1500)
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--tpu_cores", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--model", type=str, default="models/pedalnet/pedalnet.ckpt")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--model_type", type=str, default="model_relu")

    args = parser.parse_args()
    main(args)
