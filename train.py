import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import sys
import time
import os

#from model_relu import PedalNet
from prepare import prepare

# python train_relu.py data/ts9_test1_in_FP32.wav data/ts9_test1_out_FP32.wav --cpu --max_epochs 1

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
    elif args.model_type=="model_relu_stripped":
        from model_relu_stripped import PedalNet
    elif args.model_type=="model_relu_betterskip":
        from model_relu_betterskip import PedalNet
    elif args.model_type=="model_relu_clamp":
        from model_relu_clamp import PedalNet
    elif args.model_type=="model_relu_ai8x":
        from model_relu_ai8x import PedalNet
    elif args.model_type=="model_gated":
        from model_gated import PedalNet
    elif args.model_type=="model":
        from model import PedalNet
    else:
        print("Invalid model type")
 
    args.name="data"
    prepare(args)
    print(vars(args))
    model = PedalNet(vars(args))
    version = gen_timestamp_name()
    version += "_"+args.model.split('/')[-1].split('.')[-2]
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name="",version=version)

    # use checkpoint callback to store BEST model according to validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.dirname(args.model),
        filename="test-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1, #only store the best model
        mode="min",
        period = 1 #args.save_epochs #save model every n epochs
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=args.model if args.resume else None,
        gpus=None if args.cpu or args.tpu_cores else args.gpus,
        tpu_cores=args.tpu_cores,
        log_every_n_steps=50,#100,
        logger=logger,
        max_epochs=args.max_epochs,
    )

    trainer.fit(model)
    trainer.save_checkpoint(args.model)

    #get best model etc
    print("Best model:", checkpoint_callback.best_model_score,"  saved at:", checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", nargs="?", default="data/in.wav")
    parser.add_argument("out_file", nargs="?", default="data/out.wav")
    parser.add_argument("--sample_time", type=float, default=100e-3)

    parser.add_argument("--in_bit_depth", type=str, default="None") #input quantization

    parser.add_argument("--num_channels", type=int, default=12)
    parser.add_argument("--dilation_depth", type=int, default=10)
    parser.add_argument("--dilation_power",type=int,default=2)
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

    parser.add_argument("--save_epochs", type=int, default="0",help="Save model every n epoch. Don't save if 0")

    #preparation
    parser.add_argument("--max_test_samples", type=int, default=None, help="forces max test samples, debug")
    parser.add_argument("--skip_silence_samples",action='store_true',default=False,help="Skip train and validation samples where sum(abs(x)) is less then len(x)*sil_treshold"),
    parser.add_argument("--skip_silence",type=int,default=0,help="Skip portion of dataset where both x and y train are zero and 'distant' skip_silence samples from any nonzero value ")
    parser.add_argument("--sil_treshold",type=float,default=5e-3,help="Silence treshold, range 0,1")

    args = parser.parse_args()
    main(args)
