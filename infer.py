import os
import time

from torch.autograd import Variable
import math
import torch

import random
import numpy as np
import numpy
import networks
from my_args import  args

from scipy.misc import imread, imsave
from AverageMeter import  *


# -------------------- SETUP THE NN SETTINGS --------------------------

torch.backends.cudnn.benchmark = True # to speed up the
model = networks.__dict__[args.netName](channel=args.channels,
                            filter_size = args.filter_size ,
                            timestep=args.time_step,
                            training=False)
model = model.cuda() # use CUDA

args.SAVED_MODEL = './model_weights/best.pth'
pretrained_dict = torch.load(args.SAVED_MODEL)

model_dict = model.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
model.load_state_dict(model_dict)
# 4. release the pretrained dict for saving memory
pretrained_dict = []

model = model.eval() # deploy mode
use_cuda=True
save_which=args.save_which
dtype = args.dtype

# -------------------- SETUP THE NN SETTINGS --------------------------

tot_timer = AverageMeter()
proc_timer = AverageMeter()
end = time.time()

OUTPUT_DIR = "./sharedfs/output"
INPUT_DIR = "./sharedfs/input"

print("Reading files in directory...")
frames = os.listdir(INPUT_DIR) 
frames.sort()

if (os.path.isdir(OUTPUT_DIR) != True):
    os.mkdir(OUTPUT_DIR)

frameIndex = 0
for frame in frames:
    if (len(frames) != frameIndex+1):
        FrameOneNr = frame[0:len(frame)-4].split('-')[1]
        FrameTwoNr = frames[frameIndex+1][0:len(frame)-4].split('-')[1]
        FrameI = FrameOneNr + "i" + FrameTwoNr

        # Make a filename conform to the original filename and append the new interpolated number behind it.
        FrameIFileName = frame[0:len(frame)-4].split('-')[0] + '-' + FrameI + ".png"
        print(FrameIFileName)

        # Set the frames to be used and the output frame
        arguments_strFirst = os.path.join(INPUT_DIR, frame)
        arguments_strSecond = os.path.join(INPUT_DIR, frames[frameIndex+1])
        arguments_strOut = os.path.join(OUTPUT_DIR, FrameIFileName)

        # prepare frames for interpolation
        X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype)
        X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)

        y_ = torch.FloatTensor()

        assert (X0.size(1) == X1.size(1))
        assert (X0.size(2) == X1.size(2))

        intWidth = X0.size(2)
        intHeight = X0.size(1)
        channel = X0.size(0)
        if not channel == 3:
            continue

        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft =int(( intWidth_pad - intWidth)/2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 32
            intPaddingRight= 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

        torch.set_grad_enabled(False)
        X0 = Variable(torch.unsqueeze(X0,0))
        X1 = Variable(torch.unsqueeze(X1,0))
        X0 = pader(X0)
        X1 = pader(X1)

        X0 = X0.cuda()
        X1 = X1.cuda()

        proc_end = time.time()
        y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
        y_ = y_s[save_which]

        proc_timer.update(time.time() -proc_end)
        tot_timer.update(time.time() - end)
        end  = time.time()
        print("*****************current image process time \t " + str(time.time()-proc_end )+"s ******************" )

        X0 = X0.data.cpu().numpy()
        y_ = y_.data.cpu().numpy()
        offset = [offset_i.data.cpu().numpy() for offset_i in offset]
        filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
        X1 = X1.data.cpu().numpy()

        X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
        y_ = np.transpose(255.0 * y_.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
        offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
        filter = [np.transpose(
            filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
            (1, 2, 0)) for filter_i in filter]  if filter is not None else None
        X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

        #save the resulting image
        imsave(arguments_strOut, np.round(y_).astype(numpy.uint8))
    
    frameIndex = frameIndex + 1
