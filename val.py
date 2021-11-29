import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter

import pickle
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_id', default=None,
                        help='run id')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.run_id, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'])

    var_save_dir = 'variables'
    var_name = 'test_loader_'+str(args.run_id)+'.pkl'
    path = os.path.join(var_save_dir, var_name)
    with open(path, 'rb') as file:
      # Call load method to deserialze
      test_loader = pickle.load(file)

    model_path_retrieve = 'models/%s/model.pth' % args.run_id
    model.load_state_dict(torch.load(model_path_retrieve))
    model = model.to(device)

    model.eval()
    print("Testing...")
    test_loss = 0.0
    test_iou = 0

    avg_meter = AverageMeter()

    os.makedirs(os.path.join('outs', args.run_id), exist_ok=True)

    # define loss function (criterion)
    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        for batch, (input, target, _) in enumerate(test_loader):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            # need to pass output through sigmoid function to use BCE loss correctly :https://towardsdatascience.com/cuda-error-device-side-assert-triggered-c6ae1c8fa4c3
            loss = criterion(torch.sigmoid(torch.squeeze(output)), torch.squeeze(target))

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            output = np.squeeze(torch.tensor(torch.sigmoid(output)>0.5).cpu().numpy())
            target = np.squeeze(target.cpu().numpy())            
            input = np.squeeze(input.cpu().numpy()).transpose((1,2,0))

            cv2.imwrite(os.path.join('outs', args.run_id, 'result_'+str(batch) + '.jpg'),
                        (output * 255).astype('uint8'))
            cv2.imwrite(os.path.join('outs', args.run_id, 'target_'+str(batch) + '.jpg'),
                        (target * 255).astype('uint8'))
            cv2.imwrite(os.path.join('outs', args.run_id, 'input_'+str(batch) + '.jpg'),
                        (input * 255).astype('uint8'))
            
            test_loss += loss.item()
            test_iou += iou
        num_batch = len(test_loader)
        test_loss /= num_batch
        test_iou /= num_batch

    print('IoU: %.4f' % test_iou)
    print('loss: %.4f' % test_loss)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()