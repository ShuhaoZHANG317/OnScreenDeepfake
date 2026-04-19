"""
eval pretained model.
"""
import os
import json
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
# from dataset.ff_blend import FFBlendDataset
# from dataset.fwa_blend import FWABlendDataset
# from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score


import argparse
from logger import create_logger

'''
parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str,
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
# parser.add_argument("--lmdb", action='store_true', default=False)
'''
parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str,
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
parser.add_argument("--dataset_json_dir", type=str, default=None,
                    help="directory containing dataset json files")
parser.add_argument("--test_dataset_prefix", type=str, default=None,
                    help="only keep test datasets whose names start with this prefix")
parser.add_argument("--output_dir", type=str, default=None,
                    help="directory to save test metrics")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode='test',
        )
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config['test_batchSize'],
            shuffle=False,
            num_workers=int(config['workers']),
            collate_fn=test_set.collate_fn,
            drop_last=False
        )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
            data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)

        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())

    return np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)


def aggregate_to_video_level(preds, labels, img_names):
    video_dict = {}

    for p, l, name in zip(preds, labels, img_names):
        # 假设路径类似：
        # xxx/real/00009/frame_000057.png
        video_id = "/".join(name.split("/")[:-1])

        if video_id not in video_dict:
            video_dict[video_id] = {"preds": [], "labels": []}

        video_dict[video_id]["preds"].append(p)
        video_dict[video_id]["labels"].append(l)

    video_preds = []
    video_labels = []

    for v in video_dict.values():
        assert len(set(v["labels"])) == 1
        video_preds.append(np.mean(v["preds"]))   # 平均分数
        video_labels.append(v["labels"][0])       # 同一视频标签一致

    return np.array(video_preds), np.array(video_labels)


def test_epoch(model, test_data_loaders):
    model.eval()
    metrics_all_datasets = {}

    keys = test_data_loaders.keys()
    for key in keys:
        predictions_nps, label_nps, feat_nps = test_one_dataset(model, test_data_loaders[key])

        frame_pred_bin = (predictions_nps > 0.5).astype(int)

        if len(np.unique(label_nps)) > 1:
            frame_auc = roc_auc_score(label_nps, predictions_nps)
        else:
            frame_auc = 0.5

        frame_acc = accuracy_score(label_nps, frame_pred_bin)

        metrics_all_datasets[key] = {
            "frame_level": {
                "auc": frame_auc,
                "acc": frame_acc
            }
        }

        tqdm.write(f"dataset: {key}")
        tqdm.write("---- Frame-level ----")
        tqdm.write(f"auc: {frame_auc:.4f}")
        tqdm.write(f"acc: {frame_acc:.4f}")

    return metrics_all_datasets



@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']

    weights_path = None

    # If arguments are provided, they will overwrite the yaml settings
    '''
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    '''
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset

    if args.dataset_json_dir:
        config['dataset_json_folder'] = args.dataset_json_dir

    if args.test_dataset_prefix:
        all_json_names = []
        for fn in os.listdir(config['dataset_json_folder']):
            if fn.endswith(".json"):
                dataset_name = os.path.splitext(fn)[0]
                if dataset_name.startswith(args.test_dataset_prefix):
                    all_json_names.append(dataset_name)
        all_json_names = sorted(all_json_names)
        config['test_dataset'] = all_json_names

    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0

        # weights_only=False is required for loading model checkpoints with state_dict
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')

    # start testing
    '''
    best_metric = test_epoch(model, test_data_loaders)
    print('===> Test Done!')
    '''
    best_metric = test_epoch(model, test_data_loaders)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(best_metric, f, indent=2)
        print(f"===> Metrics saved to {out_path}")

    print('===> Test Done!')


if __name__ == '__main__':
    main()
