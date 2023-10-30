import os
import numpy as np
import json
import math
from torch.utils.data import Dataset
import logging
import torch

TRAIN_DATA = [
    "20221001_4139131_2overs",
    "20221119_4139084_2overs",
    "20221203_4139091_2overs",
    "20221203_4142743_2overs",
    "20221203_4139088_2overs",
    "20221203_4139162_2overs",
    "20221203_4139164_2overs",
    "20221105_4142385_2overs",
    "20221203_4143496_2overs",
    "20221112_4142388_2overs",
    "20221203_4142744_2overs",
    "20221203_4139090_2overs",
    "20221112_4139153_2overs",
    "20220327_3916886_2overs",
    "20221126_4155987_2overs",
    "20221112_4142735_2overs",
    "20221112_4155986_2overs",
    "20221001_4139132_2overs",
]
TEST_DATA = [
    "20221126_4139157_2overs", 
    "20221126_4139156_2overs",
    "20221203_4142396_2overs",
    "20221105_4139079_2overs",
    "20221203_4139160_2overs",
    "20221203_4156185_2overs",
    "20221112_4142385_2overs",
    "20221001_4139134_2overs",
]
CHALLENGE_DATA = [
    "20210109_3648673_2overs",
    "20210116_3645290_2overs",
    "20210206_3645307_2overs",
    "20210220_3645317_2overs",
    "20211120_3916896_2overs",
    "20211120_3917049_2overs",
    "20220327_3916807_2overs",
    "20221001_4142369_2overs",
    "20221119_4139083_2overs",
    "20221126_4139082_2overs",
    "20221126_4139083_2overs",
    "20221126_4142741_2overs",
    "20221203_4111235_overs_33_34",
    "20230304_4142421_overs_7_8"
]

def get_datalist(name_list, path, file_name):
    return [os.path.join(path,f)+file_name for f in name_list]

class CricketFeatureDataset(Dataset):
    """Cricket Dataset from extracted features"""
    # 对特征进行clip
    def __init__(self, feature_files=None, annotation_files=None, args=None, stride = None):
        self.feature_files = feature_files
        self.annotation_files = annotation_files
        self.length_seq = args.length_seq
        # 进行clip时候滑动窗口移动距离
        self.stride = stride
        # 存储每个视频的特征，注释，标记
        self.features = {}
        self.annotations = {}
        self.labels = {} # annotations转label
        # 每个视频的分组
        self.label_groups = {}
        self.feature_groups = {}
        self.indxs = {}
        lates_idx = 0

        for i in range(len(feature_files)):
            video_name = feature_files[i].split("/")[-1].split(".")[0]
            # feature
            self.features[video_name] = np.load(feature_files[i])
            # annotation to label
            if self.annotation_files is not None:
                with open(annotation_files[i], "r", encoding="utf-8") as ann_:
                    self.annotations[video_name] = json.load(ann_)
            
                annotated = [
                    int(k) for k in self.annotations[video_name]["event"].keys()
                ]
                self.labels[video_name] = np.zeros(
                    self.features[video_name].shape[0], dtype=int
                )
                for idx in annotated:
                    if idx > len(self.labels[video_name]):
                        continue
                    self.labels[video_name][idx] = 1.0
            
            self.label_groups[video_name] = []
            self.feature_groups[video_name] = []
            if self.stride!=0:
                # 按照滑动窗口窗口大小self.length_seq,滑动距离self.stride来clip训练数据
                num_subdivision = math.floor((len(self.features[video_name]) - self.length_seq) / self.stride) + 1
                for i in range(num_subdivision): 
                    start_idx = i * self.stride
                    self.feature_groups[video_name].append(self.features[video_name][start_idx:start_idx+self.length_seq,:])
                    if self.annotation_files is not None:
                        self.label_groups[video_name].append(self.labels[video_name][start_idx:start_idx+self.length_seq])
                # 长度不够则删除最后一组
                if len(self.feature_groups[video_name][-1]) < self.length_seq:
                    self.feature_groups[video_name] = self.feature_groups[video_name][:-1]
                    if self.annotation_files is not None:
                        self.label_groups[video_name] = self.label_groups[video_name][:-1]
            else:
                # 不采用滑动窗口直接预测整个视频
                self.feature_groups[video_name].append(self.features[video_name])
                if self.annotation_files is not None:
                    self.label_groups[video_name].append(self.labels[video_name])
            # 给每个片段建立一个索引
            for idx in range(len(self.feature_groups[video_name])):
                new_idx = lates_idx + idx
                self.indxs[new_idx] = (video_name, idx)
            lates_idx += len(self.feature_groups[video_name])

    def __len__(self):
        return sum(len(v) for v in self.feature_groups.values())

    def __getitem__(self, idx):
        video_name_, idx_ = self.indxs[idx]
        feature = self.feature_groups[video_name_][idx_]
        if self.annotation_files is None:
            return torch.Tensor(feature).float()
        label = self.label_groups[video_name_][idx_]
        return (
            torch.Tensor(feature).float(),
            torch.Tensor(label).float(),
        )

def get_dataset(phase, args):
    if phase == 'train':
        train_features = get_datalist(TRAIN_DATA, os.path.join(args.data_dir,args.features), '.npy')
        train_annotations = get_datalist(TRAIN_DATA, os.path.join(args.data_dir,args.annotations), '.json')
        # print(train_features)
        # print(train_annotations)
        return CricketFeatureDataset(train_features, train_annotations, args, args.train_stride)
    elif phase == 'test':
        test_features = get_datalist(TEST_DATA, os.path.join(args.data_dir,args.features), '.npy')
        test_annotations = get_datalist(TEST_DATA, os.path.join(args.data_dir,args.annotations), '.json')
        return CricketFeatureDataset(test_features, test_annotations, args, args.test_stride)
    elif phase == 'infer':
        challenge_feature = get_datalist(CHALLENGE_DATA, os.path.join(args.data_dir,args.features), '.npy')
        return CricketFeatureDataset(challenge_feature, None, args, args.infer_stride)
    else:
        LOGGER = logging.getLogger(__name__)
        LOGGER.info("no exist phase:{phase}")