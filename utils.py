import numpy as np
from typing import Dict, List, Tuple
from pycocotools.mask import iou
from scipy.optimize import linear_sum_assignment
import random
import numpy as np
import torch
import os
from argparse import ArgumentParser
import logging
import time
import pprint

def get_args():
    # argparser
    # 配置参数
    parser = ArgumentParser(
        description="Example of training, test, infer on Cricket dataset."
    )
    # data路径
    parser.add_argument("--data_dir", type=str, default="../cricket-bowlrelease-dataset/")
    parser.add_argument("--annotations", type=str, default="data")
    parser.add_argument("--videos", type=str, default="videos")
    parser.add_argument("--features", type=str, default="features")
    # debugging
    parser.add_argument("--debugging", action="store_true", default=False)
    # phase //train,test,infer
    parser.add_argument("--phase", default="train", type=str)
    # train set
    parser.add_argument("--model_name", type=str, default='LSTM')
    parser.add_argument("--loss_name", type=str, default='weightedloss')
    parser.add_argument("--length_seq", type=int, default=100)
    parser.add_argument("--train_stride", type=int, default=10)
    parser.add_argument("--test_stride", type=int, default=0)
    parser.add_argument("--infer_stride", type=int, default=0)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    # test，infer使用的模型路径
    parser.add_argument("--resume", type=str, default="")
    # 随机种子
    parser.add_argument("--random_seed", default=3407, type=int)
    # 模型保存
    parser.add_argument("--model_best", type=str, default="model_best.pth")
    parser.add_argument("--model_final", type=str, default="model_final.pth")
    # 保存结果
    parser.add_argument("--save_result", action="store_true", default=False)

    # args = parser.parse_args()
    args = parser.parse_known_args()[0]
    return args

def set_random_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def configure_logger(args, verbose: bool, eval: bool) -> str:
    """Configures the logging verbosity"""
    logger = logging.getLogger("bowlrelease")
    formatter = logging.Formatter(
        "%(asctime)s  %(message)s","%y-%m-%d %H:%M:%S"
    )
    handler = logging.StreamHandler()
    stream_level = logging.DEBUG if verbose else logging.INFO
    handler.setLevel(stream_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    log_path = ""
    if not eval:
        log_path = os.path.join(
            "logs/{}".format(args.model_name), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
        os.makedirs(log_path, exist_ok=True)

        fhlr = logging.FileHandler(os.path.join(log_path, "train.log"))
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)
    elif args.save_result and args.resume is not None:
        log_path = os.path.dirname(args.resume)
    logger.setLevel(logging.DEBUG)
    return log_path, logger

def print_exp_info(LOGGER, args):
    LOGGER.info(pprint.pformat(vars(args)))
    LOGGER.info("PyTorch version: {}".format(torch.__version__))
    LOGGER.info("CUDA version: {}".format(torch.version.cuda))
    LOGGER.info("{} GPUs".format(torch.cuda.device_count()))
    LOGGER.info(f"Random Seed: {args.random_seed}")

def rising_edge(data: np.ndarray, thresh: float = 0.5) -> List[List[int]]:
    """Detects events where the output goes from zero to one.
    The events are list of two elements:
    when the event starts, when the event ends."""
    sign = data >= thresh
    pos = np.where(np.convolve(sign, [1, -1]) == 1)[0] # 计算开始边界
    neg = np.where(np.convolve(sign, [1, -1]) == -1)[0] # 计算结束边界
    neg -= 1
    assert len(pos) == len(neg), "error"
    return [[int(p), int(n)] for p, n in zip(pos, neg)]

def convert_events(
    preds: Dict[str, np.ndarray], gts: Dict[str, np.ndarray]
) -> Tuple[
    Dict[str, Dict[int, List[List[int]]]],
    Dict[str, Dict[int, List[List[int]]]],
]:
    "Retruns predictions and gt events in dict format."
    gt_events = {}
    pr_events = {}
    for k, val in gts.items():
        events_ = rising_edge(val)
        gt_events[k] = dict(enumerate(events_))
    for k, val in preds.items():
        events_ = rising_edge(val)
        pr_events[k] = dict(enumerate(events_))

    return pr_events, gt_events

def _compute_matching(det, ann):
    IOU_THRESHOLD = 0.5
    iou_matrix = iou(det[:, :4], ann[:, :4], np.zeros((len(ann)))).T
    iou_matrix[iou_matrix < IOU_THRESHOLD] = 0.0
    iou_sum = 0.0
    det_idxs, ann_idxs = linear_sum_assignment(iou_matrix.T, maximize=True)
    ann_idxs_mth, det_idxs_mth = [], []
    for anid, deid in zip(ann_idxs, det_idxs):
        if iou_matrix[anid, deid] >= IOU_THRESHOLD:
            ann_idxs_mth.append(anid)
            det_idxs_mth.append(deid)
            iou_sum += iou_matrix[anid, deid]
    ann_idxs_mth = np.array(ann_idxs_mth)
    det_idxs_mth = np.array(det_idxs_mth)
    return iou_sum, det_idxs_mth, ann_idxs_mth

def _compute_pq_sq_rq(det, ann):
    iou_sum, det_idxs_mth, ann_idxs_mth = _compute_matching(det, ann) #计算IOU，返回预测为真的样例, 未分割出来的片段
    fps = 0
    for id_, _ in enumerate(det): # 预测成真的片段有多少实际为假
        if id_ not in det_idxs_mth:
            fps += 1
    fns = 0
    for id_, _ in enumerate(ann): # 多少真的片段有未被发现
        if id_ not in ann_idxs_mth:
            fns += 1
    tps = len(det_idxs_mth) # 预测的真的片段

    return tps, fps, fns, iou_sum

def compute_pq_metric(
    gt_data: Dict[str, Dict[int, List[List[int]]]],
    pred_data: Dict[str, Dict[int, List[List[int]]]],
) -> Tuple[float, float, float]:
    """Panoptic Quality metric.
    It computes the mean of the Panoptic quality scores across all videos.
    The input format is a dict with keys as video_names and values as Dict.
    The values Dict have keys as integer (event number) and values as
    a list of lists two integers (event start frame, event end frame).

    Args:
        gt_data (Dict): ground truth data.
        pred_data (Dict): prediction data.

    Returns:
        Tuple: Panoptic Quality, Segmentation Quality, Recognition Quality.
    """
    tps, fps, fns, iou_sum = 0.0, 0.0, 0.0, 0.0
    for video_key, video_val in gt_data.items():
        pred = pred_data.get(video_key, {})
        if not pred:
            fns += len(video_val)
            continue
        det_video = [[d[0], 1, d[1], 1] for d in pred.values()]
        ann_video = [[d[0], 1, d[1], 1] for d in video_val.values()]
        tps_, fps_, fns_, iou_sum_ = _compute_pq_sq_rq(
            np.array(det_video), np.array(ann_video)
        )
        tps += tps_
        fps += fps_
        fns += fns_
        iou_sum += iou_sum_
    sq_ = iou_sum / tps if tps else 0
    rq_ = tps / (tps + 0.5 * fns + 0.5 * fps)
    pq_ = sq_ * rq_
    return pq_, sq_, rq_