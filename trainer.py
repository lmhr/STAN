import torch
import os
import numpy as np
import json
from tqdm import tqdm
from utils import convert_events, compute_pq_metric, rising_edge


def train(LOGGER, dataloader, model, Loss, optimizer):
    # 训练框架
    """Training loop for the model"""
    model.train()
    correct, train_loss, num_frames = 0, 0, 0
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        # Compute
        X, y = X.cuda(), y.cuda()
        pred = model(X)
        loss = Loss(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # save_result
        correct += ((pred > 0.5) == y).type(torch.float).sum().item()
        train_loss += loss.item()*y.shape[0]*y.shape[1]
        num_frames += y.shape[0]*y.shape[1]
    correct /= num_frames
    train_loss /= num_frames
    LOGGER.info(
        f"train: Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f}"
    )

def test(LOGGER, dataloader, model, Loss, log_path=""):
    # 在测试集上验证结果
    """Test function"""
    model.eval()
    correct, test_loss, num_frames = 0, 0, 0
    pred_dict, gt_dict = {}, {}
    with torch.no_grad():
        for idx, (X, y) in enumerate(dataloader):
            # Compute
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            loss = Loss(pred, y)
            # save_result
            correct += ((pred > 0.5) == y).type(torch.float).sum().item()
            test_loss += loss.item()*y.shape[0]*y.shape[1]
            num_frames += y.shape[0]*y.shape[1]
            # restore a video
            video_batch_list = [
                dataloader.dataset.indxs[i][0]
                for i in range(len(y) * idx, len(y) * (idx + 1))
            ] 
            pred_np = (pred > 0.5).type(torch.float).cpu().numpy()
            gt_np = y.cpu().numpy()
            for ib, vid in enumerate(video_batch_list):
                pred_dict.setdefault(vid, []).append(pred_np[ib, :])
                gt_dict.setdefault(vid, []).append(gt_np[ib, :])

    # 在测试集上并未关注其clip开始的帧数
    for k, v in pred_dict.items():
        pred_dict[k] = np.concatenate(v)
    for k, v in gt_dict.items():
        gt_dict[k] = np.concatenate(v)
    test_loss /= num_frames
    correct /= num_frames
    LOGGER.info(
        f"Test: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}"
    )
    pr_events, gt_events = convert_events(pred_dict, gt_dict)
    pq_test, sq_test, rq_test = compute_pq_metric(gt_events, pr_events)
    LOGGER.info(
        f"Panoptic Quality: {(100*pq_test):>0.1f}%, Segmentation Quality: {(100*sq_test):>0.1f}%, Recognition Quality: {(100*rq_test):>0.1f}% \n"
    )
    if log_path:
        with open(
            os.path.join(log_path, "test_predictions.json"),
            "w",
            encoding="utf-8",
        ) as f_json:
            json.dump(pr_events, f_json, ensure_ascii=False, indent=4)
        with open(
            os.path.join(log_path, "test_groundtruths.json"),
            "w",
            encoding="utf-8",
        ) as f_json:
            json.dump(gt_events, f_json, ensure_ascii=False, indent=4)

    return pq_test

def infer(LOGGER, dataloader, model, log_path=""):
    # 在单纯数据集上进行推断
    """Inference function"""
    model.eval()
    pred_dict = {}
    with torch.no_grad():
        for idx, X in enumerate(dataloader):
            # Compute
            X = X.cuda()
            pred = model(X)
            # restore a video
            video_batch_list = [
                dataloader.dataset.indxs[i][0]
                for i in range(len(X) * idx, len(X) * (idx + 1))
            ]
            pred_np = (pred > 0.5).type(torch.float).cpu().numpy()
            for ib, vid in enumerate(video_batch_list):
                pred_dict.setdefault(vid, []).append(pred_np[ib, :])
    for k, v in pred_dict.items():
        pred_dict[k] = np.concatenate(v)
    pr_events = {}
    for k, val in pred_dict.items():
        events_ = rising_edge(val)
        pr_events[k] = dict(enumerate(events_))
    LOGGER.info("Inference run on the Challenge set.\n")
    if log_path:
        with open(
            os.path.join(log_path, "challenge_predictions.json"),
            "w",
            encoding="utf-8",
        ) as f_json:
            json.dump(pr_events, f_json, ensure_ascii=False, indent=4)
        LOGGER.info(
            f"Predictions saved in {log_path}/challenge_predictions.json."
        )