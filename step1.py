import torch
import os
import cv2
from torchvision.models import ResNet50_Weights, resnet50
from vidgear.gears import CamGear
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

def _frame_to_tensor(frame, device):
    return (
        (torch.tensor(frame.transpose(2, 0, 1)).float() / 255.0).to(device)
        # .unsqueeze(0)
    )

def get_device():
    """returns the device"""
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

def extract_features_from_video(
    video_name: str,
    video_dir: str,
    batch_size: int,
    device: str,
    filename: str = "features.npy",
):
    """Utiltiy to extract features from a video"""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    pred_ = []
    frames_batch = []
    batch_size = batch_size

    video_path = os.path.join(video_dir, f"{video_name}.mp4")
    stream = CamGear(source=video_path, colorspace="COLOR_BGR2RGB").start()

    frames = int(stream.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    with torch.no_grad():
        for index in tqdm(range(frames), total=frames):
            frame = stream.read()
            if frame is None:
                continue
            frame_tensor = _frame_to_tensor(frame, device)
            frames_batch.append(frame_tensor)
    
            # Once the frames reach the desired batch size, process them
            if len(frames_batch) == batch_size:
                frames_stacked = torch.stack(frames_batch)
                predictions = model(frames_stacked)
                pred_.extend(predictions.cpu().numpy())
                # Clear the batch for the next set of frames
                frames_batch = []

        if len(frames_batch) > 0:
            frames_stacked = torch.stack(frames_batch)            
            predictions = model(frames_stacked)
            pred_.extend(predictions.cpu().numpy())
    #         pred = model(frame)
    #         pred_.append(pred.cpu().numpy())
    # preds = np.concatenate(pred_)
    # np.save(filename, preds)

def extract_all_videos_features(
    video_dir: str,
    features_dir: str,
    batch_size: int,
    device: str,
    override: bool = False,
):
    """Utiltiy to extract all features from videos in the predefined folder

    Args:
        video_dir (str): videos path
        features_dir (str): features path
        device (str): current devide to extract features
        override (bool, optional): if true it will re-extract the features,
            ignoring the existing files. Defaults to False.

    Returns:
        List[str]: A list of file paths of extracted features.
    """
    list_dirs = os.listdir(video_dir)
    os.makedirs(features_dir, exist_ok=True)
    feature_files = []
    for file in list_dirs:
        video_name = file[:-4]
        feat_filename = os.path.join(features_dir, video_name + ".npy")
        feature_files.append(feat_filename)
        if (not os.path.exists(feat_filename)) or override:
            extract_features_from_video(
                video_name, video_dir, batch_size, device, feat_filename
            )
    return feature_files

if __name__ == '__main__':
    # import debugpy
    # debugpy.listen(5556)
    # print("====debugging====")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()
    parser = ArgumentParser()
    # data path
    parser.add_argument("--data_dir", type=str, default="../cricket-bowlrelease-dataset/")
    parser.add_argument("--videos", type=str, default="videos")
    parser.add_argument("--features", type=str, default="features")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_known_args()[0]
    
    device = get_device()
    feature_list = extract_all_videos_features(
        os.path.join(args.data_dir, args.videos),
        os.path.join(args.data_dir, args.features),
        args.batch_size,
        device,
        override=False,
    )