import cv2
import torch
import ndjson
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader

from model.visil import ViSiL


def center_crop(frame, desired_size):
    if frame.ndim == 3:
        old_size = frame.shape[:2]
        top = int(np.maximum(0, (old_size[0] - desired_size)/2))
        left = int(np.maximum(0, (old_size[1] - desired_size)/2))
        return frame[top: top+desired_size, left: left+desired_size, :]
    else: 
        old_size = frame.shape[1:3]
        top = int(np.maximum(0, (old_size[0] - desired_size)/2))
        left = int(np.maximum(0, (old_size[1] - desired_size)/2))
        return frame[:, top: top+desired_size, left: left+desired_size, :]

def resize_frame(frame, desired_size):
    min_size = np.min(frame.shape[:2])
    ratio = desired_size / min_size
    frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return frame

def load_video(video, all_frames=False, fps=1, cc_size=224, rs_size=256):
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video)
    fps_div = fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 144 or fps is None:
        fps = 25
    frames = []
    count = 0
    while cap.isOpened():
        ret = cap.grab()
        if int(count % round(fps / fps_div)) == 0 or all_frames:
            ret, frame = cap.retrieve()
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if rs_size is not None:
                    frame = resize_frame(frame, rs_size)
                frames.append(frame)
            else:
                break
        count += 1
    cap.release()
    frames = np.array(frames)
    if cc_size is not None:
        frames = center_crop(frames, cc_size)
    return torch.from_numpy(frames)


class MOMADataset(Dataset):
    def __init__(self, split="train", tail_range=400):
        super().__init__()
        self.split = split
        self.root_dir = Path("/data/dir_moma")
        self.anno = self._load_anno_data()
        self.tail_range = tail_range

        self._load_pairwise_similarity()
        if split == "val" or split == "test":
            self._prepare_val_batches()

    def _prepare_val_batches(self):
        self.batches = []
        for i, query in enumerate(self.anno):
            batch = {
                "query_video_id": query["video_id"], # video id e.g. '-49z-lj8eYQ'
                "query_cname": query["cname"], # activity name e.g. "basketball game"
                "trg_video_ids": [x["video_id"] for x in self.anno if query["video_id"] != x["video_id"]],
                "trg_cnames": [x["cname"] for x in self.anno if query["video_id"] != x["video_id"]],
            }

            similarities = torch.cat([self.sm[i][:i], self.sm[i][i+1:]])
            similarities = torch.clamp(similarities, min=0.)
            batch["similarities"] = similarities
            
            self.batches.append(batch)

    def _load_anno_data(self):
        with open(f"anno/moma/{self.split}.ndjson", "r") as f:
            return ndjson.load(f)
    
    def _load_pairwise_similarity(self):
        self.sm = np.load(f"anno/moma/sm_dtw_{self.split}.npy")
        self.sm = torch.from_numpy(self.sm).float()
    
    def sample_pair(self, idx):
        similarties = self.sm[idx]
        _, sorted_idx = torch.sort(similarties, descending=True)
        mask = (sorted_idx != idx)
        sorted_idx = sorted_idx[mask]
        
        # oversampling
        if np.random.rand(1) < 0.5:
            pair_idx = sorted_idx[np.random.randint(self.tail_range)]
        else:
            pair_idx = sorted_idx[np.random.randint(len(sorted_idx))]

        return pair_idx, similarties[pair_idx]
    
    def __len__(self):
        return len(self.anno)
    
    def __getitem__(self, idx):
        vid_root = Path("/data/dir_moma/videos/raw")

        if self.split == "train":
            # read anchor video
            anchor_id = self.anno[idx]['video_id']
            anchor_vid = load_video(str(vid_root / f"{anchor_id}.mp4"))

            # sample pair with similarity to anchor
            pair_idx, sim = self.sample_pair(idx)
            pair_id = self.anno[pair_idx]['video_id']
            pair_vid = load_video(str(vid_root / f"{pair_id}.mp4"))

            anchor = {
                'video_id': self.anno[idx]['video_id'],
                'cname': self.anno[idx]['cname'],
                'vid': anchor_vid
            }
            pair = {
                'video_id': self.anno[pair_idx]['video_id'],
                'cname': self.anno[pair_idx]['cname'],
                'vid': pair_vid
            }

            return anchor, pair, sim
        else:
            batch = self.batches[idx]
            query_vid = load_video(str(vid_root / f"{batch['query_video_id']}.mp4"))
            batch['query_video'] = query_vid
            return batch
        

def extract_features(model, frames, args):
    with torch.no_grad():
        return model.extract_features(frames.to(args.gpu_id).float())


def calculate_similarities_to_queries(model, query, target):
    return model.calculate_video_similarity(query, target)


if __name__ == "__main__":
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='This is the code for the evaluation of ViSiL network on five datasets.', formatter_class=formatter)
    parser.add_argument('--n_epoch', type=int, default=10000,
                        help='Number of epochs during training')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Id of the GPU used.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers used for video loading.')
    args = parser.parse_args()

    model = ViSiL(pretrained=False)
    # Fix parameters of resnet feature extractor
    for param in model.cnn.parameters():
        param.requires_grad = False
    model = model.to('cuda')

    dataset = MOMADataset("train")
    dataloader = DataLoader(dataset, shuffle=True, num_workers=args.workers)

    for epoch in tqdm(range(args.n_epoch), desc='epoch'):
        for batch in tqdm(dataloader, desc='iteration'):
            anchor_vid = batch[0]['vid'][0]
            pair_vid = batch[1]['vid'][0]
            gt_sim = batch[2].to('cuda')

            # 1. extract features
            anchor_feat = extract_features(model, anchor_vid, args)  # [n_frames, n_channels, dim]
            pair_feat = extract_features(model, pair_vid, args)  # [n_frames, n_channels, dim]

            # 2. calculate similarities
            sim = calculate_similarities_to_queries(model, anchor_feat, pair_feat)
            print("")