import torch
import ndjson
import numpy as np

from pathlib import Path
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader

from model.visil import ViSiL


class ToTensorNormalize(object):
    def __init__(self, use_ms=True):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.use_ms = use_ms

    def __call__(self, frames):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)
        frames /= 255

        if self.use_ms:
            for fi, (f, m, s) in enumerate(zip(frames, self.mean, self.std)):
                frames[fi] = (f - m) / s

        return frames

class MOMADataset(Dataset):
    def __init__(self, split="train", tail_range=400):
        super().__init__()
        self.tail_range = tail_range
        self.split = split
        self.root_dir = Path("/data/dir_moma")
        self.anno = self._load_anno_data()
        self.transform = Compose([ToTensorNormalize()])

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
        feat_root = Path("/data/dir_moma/videos/resnet_feats")

        if self.split == "train":
            # read anchor video
            anchor_id = self.anno[idx]['video_id']
            anchor_tensor = torch.from_numpy(np.load(feat_root / f"{anchor_id}.npy"))
            anchor_tensor = anchor_tensor[:min(400, anchor_tensor.shape[0]), :, :]

            # sample pair with similarity to anchor
            pair_idx, sim = self.sample_pair(idx)
            pair_id = self.anno[pair_idx]['video_id']
            pair_tensor = torch.from_numpy(np.load(feat_root / f"{pair_id}.npy"))
            pair_tensor = pair_tensor[:min(400, pair_tensor.shape[0]), :, :]

            anchor = {
                'video_id': self.anno[idx]['video_id'],
                'cname': self.anno[idx]['cname'],
                'feat': anchor_tensor
            }
            pair = {
                'video_id': self.anno[pair_idx]['video_id'],
                'cname': self.anno[pair_idx]['cname'],
                'feat': pair_tensor
            }

            return anchor, pair, sim
        else:
            batch = self.batches[idx]
            query_tensor = torch.from_numpy(np.load(feat_root / f"{batch['query_video_id']}.npy"))
            query_tensor = query_tensor[:min(400, query_tensor.shape[0]), :, :]
            batch['query_video'] = query_tensor
            return batch


if __name__ == "__main__":
    dataset = MOMADataset("train")
    anchor, pair, sim = dataset[0]

    #model = ViSiL(pretrained=False)
    print("")