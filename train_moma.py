import cv2
import wandb
import torch
import ndjson
import argparse
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers.wandb import WandbLogger

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
    
    frames = torch.from_numpy(frames)
    # duplicate if video is too short
    if frames.shape[0] < 10:
        frames = torch.cat([frames, frames], dim=0)
    return frames


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


class nDCGMetric:
    def __init__(self, topK):
        self.topK = topK
        self.score = {f"nDCG@{k}": [] for k in self.topK}
            
    def update(self, pred, proxy):
        _, pred_idx = torch.topk(pred, max(self.topK))
        _, opt_idx = torch.topk(proxy, max(self.topK))

        # proxy = torch.clamp(proxy, min=0.5)
        # proxy = (proxy - proxy.min()) / (proxy.max() - proxy.min())

        for k in self.topK:
            pred_rel = proxy[pred_idx[:k]]
            opt_rel = proxy[opt_idx[:k]]
            
            dcg = ((2**pred_rel - 1) / torch.log2(torch.arange(2, k+2)).to("cuda")).sum()
            idcg = ((2**opt_rel - 1) / torch.log2(torch.arange(2, k+2)).to("cuda")).sum()
            
            self.score[f"nDCG@{k}"].append(dcg / idcg)
        
    def compute(self):
        return {
            k: torch.tensor(v).mean() for k, v in self.score.items()
        }
        
    def reset(self):
        self.score = {f"nDCG@{k}": [] for k in self.topK}
        

class MSEError:
    def __init__(self):
        self.mse_log = []
        
    def update(self, pred_similarities, proxy_similarities):
        diff = pred_similarities - proxy_similarities
        self.mse_log.append((diff ** 2).mean())
        
    def compute(self):
        return torch.tensor(self.mse_log).mean()
        
    def reset(self):
        self.mse_log = []


class VisilWrapper(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
        self.model = ViSiL(pretrained=False)
        for param in self.model.cnn.parameters():
            param.requires_grad = False
        
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        # for evaluation
        self.id2emb = {}
        self.eval_query_ids = []
        self.eval_trg_ids = []
        self.eval_sim_ids = []
        self.ndcg_metric = nDCGMetric([5, 10, 20, 40])
        self.mse_error = MSEError()
    
    def forward(self, anchor, pair):
        with torch.no_grad():
            anchor = self.model.extract_features(anchor.float())
            pair = self.model.extract_features(pair.float())
        return self.model.calculate_video_similarity(anchor, pair)
    
    def training_step(self, batch):
        anchor_vid = batch[0]['vid'][0]
        pair_vid = batch[1]['vid'][0]
        gt = batch[2]  # gt similarity between anchor and pair
        pred = self(anchor_vid, pair_vid)  # similarity prediction

        loss = self.mse_loss(pred, gt)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return loss
    
    def validation_step(self, batch, _):
        if batch['query_video_id'] not in self.id2emb:
            self.id2emb[batch['query_video_id']] = self.model.extract_features(batch['query_video'].float()).cpu()
        self.eval_query_ids.append(batch['query_video_id'])
        self.eval_trg_ids.append(batch['trg_video_ids'])
        self.eval_sim_ids.append(batch['similarities'])
    
    def validation_epoch_end(self, _):
        # TODO implement calculating similarities between query and trg and log ndcg metric
        for query_id, trg_ids, sim in tqdm(list(zip(
            self.eval_query_ids,
            self.eval_trg_ids,
            self.eval_sim_ids
        )), desc="validation epoch end"):
            query_emb = self.id2emb[query_id].to("cuda")
            preds = []
            for trg_id in trg_ids:
                trg_emb = self.id2emb[trg_id].to("cuda")
                pred = self.model.calculate_video_similarity(query_emb, trg_emb)
                preds.append(pred)
            preds = torch.stack(preds, dim=0).squeeze(-1)
            self.ndcg_metric.update(preds, sim)
            self.mse_error.update(preds, sim)
        
        score = self.ndcg_metric.compute()
        score['mse_error'] = self.mse_error.compute()
        for k, v in score.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        self.id2emb = {}
        self.eval_query_ids = []
        self.eval_trg_ids = []
        self.eval_sim_ids = []
        self.ndcg_metric.reset()
        self.mse_error.reset() 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr
        )
        return optimizer


if __name__ == "__main__":
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='This is the code for the evaluation of ViSiL network on five datasets.', formatter_class=formatter)
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=100000000,
                        help='Number of epochs during training')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Id of the GPU used.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers used for video loading.')
    parser.add_argument('--project', type=str, default="visil",
                        help='project name on wandb')
    parser.add_argument('--run', type=str, default="moma_v1",
                        help='run name on wandb')
    args = parser.parse_args()
    wandb.init(config=args, project=args.project, name=args.run)

    model = VisilWrapper(args)
    wandb.watch(model, log_freq=100)

    train_dataset = MOMADataset("train")
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=args.workers)
    val_dataset = MOMADataset("test")
    val_dataloader = DataLoader(val_dataset, shuffle=True, num_workers=args.workers, collate_fn=lambda x: x[0])

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[args.gpu_id],
        strategy='ddp',
        logger=WandbLogger(project=args.project, name=args.run),
        log_every_n_steps=1,
        check_val_every_n_epoch=5,
        num_sanity_val_steps=0,
        max_epochs=args.n_epoch
    )
    trainer.fit(model, train_dataloader, val_dataloader)
