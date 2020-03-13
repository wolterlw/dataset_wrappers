
from pathlib import Path
from imageio import get_reader
from datasets import BaseDataset, IndexedDataset

class SURREAL(BaseDataset):
    r"""root - path to the ./cmu/{train,val,test} 
    directory of the SURREAL dataset
    """
    def __init__(self, root: str):
        super(SURREAL, self).__init__()
        self.root = Path(root).expanduser().absolute()
        self.seqs = sorted(self.root.glob("*/*/*.mp4"))

    def _mp4_to_info(self, path):
        return path.with_name(
            path.name.replace(".mp4", "_info.mat"))

    def __len__(self, ):
        return len(self.seqs)

    def __getitem__(self, idx):
        vid_f = self.seqs[idx]
        anno_f = self._mp4_to_info(vid_f)

        sample = {
            'video_file': vid_f,
            'anno_file': anno_f
        }
        return sample

class SurrealSequential(IndexedDataset):
    r"""Simplest possible sequential indexing of the SURREAL dataset"""
    def __init__(self, dataset, transforms, num_frames=1, stride=1):
        """
        TODO: implement full indexing
        """
        super(SurrealSequential, self).__init__(dataset, transforms)

        self.num_frames = num_frames
        self.stride = stride
        self.valid_samples = [x for x in dataset if self._valid(x)]
        self.cache()

    def _valid(self, sample):
        has_frames = False
        with get_reader(sample['video_file']) as rd:
            has_frames = rd.count_frames() > self.num_frames * self.stride
        return has_frames

    def __len__(self, ):
        return len(self.valid_samples)

    def __getitem__(self, index):
        return self.T(self.valid_samples[index])

    def _cache(self, ):
        pass
        

    