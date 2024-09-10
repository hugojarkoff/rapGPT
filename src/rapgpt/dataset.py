import torch
from torch.utils.data import Dataset
from rapgpt.data import Artist
from rapgpt.encoder import Encoder


class ArtistDataset(Dataset):
    def __init__(self, artist: Artist, encoder: Encoder) -> None:
        self.encoder = encoder
        self.artist = artist
        self.data: torch.Tensor = torch.Tensor(self.encoder.encode_data(self.artist.lyrics))

    def __len__(self) -> int:
        return len(self.data)
    
    def get_random_batch(self, batch_size: int, block_size: int):
        idx = torch.randint(len(self)-block_size, (batch_size,))
        x = torch.stack([self.data[i:i+block_size] for i in idx])
        y = torch.stack([self.data[i+1:i+block_size+1] for i in idx])
        return x, y
