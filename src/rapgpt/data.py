from pathlib import Path
from rapgpt.encoder import Encoder
import torch
from functools import cached_property
import random

class ArtistLyrics:
    @classmethod
    def from_file(cls, filename: Path) -> str:
        return filename.read_text()


class Artist:
    def __init__(self, artist_file: Path) -> None:
        self.name: str = artist_file.name.strip(".txt")
        self.lyrics = ArtistLyrics.from_file(artist_file)


class Corpus:
    def __init__(self, data_path: str | Path, encoder: Encoder) -> None:
        self.data_path: Path = Path(data_path)
        self.encoder = encoder
        # TODO: Move these assertions to Config init?
        assert self.data_path.exists()
        assert self.data_path.glob("*.txt")

        self.artists: list[Artist] = [
            Artist(path) for path in self.data_path.glob("*.txt")
        ]

    @cached_property
    def data(self) -> dict[str, torch.Tensor]:
        result = {}
        for artist in self.artists:
            if len(artist.lyrics) < 1000:
                continue
            result[artist] = torch.Tensor(self.encoder.encode_data(artist.lyrics))
        return result

    def get_random_batch(self, batch_size: int, block_size: int):
        batch_inputs = []
        batch_targets = []
        selected_artists = []

        for _ in range(batch_size):
            # Randomly select an artist
            artist = random.choice(list(self.data.keys()))

            # Save artist at the same position from batch
            selected_artists.append(artist)
            
            # Get the tensor for the selected artist
            lyrics_tensor = self.data[artist]
            
            # Ensure there are enough tokens for a full passage
            max_start_idx = lyrics_tensor.size(0) - block_size
            
            if max_start_idx <= 0:
                raise ValueError(f"Lyrics for {artist} are too short for the given block size!")
            
            # Randomly select a start index
            start_idx = random.randint(0, max_start_idx)
            
            # Extract a passage of length block_size
            inputs = lyrics_tensor[start_idx : start_idx + block_size]
            targets = lyrics_tensor[start_idx + 1 : start_idx + block_size + 1]
            
            # Append to the batch
            batch_inputs.append(inputs)
            batch_targets.append(targets)
        
        # Stack the passages into a tensor of shape (batch_size, block_size)
        batch_inputs = torch.stack(batch_inputs)
        batch_targets = torch.stack(batch_targets)
        
        return batch_inputs, batch_targets, selected_artists

