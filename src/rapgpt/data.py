from pathlib import Path
from rapgpt.encoder import Encoder
from rapgpt.config import Config
import torch
from functools import cached_property
import random

class Lyrics(str):
    pass

class ArtistLyrics:
    @classmethod
    def from_file(cls, filename: Path) -> str:
        return filename.read_text()


class Artist:
    def __init__(self, artist_file: Path) -> None:
        self.name: str = artist_file.name.rstrip(".txt")
        self.lyrics = ArtistLyrics.from_file(artist_file)


class Corpus:
    def __init__(self, config: Config, encoder: Encoder) -> None:
        self.config = config        
        self.encoder = encoder
        
        self.data_path: Path = Path(self.config.data.path)
        
        self.artists: list[Artist] = []
        for path in self.data_path.glob("*.txt"):
            artist = Artist(path)
            if len(artist.lyrics) < self.config.corpus.min_artist_length:
                continue 
            self.artists.append(Artist(path))

        self.artists_tokens = {artist.name:i for i, artist in enumerate(self.artists)}
        self.split_train_val = self.config.corpus.split_train_val

    def dump_artists_tokens(self, filepath: Path) -> None:
        """Saves artists tokens"""
        with open(filepath, "w+") as f:
            for k,v in self.artists_tokens.items():
                f.write(f"{k}:{v}\n")
        

    @cached_property
    def train_data(self) -> dict[str, torch.Tensor]:
        return {
            artist.name: torch.Tensor(
                self.encoder.encode_data(
                    artist.lyrics[: int(len(artist.lyrics) * self.split_train_val)]
                )
            )
            for artist in self.artists
        }
    
    @cached_property
    def val_data(self) -> dict[str, torch.Tensor]:
        return {
            artist.name: torch.Tensor(
                self.encoder.encode_data(
                    artist.lyrics[int(len(artist.lyrics) * self.split_train_val) :]
                )
            )
            for artist in self.artists
        }

    def get_random_batch(self, batch_size: int, block_size: int, split: str = "train"):
        batch_inputs = []
        batch_targets = []
        selected_artists = []

        match split:
            case "train":
                source = self.train_data
            case "val":
                source = self.val_data
            case _:
                raise ValueError(f"Incorrect split: {split}")

        for _ in range(batch_size):
            # Randomly select an artist
            artist_name = random.choice(list(source.keys()))

            # Save artist at the same position from batch
            selected_artists.append(self.artists_tokens[artist_name])
            
            # Get the tensor for the selected artist
            lyrics_tensor = source[artist_name]
            
            # Ensure there are enough tokens for a full passage
            max_start_idx = lyrics_tensor.size(0) - block_size - 1
            
            if max_start_idx <= 0:
                raise ValueError(f"Lyrics for {artist_name} are too short for the given block size!")
            
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
        selected_artists = torch.Tensor(selected_artists)
        
        return batch_inputs, batch_targets, selected_artists

