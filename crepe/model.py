import numpy as np
from scipy.io import wavfile
import functools

import torch
import torch.nn.functional as F


###########################################################################
# Model definition
###########################################################################
PITCH_BINS = 360
WINDOW_SIZE = 1024


class Crepe(torch.nn.Module):
    """Crepe model definition"""

    def __init__(self, model='full'):
        super().__init__()

        # Model-specific layer parameters
        if model == 'full':
            in_channels = [1, 1024, 128, 128, 128, 256]
            out_channels = [1024, 128, 128, 128, 256, 512]
            self.in_features = 2048
        elif model == 'tiny':
            in_channels = [1, 128, 16, 16, 16, 32]
            out_channels = [128, 16, 16, 16, 32, 64]
            self.in_features = 256
        else:
            raise ValueError(f'Model {model} is not supported')

        # Shared layer parameters
        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        # Overload with eps and momentum conversion given by MMdnn
        batch_norm_fn = functools.partial(torch.nn.BatchNorm2d,
                                          eps=0.0010000000474974513,
                                          momentum=0.0)

        # Layer definitions
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0])
        self.conv1_BN = batch_norm_fn(
            num_features=out_channels[0])

        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=kernel_sizes[1],
            stride=strides[1])
        self.conv2_BN = batch_norm_fn(
            num_features=out_channels[1])

        self.conv3 = torch.nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=kernel_sizes[2],
            stride=strides[2])
        self.conv3_BN = batch_norm_fn(
            num_features=out_channels[2])

        self.conv4 = torch.nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_sizes[3],
            stride=strides[3])
        self.conv4_BN = batch_norm_fn(
            num_features=out_channels[3])

        self.conv5 = torch.nn.Conv2d(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=kernel_sizes[4],
            stride=strides[4])
        self.conv5_BN = batch_norm_fn(
            num_features=out_channels[4])

        self.conv6 = torch.nn.Conv2d(
            in_channels=in_channels[5],
            out_channels=out_channels[5],
            kernel_size=kernel_sizes[5],
            stride=strides[5])
        self.conv6_BN = batch_norm_fn(
            num_features=out_channels[5])

        self.classifier = torch.nn.Linear(
            in_features=self.in_features,
            out_features=PITCH_BINS)

    def forward(self, x, embed=False):
        # Forward pass through first five layers
        x = self.embed(x)

        # Forward pass through layer six
        x = self.layer(x, self.conv6, self.conv6_BN)

        # shape=(batch, self.in_features)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)

        if embed:
            return x

        # Compute logits
        return torch.sigmoid(self.classifier(x))

    ###########################################################################
    # Forward pass utilities
    ###########################################################################

    def embed(self, x):
        """Map input audio to pitch embedding"""
        # shape=(batch, 1, 1024, 1)
        x = x[:, None, :, None]

        # Forward pass through first five layers
        x = self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
        x = self.layer(x, self.conv2, self.conv2_BN)
        x = self.layer(x, self.conv3, self.conv3_BN)
        x = self.layer(x, self.conv4, self.conv4_BN)
        x = self.layer(x, self.conv5, self.conv5_BN)

        return x

    def layer(self, x, conv, batch_norm, padding=(0, 0, 31, 32)):
        """Forward pass through one layer"""
        x = F.pad(x, padding)
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        return F.max_pool2d(x, (2, 1), (2, 1))


class CrepeInfer():

    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        self.model = Crepe("full")
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        # Place on device
        self.model = self.model.to(device)
        # Eval mode
        self.model.eval()

    def load_audio(self, filename):
        """Load audio from disk"""
        sample_rate, audio = wavfile.read(filename)
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / np.iinfo(np.int16).max
        # PyTorch is not compatible with non-writeable arrays, so we make a copy
        return torch.tensor(np.copy(audio))[None], sample_rate

    def preprocess(self, audio, hop_length, batch_size=None, pad=True):
        # Get total number of frames
        # Maybe pad
        if pad:
            total_frames = 1 + int(audio.size(1) // hop_length)
            audio = torch.nn.functional.pad(
                audio,
                (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        else:
            total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)
        # Default to running all frames in a single batch
        batch_size = total_frames if batch_size is None else batch_size
        # Generate batches
        for i in range(0, total_frames, batch_size):
            # Batch indices
            start = max(0, i * hop_length)
            end = min(audio.size(1),
                      (i + batch_size - 1) * hop_length + WINDOW_SIZE)
            # Chunk
            frames = torch.nn.functional.unfold(
                audio[:, None, None, start:end],
                kernel_size=(1, WINDOW_SIZE),
                stride=(1, hop_length))
            # shape=(1 + int(time / hop_length, 1024)
            frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE)
            # Place on device
            frames = frames.to(self.device)
            # Mean-center
            frames -= frames.mean(dim=1, keepdim=True)
            # Scale
            # Note: during silent frames, this produces very large values. But
            # this seems to be what the network expects.
            frames /= torch.max(torch.tensor(1e-10, device=frames.device),
                                frames.std(dim=1, keepdim=True))
            yield frames

    def embed(self, audio, hop_length=None, batch_size=None, pad=True):
        results = []
        # Preprocess audio
        generator = self.preprocess(audio, hop_length, batch_size, pad)
        for frames in generator:
            with torch.no_grad():
                # Infer pitch embeddings
                embedding = self.model(frames, embed=True)
                # shape=(batch, time / hop_length, 32, embedding_size)
                result = embedding.reshape(
                    audio.size(0), frames.size(0), 32, -1)
                # Place on same device as audio. This allows for large inputs.
                results.append(result.to(self.device))
        # Concatenate
        return torch.cat(results, 1)

    def compute_f0(self, audio):
        audio = torch.tensor(np.copy(audio))[None]
        audio = audio.to(self.device)
        hop_length = 160
        f0 = self.embed(audio, hop_length, 256).squeeze()
        length = f0.shape[0]
        f0 = torch.reshape(f0, (length, -1))
        return f0.data.cpu().float().numpy()

    def compute_f0_file(self, path):
        audio, sr = self.load_audio(path)
        audio = audio.to(self.device)
        assert sr == 16000
        # Here we'll use a 10 millisecond hop length
        hop_length = int(sr / 100.0)
        assert hop_length == 160
        f0 = self.embed(audio, hop_length, 256).squeeze()
        length = f0.shape[0]
        f0 = torch.reshape(f0, (length, -1))
        return f0.data.cpu().float().numpy()