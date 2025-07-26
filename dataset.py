import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import Vocabulary

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None, split='train', test_size=0.1, val_size=0.1, random_state=42):
        self.root_dir = root_dir
        self.captions_file = captions_file
        self.vocab = vocab
        self.transform = transform
        self.split = split
        self.data = self._load_data()
        self.train_data, self.val_data, self.test_data = self._split_data(test_size, val_size, random_state)
        if split == 'train':
            self.samples = self.train_data
        elif split == 'val':
            self.samples = self.val_data
        else:
            self.samples = self.test_data

    def _load_data(self):
        # Parse captions file (comma-separated, skip header)
        df = pd.read_csv(self.captions_file, sep=',', names=['image', 'caption'], header=0)
        return df

    def _split_data(self, test_size, val_size, random_state):
        train_val, test = train_test_split(self.data, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_state)
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        caption = row['caption']
        numericalized_caption = [self.vocab.word2idx['<SOS>']] + self.vocab.numericalize(caption) + [self.vocab.word2idx['<EOS>']]
        return image, torch.tensor(numericalized_caption)

# Helper function to build vocabulary from all captions
def build_vocab_from_captions(captions_file, freq_threshold=5):
    df = pd.read_csv(captions_file, sep=',', names=['image', 'caption'], header=0)
    sentences = df['caption'].tolist()
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(sentences)
    return vocab 