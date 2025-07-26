import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Flickr8kDataset, build_vocab_from_captions
from model import EncoderCNN, DecoderRNN

# Paths
DATA_DIR = 'data/Flickr8k_Dataset'
CAPTIONS_FILE = 'data/Flickr8k.token.txt'
CHECKPOINT_PATH = 'models/best_model.pth'

# Hyperparameters
EMBED_SIZE = 256
ATTENTION_DIM = 256
DECODER_DIM = 512
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
FREQ_THRESHOLD = 5
NUM_WORKERS = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def load_checkpoint(checkpoint_path, encoder, decoder, optimizer):
    """Load checkpoint and resume training."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f'Resuming from epoch {start_epoch + 1}, batch {checkpoint["batch"]}')
        return start_epoch, best_loss
    return 0, float('inf')

def main():
    print('Building vocabulary...')
    vocab = build_vocab_from_captions(CAPTIONS_FILE, freq_threshold=FREQ_THRESHOLD)
    vocab_size = len(vocab.word2idx)
    print(f'Vocabulary size: {vocab_size}')

    print('Loading datasets...')
    train_dataset = Flickr8kDataset(
        root_dir=DATA_DIR,
        captions_file=CAPTIONS_FILE,
        vocab=vocab,
        transform=transform,
        split='train'
    )
    val_dataset = Flickr8kDataset(
        root_dir=DATA_DIR,
        captions_file=CAPTIONS_FILE,
        vocab=vocab,
        transform=transform,
        split='val'
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    print('Initializing models...')
    encoder = EncoderCNN().to(DEVICE)
    decoder = DecoderRNN(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_SIZE,
        decoder_dim=DECODER_DIM,
        vocab_size=vocab_size
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<PAD>'])
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)

    # Try to resume from checkpoint
    start_epoch, best_loss = load_checkpoint('models/checkpoint_latest.pth', encoder, decoder, optimizer)

    print('Starting training...')
    for epoch in range(start_epoch, NUM_EPOCHS):
        encoder.train()
        decoder.train()
        total_loss = 0
        for i, (images, captions) in enumerate(train_loader):
            images, captions = images.to(DEVICE), captions.to(DEVICE)
            optimizer.zero_grad()
            features = encoder(images)
            outputs, _ = decoder(features, captions[:, :-1])
            targets = captions[:, 1:]
            outputs = outputs.contiguous()
            targets = targets.contiguous()
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            # Save checkpoint every 500 batches
            if (i+1) % 500 == 0:
                torch.save({
                    'epoch': epoch,
                    'batch': i+1,
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'vocab': vocab.word2idx,
                    'best_loss': best_loss
                }, 'models/checkpoint_latest.pth')
                print(f'Checkpoint saved at epoch {epoch+1}, batch {i+1}')
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Training Loss: {avg_loss:.4f}')
        # Validation
        val_loss = evaluate(encoder, decoder, val_loader, criterion, vocab_size, vocab)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Validation Loss: {val_loss:.4f}')
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'vocab': vocab.word2idx
            }, CHECKPOINT_PATH)
            print('Model checkpoint saved!')

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]
    return images, padded_captions

def evaluate(encoder, decoder, val_loader, criterion, vocab_size, vocab):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions in val_loader:
            images, captions = images.to(DEVICE), captions.to(DEVICE)
            features = encoder(images)
            outputs, _ = decoder(features, captions[:, :-1])
            targets = captions[:, 1:]
            outputs = outputs.contiguous()
            targets = targets.contiguous()
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

if __name__ == '__main__':
    main() 