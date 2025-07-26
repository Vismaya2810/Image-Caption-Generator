import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Flickr8kDataset, build_vocab_from_captions
from model import EncoderCNN, DecoderRNN
import nltk
from nltk.translate.bleu_score import corpus_bleu
import os
from PIL import Image

DATA_DIR = 'data/Flickr8k_Dataset'
CAPTIONS_FILE = 'data/Flickr8k.token.txt'
CHECKPOINT_PATH = 'models/best_model.pth'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def load_model(vocab_size):
    encoder = EncoderCNN().to(DEVICE)
    decoder = DecoderRNN(
        attention_dim=256,
        embed_dim=256,
        decoder_dim=512,
        vocab_size=vocab_size,
        encoder_dim=2048
    ).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    return encoder, decoder

def generate_caption(encoder, decoder, image, vocab, max_len=30, already_tensor=False):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        assert '<EOS>' in vocab.word2idx, "Vocabulary missing <EOS> token!"
        if not already_tensor:
            image = transform(image).unsqueeze(0).to(DEVICE)
        else:
            image = image.unsqueeze(0) if image.dim() == 3 else image
        features = encoder(image)
        caption = [vocab.word2idx['<SOS>']]
        h, c = decoder.init_hidden_state(features)
        for step in range(max_len):
            cap_tensor = torch.tensor([caption[-1]]).to(DEVICE)
            emb = decoder.embedding(cap_tensor)  # [1, embed_dim]
            if h.dim() == 1:
                h = h.unsqueeze(0)
            if h.dim() == 3 and h.shape[1] == 1:
                h = h.squeeze(1)
            if c.dim() == 1:
                c = c.unsqueeze(0)
            if c.dim() == 3 and c.shape[1] == 1:
                c = c.squeeze(1)
            attention_weighted_encoding, alpha = decoder.attention(features.view(1, -1, features.size(-1)), h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            input_lstm = torch.cat([emb, attention_weighted_encoding], dim=1)  # [1, embed_dim + encoder_dim]
            h, c = decoder.decode_step(input_lstm, (h, c))
            if h.dim() == 1:
                h = h.unsqueeze(0)
            if h.dim() == 3 and h.shape[1] == 1:
                h = h.squeeze(1)
            if c.dim() == 1:
                c = c.unsqueeze(0)
            if c.dim() == 3 and c.shape[1] == 1:
                c = c.squeeze(1)
            preds = decoder.fc(h)
            predicted = preds.argmax(1).item()
            word = vocab.idx2word[predicted]
            print(f"Step {step}: predicted word = {word}")
            caption.append(predicted)
            if predicted == vocab.word2idx['<EOS>']:
                print("Generated <EOS>, stopping.")
                break
        else:
            print("Warning: <EOS> token not generated within max_len steps.")
        words = [vocab.idx2word[idx] for idx in caption[1:] if idx not in (vocab.word2idx['<EOS>'], vocab.word2idx['<PAD>'])]
        return ' '.join(words)

def main():
    print('Loading vocabulary and validation set...')
    vocab = build_vocab_from_captions(CAPTIONS_FILE)
    val_dataset = Flickr8kDataset(
        root_dir=DATA_DIR,
        captions_file=CAPTIONS_FILE,
        vocab=vocab,
        transform=transform,
        split='val'
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    encoder, decoder = load_model(len(vocab.word2idx))

    print('Evaluating BLEU score on validation set...')
    references = []
    hypotheses = []
    for i, (image, caption_tensor) in enumerate(val_loader):
        image = image.to(DEVICE)
        gt_caption = caption_tensor[0].tolist()
        gt_words = [vocab.idx2word[idx] for idx in gt_caption if idx not in (vocab.word2idx['<SOS>'], vocab.word2idx['<EOS>'], vocab.word2idx['<PAD>'])]
        generated = generate_caption(encoder, decoder, image[0], vocab, already_tensor=True)
        references.append([gt_words])
        hypotheses.append(generated.split())
        if i < 5:
            print(f'GT: {" ".join(gt_words)}')
            print(f'PR: {generated}\n')
        if i >= 9:
            break
    bleu4 = corpus_bleu(references, hypotheses)
    print(f'Validation BLEU-4 score: {bleu4:.4f}')

if __name__ == '__main__':
    main() 