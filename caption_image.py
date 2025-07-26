import torch
from torchvision import transforms
from PIL import Image
from model import EncoderCNN, DecoderRNN
from dataset import build_vocab_from_captions
import sys

DATA_DIR = 'data/Flickr8k_Dataset'
CAPTIONS_FILE = 'data/Flickr8k.token.txt'
CHECKPOINT_PATH = 'models/best_model.pth'
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

def generate_caption(encoder, decoder, image, vocab, max_len=30):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        assert '<EOS>' in vocab.word2idx, "Vocabulary missing <EOS> token!"
        image = transform(image).unsqueeze(0).to(DEVICE)
        features = encoder(image)
        caption = [vocab.word2idx['<SOS>']]
        h, c = decoder.init_hidden_state(features)
        for step in range(max_len):
            cap_tensor = torch.tensor([caption[-1]]).to(DEVICE)
            emb = decoder.embedding(cap_tensor).unsqueeze(0)
            attention_weighted_encoding, alpha = decoder.attention(features.view(1, -1, features.size(-1)), h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            input_lstm = torch.cat([emb.squeeze(0), attention_weighted_encoding], dim=1)
            h, c = decoder.decode_step(input_lstm, (h, c))
            if h.dim() == 1:
                h = h.unsqueeze(0)
            if c.dim() == 1:
                c = c.unsqueeze(0)
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
    if len(sys.argv) < 2:
        print('Usage: python src/caption_image.py <image_path>')
        return
    image_path = sys.argv[1]
    image = Image.open(image_path).convert('RGB')
    vocab = build_vocab_from_captions(CAPTIONS_FILE)
    encoder, decoder = load_model(len(vocab.word2idx))
    caption = generate_caption(encoder, decoder, image, vocab)
    print(f'Generated Caption: {caption}')

if __name__ == '__main__':
    main() 