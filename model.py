import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze ResNet weights

    def forward(self, images):
        features = self.resnet(images)  # (batch, 2048, H/32, W/32)
        features = self.adaptive_pool(features)  # (batch, 2048, encoded_image_size, encoded_image_size)
        features = features.permute(0, 2, 3, 1)  # (batch, enc_image_size, enc_image_size, 2048)
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_out.size(-1))  # (batch, num_pixels, encoder_dim)
        att1 = self.encoder_att(encoder_out)  # (batch, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (batch, 1, attention_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch, num_pixels)
        alpha = self.softmax(att)  # (batch, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch, encoder_dim)
        return attention_weighted_encoding, alpha

class DecoderRNN(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        # encoder_out: (batch, enc_image_size, enc_image_size, encoder_dim)
        mean_encoder_out = encoder_out.mean(dim=(1, 2))  # (batch, encoder_dim)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        embeddings = self.embedding(captions)  # (batch, seq_len, embed_dim)
        h, c = self.init_hidden_state(encoder_out)  # (batch, decoder_dim)
        decode_len = int(captions.size(1))
        predictions = torch.zeros(batch_size, decode_len, vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, decode_len, num_pixels).to(encoder_out.device)

        for t in range(decode_len):
            attention_weighted_encoding, alpha = self.attention(
                encoder_out, h
            )
            gate = self.sigmoid(self.f_beta(h))  # gating scalar
            attention_weighted_encoding = gate * attention_weighted_encoding
            input_lstm = torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1)
            h, c = self.decode_step(input_lstm, (h, c))
            preds = self.fc(self.dropout(h))
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha
        return predictions, alphas 