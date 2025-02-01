from typing import Tuple
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, CharErrorRate


class MTLNetwork(nn.Module):
  def __init__(self, feature_extractor, num_initials, num_finals, num_tones=5, lstm_hidden_dim=256, hard_param_sharing=True):
    super(MTLNetwork, self).__init__()
    
    # remove the last two layers of resnet (AdaptiveAvgPool2d and Linear)
    self.shared_encoder = nn.Sequential(*list(feature_extractor.children())[:-2])
    self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # if no lstm have this

    # gap to compress spatial feats, lstm cares ab temporal
    # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, None))  # Keep time dim, reduce freq dim
    
    # rnn following up - maybe transformer model
    # self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)

    # shared layer to get more connections
    # self.shared_fc = nn.Linear(lstm_hidden_dim * 2, 256) # *2 for bidirectional LSTM
    
    # task specific heads
    self.initial_head = nn.Linear(256, num_initials)
    self.final_head = nn.Linear(256, num_finals)
    self.tone_head = nn.Linear(256, num_tones)
    
    # class based metrics
    self.initial_accuracy = Accuracy(task="multiclass", num_classes=num_initials)
    self.initial_f1 = F1Score(task="multiclass", num_classes=num_initials)

    self.final_accuracy = Accuracy(task="multiclass", num_classes=num_finals)
    self.final_f1 = F1Score(task="multiclass", num_classes=num_finals)

    self.tones_accuracy = Accuracy(task="multiclass", num_classes=num_tones)
    self.tones_f1 = F1Score(task="multiclass", num_classes=num_tones)
  
  def forward(self, x):
    """
      x: (batch_size, channel=1, freq_bins, time_steps) -> Spectrogram input
    """
    feature_map = self.shared_encoder(x) # personal resnet for now
    # pooling after resnet, use the other one for lstm
    pooled_features = self.global_avg_pool(feature_map)
    pooled_features = pooled_features.view(pooled_features.size(0), -1)

    # to remove freq dim 
    # pooled_features = self.global_avg_pool(feature_map) # Shape: (batch_size, 512, 1, time_steps')
    # pooled_features = pooled_features.squeeze(2) # Shape: (batch_size, 512, time_steps')
    
    # lstm wants it in (batch_size, time_steps, feature_dim)
    # features_seq = pooled_features.permute(0, 2, 1)
    
    # lstm_map, _ = self.lstm(features_seq)
    # lstm_out = lstm_map[:, -1, :] # takes just the last time step, which should hold most of the info like attention heads    
    # lstm_out = torch.mean(lstm_map, dim=1)  # simple mean pooling (?)

    # after lstm, put thru shared fc layer
    # shared_rep = nn.ReLU()(self.shared_fc(lstm_out))
    
    # then make task predictions
    initials_out = self.initial_head(pooled_features)
    finals_out = self.final_head(pooled_features)
    tones_out = self.tone_head(pooled_features)

    # TODO add masking to make certain combos not possible!!
    
    return initials_out, finals_out, tones_out

  # TODO consider uncertainty-based weighting, where the loss weights are also learned dynamically
  def compute_loss(predictions, targets, weights=[1.0, 1.0, 1.0]):
    initial_preds, final_preds, tone_preds = predictions
    initial_target, final_target, tone_target = targets
    
    initial_loss = nn.CrossEntropyLoss(initial_preds, initial_target)
    final_loss = nn.CrossEntropyLoss(final_preds, final_target)
    tone_loss = nn.CrossEntropyLoss(tone_preds, tone_target)

    total_loss = (initial_loss * weights[0]) + (final_loss * weights[1]) + (tone_loss * weights[2])
    return total_loss
  
  def compute_accuracy(self, predictions, targets, average=True) -> int | Tuple[int, int, int]:
    initial_preds, final_preds, tone_preds = predictions
    initial_target, final_target, tone_target = targets

    initial_acc = self.initial_accuracy(initial_preds, initial_target)
    final_acc = self.final_accuracy(final_preds, final_target)
    tone_acc = self.initial_accuracy(tone_preds, tone_target)

    if average:
      return (initial_acc + final_acc + tone_acc) / 3

    return initial_acc, final_acc, tone_acc
  
  def compute_f1(self, predictions, targets, average=True) -> int | Tuple[int, int, int]:
    initial_preds, final_preds, tone_preds = predictions
    initial_target, final_target, tone_target = targets

    initial_f1 = self.initial_f1(initial_preds, initial_target)
    final_f1 = self.final_f1(final_preds, final_target)
    tone_f1 = self.initial_f1(tone_preds, tone_target)

    if average:
      return (initial_f1 + final_f1 + tone_f1) / 3

    return initial_f1, final_f1, tone_f1
