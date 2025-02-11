from typing import List, Literal, Tuple, Union
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, CharErrorRate

class MTLNetwork(nn.Module):
  def __init__(self, feature_extractor, num_initials, num_finals, num_tones, lstm_hidden_dim=256, hard_param_sharing=True):
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
    self.initial_head = nn.Linear(256 * 2, num_initials)
    self.final_head = nn.Linear(256 * 2, num_finals)
    self.tone_head = nn.Linear(256 * 2, num_tones)
    self.sanity_head = nn.Linear(256 * 2, 2) # sane or insane
    
    self.num_initials = num_initials
    self.num_finals = num_finals
    self.num_tones = num_tones
    
    # class based metrics
    self.initial_accuracy = Accuracy(task="multiclass", num_classes=num_initials)
    self.initial_f1 = F1Score(task="multiclass", num_classes=num_initials)

    self.final_accuracy = Accuracy(task="multiclass", num_classes=num_finals)
    self.final_f1 = F1Score(task="multiclass", num_classes=num_finals)

    self.tones_accuracy = Accuracy(task="multiclass", num_classes=num_tones)
    self.tones_f1 = F1Score(task="multiclass", num_classes=num_tones)

    self.sanity_accuracy = Accuracy(task="binary")
    self.sanity_f1 = F1Score(task="binary")

    self.accuracy_metrics = [self.initial_accuracy, self.final_accuracy, self.tones_accuracy, self.sanity_accuracy]
    self.f1_metrics = [self.initial_f1, self.final_f1, self.tones_f1, self.sanity_f1]
  
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
    sanity_out = self.tone_head(pooled_features)

    # TODO add masking to make certain combos not possible!!
    
    return initials_out, finals_out, tones_out, sanity_out

  # TODO might consider uncertainty-based weighting, where the loss weights are also learned dynamically
  def compute_loss(self, predictions, targets, criterion=nn.CrossEntropyLoss(), weights: List=None) -> float:
    if weights is None:
      weights = [1.0] * len(predictions)

    target_list = [targets[:, i]for i in range(targets.shape[1])]

    total_loss = 0
    for pred, target, weight in zip(predictions, target_list, weights):
      loss = criterion(pred, target)
      total_loss += loss * weight

    return total_loss

  def compute_metrics(self, predictions, targets, metric: Literal["f1", "acc"], average: bool=True) -> Union[float | Tuple[float, float, float, float]]:
    assert metric in ['f1', 'acc']
    target_list = [targets[:, i] for i in range(targets.shape[1])]

    metrics = self.f1_metrics if metric == "f1" else self.accuracy_metrics

    computed_metrics = []
    for prediction, target, evaluator in zip(predictions, target_list, metrics):
      computed_metrics.append(evaluator(prediction, target))

    if average:
      return sum(computed_metrics) / len(computed_metrics)

    return tuple(computed_metrics) 