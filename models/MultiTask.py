from typing import List, Literal, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, CharErrorRate

class WeightedFocalLoss(nn.Module):
  def __init__(self, alpha=1, gamma=2):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
        
  def forward(self, inputs, targets, class_weights=None):
    num_classes = inputs.shape[1]
        
    if class_weights is not None:
      # create a full tensor of ones for all possible classes
      full_weights = torch.ones(num_classes, device=inputs.device)
      unique_classes = torch.unique(targets).long()

      # update weights only for valid classes present in the batch
      valid_classes = unique_classes[unique_classes < num_classes]
      for idx, class_idx in enumerate(valid_classes):
        if idx < len(class_weights):
          full_weights[class_idx] = class_weights[idx]
      
      ce_loss = F.cross_entropy(inputs, targets, weight=full_weights, reduction='none')
    else:
      ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
    pt = torch.exp(-ce_loss)
    focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
    return focal_loss.mean()


class MTLNetwork(nn.Module):
  # TODO maybe do something with soft param sharing
  def __init__(self, feature_extractor, num_initials, num_finals, num_tones, lstm_hidden_dim=256, hard_param_sharing=True):
    super(MTLNetwork, self).__init__()
    
    # remove the last two layers of resnet (AdaptiveAvgPool2d and Linear)
    self.shared_encoder = nn.Sequential(*list(feature_extractor.children())[:-2])

    # gap to compress spatial feats, lstm cares ab temporal
    self.global_avg_pool = nn.AdaptiveAvgPool2d((1, None))  # Keep time dim, reduce freq dim

    # rnn following up - maybe transformer model
    self.lstm = nn.LSTM(
      input_size=512, 
      hidden_size=lstm_hidden_dim, 
      num_layers=2, 
      batch_first=True, 
      bidirectional=True, 
      dropout=0.2
    )

    # Add attention layer to focus on important temporal features
    self.attention = nn.Sequential(
      nn.Linear(lstm_hidden_dim * 2, 1),
      nn.Softmax(dim=1)
    )

    self.dropout = nn.Dropout(0.5)
    self.bn1 = nn.BatchNorm1d(lstm_hidden_dim * 2)

    # shared layer to get more connections
    self.shared_fc = nn.Linear(lstm_hidden_dim * 2, 256) # *2 for bidirectional LSTM
    
    # task specific heads
    self.initial_head = nn.Linear(256, num_initials)
    self.final_head = nn.Linear(256, num_finals)
    self.tone_head = nn.Linear(256, num_tones)
    self.sanity_head = nn.Linear(256, 2) # sane or insane - phonotactic constraint learning

    # -------------------------------------------------- #
    
    # used for debugging
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

    # to remove freq dim 
    pooled_features = self.global_avg_pool(feature_map) # Shape: (batch_size, 512, 1, time_steps')
    pooled_features = pooled_features.squeeze(2) # Shape: (batch_size, 512, time_steps')
    
    # lstm wants it in (batch_size, time_steps, feature_dim)
    features_seq = pooled_features.permute(0, 2, 1)
    
    # apply some attention mechanism to the lstm output
    lstm_out, _ = self.lstm(features_seq)
    attention_weights = self.attention(lstm_out)
    lstm_out = torch.sum(attention_weights * lstm_out, dim=1)

    # after lstm, put thru shared fc layer
    lstm_out = self.dropout(lstm_out)
    lstm_out = self.bn1(lstm_out)
    shared_rep = nn.ReLU()(self.shared_fc(lstm_out))
    
    # then make task predictions
    initials_out = self.initial_head(shared_rep)
    finals_out = self.final_head(shared_rep)
    tones_out = self.tone_head(shared_rep)
    sanity_out = self.sanity_head(shared_rep) # for phonotactic constraint learning

    # TODO add masking to make certain combos not possible!!
    
    return initials_out, finals_out, tones_out, sanity_out

  # TODO might consider uncertainty-based weighting, where the loss weights are also learned dynamically
  def compute_loss(self, predictions, targets, criterion=nn.CrossEntropyLoss(), weights: List=None, l1_lambda: float=0.01, l2_lambda: float=0.01) -> float:
    if weights is None:
      weights = [1.0] * len(predictions)

    target_list = [targets[:, i] for i in range(targets.shape[1])]
    
    total_loss = 0
    for pred, target, weight in zip(predictions, target_list, weights):
      loss = criterion(pred, target)
      total_loss += loss * weight
    
    l1_reg = torch.tensor(0., requires_grad=True)
    l2_reg = torch.tensor(0., requires_grad=True)
    for param in self.parameters():
      l1_reg = l1_reg + torch.norm(param, 1)
      l2_reg = l2_reg + torch.norm(param, 2)
    
    return total_loss + l1_lambda * l1_reg + l2_lambda * l2_reg

  def compute_metrics(self, predictions, targets, metric: Literal["f1", "acc"], average: bool=True) -> Union[float | Tuple[float, float, float, float]]:
    assert metric in ['f1', 'acc']
    target_list = [targets[:, i] for i in range(targets.shape[1])]

    metrics = self.f1_metrics if metric == "f1" else self.accuracy_metrics

    computed_metrics = []
    for prediction, target, evaluator in zip(predictions, target_list, metrics):
      computed_metrics.append(evaluator(prediction.argmax(dim=1), target))

    if average:
      return sum(computed_metrics) / len(computed_metrics)

    return tuple(computed_metrics) 

