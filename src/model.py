"""
Generic Transformer-based Binary Classifier for Hate Speech Detection
Supports any transformer model (BERT, RoBERTa, etc.) through AutoModel
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TransformerBinaryClassifier(nn.Module):
    """
    Transformer-based binary classifier for hate speech detection.
    """

    def __init__(self, model_name, dropout=0.1, pooling_type='cls', use_multi_layers=False):
        """
        Initialize the binary classifier.

        Args:
            model_name (str): Name or path of pre-trained transformer model
            dropout (float): Dropout rate for regularization
            pooling_type (str): Type of pooling ('cls', 'mean', 'max')
            use_multi_layers (bool): Whether to use multiple hidden layers
        """
        super(TransformerBinaryClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = config.hidden_size
        self.pooling_type = pooling_type
        self.use_multi_layers = use_multi_layers

        # Classification head for binary classification
        head_input_size = self.hidden_size
        if use_multi_layers:
            head_input_size = self.hidden_size * 4  # Concatenate last 4 layers

        self.classifier = nn.Sequential(
            nn.Linear(head_input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # Single output for binary classification
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            labels: Ground truth labels (optional, for loss calculation)

        Returns:
            dict: Dictionary containing loss (if labels provided) and logits
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.use_multi_layers:
            # Concatenate last 4 hidden layers' [CLS] tokens
            hidden_states = outputs.hidden_states
            cls_outputs = [hidden_states[i][:, 0, :] for i in range(-1, -5, -1)]
            cls_output = torch.cat(cls_outputs, dim=-1)
        else:
            if self.pooling_type == 'cls':
                cls_output = outputs.last_hidden_state[:, 0, :]
            elif self.pooling_type == 'mean':
                # Mean pooling across tokens, respecting attention mask
                mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                cls_output = sum_embeddings / sum_mask
            elif self.pooling_type == 'max':
                # Max pooling across tokens, respecting attention mask
                mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                masked_embeddings = outputs.last_hidden_state.clone()
                masked_embeddings[mask == 0] = -1e9  # Set masked tokens to very low value
                cls_output = torch.max(masked_embeddings, 1)[0]
            else:
                cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)  # Shape: (batch_size, 1)

        loss = None
        if labels is not None:
            labels = labels.view(-1, 1)  # Reshape to (batch_size, 1)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits}


    def save_pretrained(self, save_path):
        """
        Save the model in a format that can be reloaded.
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save encoder
        self.encoder.save_pretrained(os.path.join(save_path, 'encoder'))
        
        # Save classifier
        torch.save(
            self.classifier.state_dict(),
            os.path.join(save_path, 'classifier.pt')
        )
        
        # Save config
        config = {
            'model_name': self.encoder.config._name_or_path,
            'pooling_type': self.pooling_type,
            'use_multi_layers': self.use_multi_layers,
            'dropout': self.classifier[2].p if len(self.classifier) > 2 else 0.1
        }
        with open(os.path.join(save_path, 'model_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Model saved to: {save_path}")

    @classmethod
    def from_pretrained(cls, load_path, device='cpu', **kwargs):
        """
        Load a pre-trained model.
        """
        config_path = os.path.join(load_path, 'model_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model = cls(
                model_name=os.path.join(load_path, 'encoder'),
                dropout=config.get('dropout', 0.1),
                pooling_type=config.get('pooling_type', 'cls'),
                use_multi_layers=config.get('use_multi_layers', False),
                **kwargs
            )
            
            # Load classifier weights
            classifier_path = os.path.join(load_path, 'classifier.pt')
            if os.path.exists(classifier_path):
                classifier_state = torch.load(classifier_path, map_location=device)
                model.classifier.load_state_dict(classifier_state)
        else:
            # Fallback to loading as a standard HF model if possible, 
            # though this class is specifically for our custom head.
            model = cls(model_name=load_path, **kwargs)
            
        return model.to(device)


    def freeze_base_layers(self):
        """
        Freeze encoder parameters for feature extraction.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

        frozen_params = sum(p.numel() for p in self.encoder.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Frozen {frozen_params:,} parameters out of {total_params:,} total parameters")
        print(f"Trainable parameters: {total_params - frozen_params:,}")

    def unfreeze_base_layers(self):
        """
        Unfreeze encoder parameters for fine-tuning.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"All parameters unfrozen. Trainable parameters: {trainable_params:,}")
