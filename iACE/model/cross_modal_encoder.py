import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalEncoder(nn.Module):
    def __init__(
        self,
        in_lan_feature_dim,
        in_vis_feature_dim,
        out_fusion_dim,
        num_classes,
        dropout_prob,
        loss_fn,
    ):
        super(CrossModalEncoder, self).__init__()
        self.loss_fn = loss_fn

        # layers
        self.lan_fc = nn.Sequential(
            nn.Linear(in_features=in_lan_feature_dim, out_features=in_lan_feature_dim),
            nn.ReLU()
        )
        self.vis_fc = nn.Sequential(
            nn.Linear(in_features=in_vis_feature_dim, out_features=in_vis_feature_dim),
            nn.ReLU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(in_features=in_lan_feature_dim + in_vis_feature_dim, out_features=out_fusion_dim),
            nn.ReLU()
        )
        self.representation_fc = nn.Linear(in_features=out_fusion_dim, out_features=out_fusion_dim)
        self.logits_fc = nn.Linear(in_features=out_fusion_dim, out_features=num_classes)

        # function
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, img_features, txt_features, labels):
        # fused embedding
        Xs = torch.cat(
            [self.lan_fc(img_features), self.vis_fc(txt_features)], dim=1
        )

        # two fully connected layers
        fused = self.dropout(self.fuse(Xs))
        img_aug_lan_representation = self.dropout(self.representation_fc(fused))

        # linear layer to logits
        logits = F.tanh(self.logits_fc(img_aug_lan_representation))

        # calculate loss
        loss = self.loss_fn(logits, labels) if labels is not None else None

        return logits, loss

