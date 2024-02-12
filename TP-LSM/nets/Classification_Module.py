import torch.nn as nn

class Classification_Module(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.linear_fuse = nn.Conv1d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1)
        self.linear_pred = nn.Conv1d(embedding_dim, num_classes, kernel_size=1)

        self.dropout = nn.Dropout()

    def forward(self, concat_feature):
        # Classification Branch
        x = self.linear_fuse(concat_feature)
        x = self.dropout(x)
        x = self.linear_pred(x)
        x = x.permute(0, 2, 1)

        return x
