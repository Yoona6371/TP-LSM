import torch.nn as nn



from .Classification_Module import Classification_Module
from .PFConnect import PFConnect
from .LSM import TPLSM_Encoder
from .ASL import AsymmetricLossOptimized, AsymmetricLoss


class TPLSM(nn.Module):
    """
    MS-TCT for action detection
    [], 3, 8, 4, 1024, 512
    """
    def __init__(self, inter_channels, s_size, num_block, mlp_ratio, in_feat_dim, final_embedding_dim, num_classes):
        super(TPLSM, self).__init__()

        self.dropout=nn.Dropout()

        self.TPLSM_Encoder=TPLSM_Encoder(
            in_feat_dim=in_feat_dim, embed_dims=inter_channels, 
            s_size=s_size, mlp_ratio=mlp_ratio, num_block=num_block)

        self.PFConnect=PFConnect(inter_channels=inter_channels, embedding_dim=final_embedding_dim)

        self.Classfication_Module=Classification_Module(num_classes=num_classes, embedding_dim=final_embedding_dim)

        self.APLLoss = AsymmetricLoss()

    def forward(self, inputs, label):
    

        inputs = self.dropout(inputs)
        
        x = self.TPLSM_Encoder(inputs)

        concat_feature = self.PFConnect(x)

        x = self.Classfication_Module(concat_feature)

        # loss
        l = self.APLLoss(x, label)


        return x, l 






