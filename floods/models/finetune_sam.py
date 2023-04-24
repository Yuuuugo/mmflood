import torch
import torch.nn as nn
from floods.losses.functional import DiceLoss


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



class FineTunedSAM(nn.Module):
    def __init__(
                self, 
                checkpoint_path : str ,
                model_type : str = "vit_l", 
                number_of_channels : int = 2,
                lr : float = 3e-4
                ):
        super().__init__()

        if number_of_channels != 3:
            self.first_layer = nn.Conv2d(number_of_channels, 3, padding="same", kernel_size=(3,3) )
        else:
            self.first_layer = nn.Identity()

        self.sam_model = sam_model_registry[model_type](checkpoint_path)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam_model)
        self.loss = DiceLoss()

        params = [{"params" : self.sam_model.mask_decoder.parameters()}]
        params += [{'params': self.first_layer._parameters, 'weight_decay': 0}]
        #self.optimizer =  torch.optim.Adam(params)

    def forward(self,x):
        x = self.first_layer(x)
        print(x.shape)
        x = self.mask_generator.generate(x)
        return x 
    


if __name__ == "__main__":
    sam_checkpoint = "/home/gabrielidis/Project/segment-anything/sam_vit_l_0b3195.pth"
    model = FineTunedSAM(checkpoint_path= sam_checkpoint)
    x = torch.rand((2,512,512))
    print(model.first_layer.parameters())
    for param in model.first_layer.parameters():
        print(param.shape)
    #y = model(x)
    #print(y[0]["segmentation"].shape)