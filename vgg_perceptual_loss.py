import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0,1,2,3], style_layers=[1,2,3]):
        # print(f"input is {input.type()}")
        # print(f"target is {target.type()}")
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        # if self.resize:
        #     input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        #     target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        # print(f"input {x.type()} {x.min()} {x.max()}")
        # print(f"target {y.type()} {y.min()} {y.max()}")  
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                #print(x.shape)
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                #act_x = act_x / (act_x.shape[2] + 1e-8)

                #print(f"act_layer {act_x.shape} {act_x.min()} {act_x.max()}")

                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                #act_y = act_y / (act_y.shape[2] + 1e-8)
                #print(f"act_layer {act_y.shape} {act_y.min()} {act_y.max()}")
                
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)

                gram_x = gram_x*act_x.shape[2]
                gram_y = gram_y*act_y.shape[2]
                # print(f"gram_x {gram_x.type()} {gram_x.min()} {gram_x.max()}")
                # print(f"gram_y {gram_y.type()} {gram_y.min()} {gram_y.max()}")
                gram_loss = torch.nn.functional.l1_loss(gram_x, gram_y)
                gram_loss = gram_loss / (act_x.shape[2] * act_x.shape[1])
                # print(f"gram loss {gram_loss}") 
                # print("")
                loss += gram_loss
        return loss
    

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    vgg_loss = VGGPerceptualLoss().to("cuda")
    x = np.array(Image.open("y_1.png"))
    print(x.shape)
    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).float().to("cuda") / 255.0

    #y = torch.randn(1, 3, 256, 256).to("cuda")
    y = np.array(Image.open("y_fake.png"))
    print(y.shape)
    y = torch.tensor(y).permute(2, 0, 1).unsqueeze(0).float().to("cuda") / 255.0
    print(y.shape)

    loss = vgg_loss(x, y)
    print(loss)
    # tensor(1.0920, device='cuda:0', grad_fn=<AddBackward0>)
    # This loss is not normalized, so it is not directly comparable to the L1 loss.
    # But it can be used as a perceptual loss for GANs.
