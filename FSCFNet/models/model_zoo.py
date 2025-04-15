import torch
from models.sseg.unet import UNet

from models.sseg.mynet import Cloudnet



def get_model(args, device):
    model_name = args.model_name
    in_channels, out_channels = args.in_channels, args.num_classes
    if model_name == "unet":
        model = UNet(in_channels=in_channels, out_channels=out_channels)

    elif model_name == "mynet":
        model = Cloudnet(n_channels=in_channels, n_classes=out_channels)








    else:
        exit("\nError: MODEL \'%s\' is not implemented!\n")

    model = model.to(device)

    if model_name == "mynet":
        inputs = [torch.randn(args.batch_size, args.in_channels, args.img_size, args.img_size, device=device),
                  torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device),
                  torch.randn(args.batch_size, 1, args.img_size, args.img_size, device=device),
                  torch.randn(args.batch_size, 1, args.img_size, args.img_size, device=device)]


    else:
        inputs = [torch.randn(args.batch_size, args.in_channels, args.img_size, args.img_size, device=device)]
    model(*inputs)
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("%s Params: %.2fM" % (model_name, params_num / 1e6))

    return model
