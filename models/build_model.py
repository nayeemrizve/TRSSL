import torch


def build_model(args, ema=False):
    if args.dataset in ['cifar10', 'cifar100']:
        from . import resnet_cifar as models 
    elif args.dataset == 'tinyimagenet':
        from . import resnet_tinyimagenet as models
    else:
        from . import resnet as models

    if args.arch == 'resnet18':
        model = models.resnet18(no_class=args.no_class)
    if args.arch == 'resnet50':
        model = models.resnet50(no_class=args.no_class)
    
    # use dataparallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model