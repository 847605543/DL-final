from .detector import Dist_Detector, save_dist_model, load_dist_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data, load_dist_data
from torchvision import transforms

def train(args):
    from os import path
    
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Dist_Detector().to(device)
    if args.continue_training:
        model = load_dist_model().to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(transforms) if inspect.isclass(v)})
    
    aim_loss = torch.nn.MSELoss(reduction='none')

    global_step = 0

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(transforms) if inspect.isclass(v)})
    dist_data = load_dist_data('data/with_Ball', num_workers=4, transform=transform)
    
    for epoch in range(args.num_epoch):
        model.train()    
        for img, aim in dist_data:
            aim = torch.tensor(np.asarray(aim))
            #print(len(img))
            img = torch.tensor(img)
            img = img.to(device)
            aim = aim.to(device)

            pred = model(img)
            pred = pred[:,0]
            # Continuous version of focal loss
            det_loss_val = (aim_loss(pred,aim)).mean()
            loss_val = det_loss_val
            
            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, gt_det, det, global_step)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            opt.zero_grad()
            loss_val.backward()
            opt.step()
            global_step += 1
            if(global_step%10 ==0):
                print(pred[0],aim[0])
        print('Epoch ', epoch)
        print(pred[0],aim[0])
        save_dist_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1),RandomHorizontalFlip(), ToTensor()])')
    parser.add_argument('-w', '--size-weight', type=float, default=0.01)
    
    args = parser.parse_args()
    train(args)
