import os
import torch
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from model import D_Loss, R_Loss
import os
import argparse
import torch
import torchvision
import torchvision.transforms as tf
from train import train_model
from model import R_Net, D_Net, R_Loss, D_Loss, R_WLoss, D_WLoss, Dataset



def test_model(r_net: torch.nn.Module,
               d_net: torch.nn.Module,
               valid_dataset: torch.utils.data.Dataset,
               batch_size: int = 1,
               pin_memory: bool = True,
               num_workers: int = 1,
               epoch_step: int = 1,
               save_step: int = 5,
               rec_loss_bound: float = 0.1,
               lambd: float = 0.2,
               save_path: tuple = ('.', 'r_net.pth', 'd_net.pth')):

    device: torch.device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_path = os.path.join(save_path[0], 'models')
    r_net_path = os.path.join(model_path, save_path[1])
    d_net_path = os.path.join(model_path, save_path[2])

    assert (os.path.exists(r_net_path) and os.path.exists(d_net_path))

    test_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)

    start = timer()

    with torch.no_grad():
        for data in test_loader:
            x_real = data[0].to(device)
            labels = data[1]
            x_fake = r_net(x_real)
            y_disci = d_net(x_fake)
            print(y_disci)
            pass;


def plt_curvese(x_real,x_fake,need_transform):
    
    plt.figure();
    plt.subplot(121)
    plt.imshow(x_real)
    plt.subplot(122)
    plt.imshow(x_fake)
    plt.show()

def main(args):
    
    test_raw_dataset = valid_raw_dataset = torchvision.datasets.MNIST(root='./mnist',
                                                   train=False,
                                                   download=True,
                                                   transform=tf.Compose([tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]))
    
    
    test_dataset = Dataset(test_raw_dataset, [9])
    
    
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f'Using GPU {torch.cuda.get_device_name()}')
        print(torch.cuda.get_device_properties(device))
    else:
        device = torch.device('cpu')
        print('Using CPU')
    
    
    assert(args.load_path)
    
    r_net_path = os.path.join(args.load_path, args.r_load_path)
    d_net_path = os.path.join(args.load_path, args.d_load_path)
    
    r_net = torch.load(r_net_path).to(device)
    
    print(f'Loaded R_Net from {r_net_path}')
    
    d_net = torch.load(d_net_path).to(device)
    
    print(f'Loaded D_Net from {d_net_path}')

    test_model(r_net,d_net,test_dataset)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_false',
                        help='Turns off gpu training')
    parser.add_argument('--save_path', '-sp', type=str, default='.',
                        help='Path to a folder where metrics and models will be saved')
    parser.add_argument('--d_save_path', '-dsp', type=str,
                        default='d_net.pth', help='Name of .pth file for d_net to be saved')
    parser.add_argument('--r_save_path', '-rsp', type=str,
                        default='r_net.pth', help='Name of .pth file for r_net to be saved')
    parser.add_argument('--load_path', '-lp', default='models',
                        help='Path to a folder from which models will be loaded')
    parser.add_argument('--d_load_path', '-dlp', type=str, default='d_net.pth',
                        help='Name of .pth file for d_net to be loaded')
    parser.add_argument('--r_load_path', '-rlp', type=str, default='r_net.pth',
                        help='Name of .pth file for r_net to be loaded')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for data loader')
    parser.add_argument('--rec_bound', type=float, default=0.1,
                        help='Upper bound of reconstruction loss')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Initial learning rate')
    parser.add_argument('--nw', type=int, default=1,
                        help='num_workers for DataLoader')
    parser.add_argument('--sstep', type=int, default=5,
                        help='Step in epochs for saving models')
    parser.add_argument('--std', type=float, default=0.155,
                        help='Standart deviation for noise in R_Net')
    parser.add_argument('--lambd', type=float, default=0.2,
                        help='Lambda parameter for LR loss')
    parser.add_argument('--cat', action='store_true',
                        help='Turns on skip connections with concatenation')
    parser.add_argument('--res', action='store_true',
                        help='Turns on residual connections')
    args = parser.parse_args()

    main(args)

    