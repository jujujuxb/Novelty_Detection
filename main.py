import os
import argparse
import torch
import torchvision
import torchvision.transforms as tf
from train import train_model
from model import R_Net, D_Net, R_Loss, D_Loss, R_WLoss, D_WLoss, Dataset
from TrajectoriesDataSet import TrajectoryDataset
import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_score, recall_score, roc_auc_score, log_loss, auc
# plt.rcParams['font.family'] = ['sans-serif']
# # plt.rcParams['font.size'] = '20'
# plt.rcParams['font.sans-serif'] = ['SimHei']


def main(args):

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Uncomment this if HTTP error happened

    # new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    # torchvision.datasets.MNIST.resources = [
    #    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    #    for url, md5 in torchvision.datasets.MNIST.resources
    # ]

    # train_raw_dataset = torchvision.datasets.MNIST(root='./mnist',
    #                                                train=True,
    #                                                download=True,
    #                                                transform=tf.Compose([tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]))

    # valid_raw_dataset = torchvision.datasets.MNIST(root='./mnist',
    #                                                train=False,
    #                                                download=True,
    #                                                transform=tf.Compose([tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]))
    # Train and validate only on pictures of 1

    # train_dataset = Dataset(train_raw_dataset, [1])
    # valid_dataset = Dataset(valid_raw_dataset, [1])

    train_dataset = TrajectoryDataset(
        dataset_dir=args.data_path, labels={1, 2, 3, 4, 5, 6, 15})
    valid_dataset = None

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f'Using GPU {torch.cuda.get_device_name()}')
        print(torch.cuda.get_device_properties(device))
    else:
        device = torch.device('cpu')
        print('Using CPU')

    if args.load_path:
        r_net_path = os.path.join(args.load_path, args.r_load_path)
        d_net_path = os.path.join(args.load_path, args.d_load_path)
        r_net = torch.load(r_net_path).to(device)
        print(f'Loaded R_Net from {r_net_path}')
        d_net = torch.load(d_net_path).to(device)
        print(f'Loaded D_Net from {d_net_path}')
    else:
        r_net = R_Net(in_channels=3, std=args.std,
                      skip=args.res, cat=args.cat).to(device)
        d_net = D_Net(in_resolution=(120, 120), in_channels=3).to(device)
        print('Created models')

    # TRAINING PARAMETERS

    save_path = (args.save_path, args.r_save_path, args.d_save_path)
    optim_r_params = {'alpha': 0.9, 'weight_decay': 1e-9}
    optim_d_params = {'alpha': 0.9, 'weight_decay': 1e-9}

    model = train_model(r_net, d_net, train_dataset, valid_dataset, R_Loss, D_Loss, optimizer_class=torch.optim.RMSprop,
                        device=device, batch_size=args.batch_size, optim_r_params=optim_r_params, optim_d_params=optim_d_params,
                        learning_rate=args.lr, rec_loss_bound=args.rec_bound,
                        save_step=args.sstep, num_workers=args.nw, save_path=save_path, lambd=args.lambd)


def plt_imsave(x, pred, root_path, idx):
    src_img = np.transpose(x[0].detach().cpu().numpy(), (1, 2, 0))
    recur_img = np.transpose(pred[0].detach().cpu().numpy(), (1, 2, 0))

    # img = np.concatenate((src_img, recur_img))

    resimg = cv2.hconcat((src_img, recur_img))

    cv2.imwrite(os.path.join(root_path, "{}.jpg".format(idx)), resimg)


def test(args):

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    assert (torch.cuda.is_available())

    device = torch.device('cuda:0')

    r_net_path = os.path.join(args.load_path, args.r_load_path)
    d_net_path = os.path.join(args.load_path, args.d_load_path)
    r_net = torch.load(r_net_path).to(device)
    print(f'Loaded R_Net from {r_net_path}')
    d_net = torch.load(d_net_path).to(device)
    print(f'Loaded D_Net from {d_net_path}')

    # TRAINING PARAMETERS

    in_dataset = TrajectoryDataset(
        dataset_dir=args.test_path, labels={1, 2, 3, 4, 5, 6, 15})

    in_dataorder = torch.utils.data.DataLoader(
        in_dataset, shuffle=False, batch_size=1)

    r_net.cuda()
    d_net.cuda()

    critation = R_Loss

    losses = []
    labels = []

    r_net.eval()
    d_net.eval()

    png_root_path = os.path.join(os.getcwd(), "RecurImg")

    idx = 0

    for data, y in in_dataorder:
        x = data.cuda()

        pred = r_net(x)

        loss = critation(d_net, x, pred, args.lambd)['rec_loss']

        losses.append(loss.item())
        labels.append(0)

    out_dataset = TrajectoryDataset(
        dataset_dir=args.data_path, labels={7, 8, 9, 10, 11, 12, 13, 14})

    out_dataloader = torch.utils.data.DataLoader(
        out_dataset, shuffle=False, batch_size=1)

    for data, y in out_dataloader:
        x = data.cuda()

        pred = r_net(x)

        loss = critation(d_net, x, pred, args.lambd)['rec_loss']

        losses.append(loss.item())
        labels.append(1)

    labels = np.array(labels)
    losses = np.array(losses)

    metrics_calalute(labels, losses)


def pred_labels(losses, threhold):

    pred = losses

    pred[np.where(losses <= threhold)] = 0
    pred[np.where(losses > threhold)] = 1

    return pred


def metrics_calalute(labels, losses):

    fpr, tpr, _ = roc_curve(labels, losses)

    img_auc = auc(fpr, tpr)

    fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]

    fig = plt.figure()

    plt.plot(fpr, tpr, color='darkblue', label='ROC curve (area = {:.2f})'.format(
        img_auc), lw=3)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.legend()
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    # fpr95的指标
    fpr95 = tpr[np.where(fpr >= 0.95)[0][0]]

    png_root_path = os.path.join(os.getcwd(), 'RecurImg', "ocgan")

    if not os.path.exists(png_root_path):
        os.mkdir(png_root_path)

    plt.savefig(os.path.join(png_root_path, "{}_res.png".format("ocgan")))

    figure = plt.figure()

    idx_0 = 0

    idx_1 = 0

    for loss, label in zip(losses, labels):
        if label == 0:
            plt.scatter(idx_0, loss, c='b', linewidths=0.1)
            idx_0 = idx_0 + 1
        else:
            plt.scatter(idx_1, loss, c='r', linewidths=0.1)
            idx_1 = idx_1 + 1

    plt.savefig('loss.png')

    pred = pred_labels(losses, 50)

    p = precision_score(labels, pred, pos_label=0)

    r = recall_score(labels, pred, pos_label=0)

    f1 = f1_score(labels, pred, pos_label=0)

    print("fpr95:{},img_auc:{:.2f},precision:{:.2f},recall:{:.2f},f1:{:.2f}".format(
        fpr95, img_auc, p, r, f1))


if __name__ == "__main__":

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
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for data loader')
    parser.add_argument('--rec_bound', type=float, default=0.1,
                        help='Upper bound of reconstruction loss')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--nw', type=int, default=1,
                        help='num_workers for DataLoader')
    parser.add_argument('--sstep', type=int, default=5,
                        help='Step in epochs for saving models')
    parser.add_argument('--std', type=float, default=0.155,
                        help='Standart deviation for noise in R_Net')
    parser.add_argument('--lambd', type=float, default=0.8,
                        help='Lambda parameter for LR loss')
    parser.add_argument('--cat', action='store_true',
                        help='Turns on skip connections with concatenation')
    parser.add_argument('--res', action='store_true',
                        help='Turns on residual connections')

    parser.add_argument('--data_path', type=str, default="/home/juxiaobing/code/GraduactionCode/data/T15/train",
                        help='Train Dataset\'s path')

    parser.add_argument('--test_path', type=str, default="/home/juxiaobing/code/GraduactionCode/data/T15/test",
                        help='Test Dataset\'s path')

    parser.add_argument('--odatapath', type=str, default="/home/juxiaobing/code/GraduactionCode/data/T15/T15_images",
                        help='Test Dataset\'s path')

    args = parser.parse_args()

    main(args)
