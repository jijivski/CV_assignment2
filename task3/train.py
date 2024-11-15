from torch.utils.data import Dataset, DataLoader
import os
import os.path as osp
import tqdm
import numpy as np
import torch
import time
# from dataset import KITTIDataset, PatchProvider
from dataset import KITTIDataset, PatchDataset
import wandb
from siamese_neural_network import StereoMatchingNetwork, calculate_similarity_score
import argparse


def hinge_loss(score_pos, score_neg, label, margin=0.2):
    """
    Computes the hinge loss for the similarity of a positive and a negative example.

    Args:
        score_pos (torch.Tensor): similarity score of the positive example
        score_neg (torch.Tensor): similarity score of the negative example
        label (torch.Tensor): the true labels

    Returns:
        avg_loss (torch.Tensor): the mean loss over the patch and the mini batch
        acc (torch.Tensor): the accuracy of the prediction
    """

    '''
    The network is trained by minimizing a hinge loss. The loss is computed by considering
pairs of examples centered around the same image position where one example belongs to
the positive and one to the negative class. Let s+ be the output of the network for the
positive example, s− be the output of the network for the negative example, and let m, the
margin, be a positive real number. The hinge loss for that pair of examples is defined as
max(0, m + s− − s+). The loss is zero when the similarity of the positive example is greater
than the similarity of the negative example by at least the margin m. We set the margin
to 0.2 in our experiments.'''

    loss = torch.clamp(margin + score_neg - score_pos, min=0)
    # breakpoint()
    print(loss[:10])
    avg_loss = loss.mean()
    # acc = (loss <= 0).float().mean() #TODO
    acc = (score_pos > score_neg).float().sum() / len(score_pos)
    return avg_loss, acc

    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 6)
    # -------------------------------------


def training_loop(
    infer_similarity_metric,
    patches,
    optimizer,
    out_dir,
    iterations=1000,
    # batch_size=128,
):
    """
    Runs the training loop of the siamese network.

    Args:
        infer_similarity_metric (obj): pytorch module
        patches (obj): patch provider object
        optimizer (obj): optimizer object
        out_dir (str): output file directory
        iterations (int): number of iterations to perform
        batch_size (int): batch size
    """
    #######################################
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 6)
    # -------------------------------------
    time_str = time.strftime("%Y%m%d-%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # infer_similarity_metric.to(device)
    print(f"Device: {device}")
    # breakpoint()

    accumulation_steps = 10
    cnt_batch = 0
    for i in tqdm.tqdm(range(iterations)):
        cumulative_loss = 0.0
        cumulative_acc = 0.0
        # left_patches, right_patches, labels = patches.iterate_batches(batch_size)
        # ref_batch, pos_batch, neg_batch = next(patches.iterate_batches(batch_size))
        for batch_iter,(ref_batch, pos_batch, neg_batch) in enumerate(patches):
            # breakpoint()
            # device = infer_similarity_metric.device
            ref_batch = ref_batch.to(device)
            pos_batch = pos_batch.to(device)
            neg_batch = neg_batch.to(device)
            # breakpoint()
            # pos_batch.shape
            # torch.Size([100, 9, 9, 1])
            # want it to be BCHW, it is now BHWC
            ref_batch = ref_batch.permute(0, 3, 1, 2)
            pos_batch = pos_batch.permute(0, 3, 1, 2)
            neg_batch = neg_batch.permute(0, 3, 1, 2)
            # check the shape, check the names, if reversed


            optimizer.zero_grad()
            similarity_pos = calculate_similarity_score(infer_similarity_metric, ref_batch, pos_batch)
            similarity_neg = calculate_similarity_score(infer_similarity_metric, ref_batch, neg_batch)
            # print('similarity_pos',similarity_pos[:10])
            # print('similarity_neg',similarity_neg[:10])

            loss, acc = hinge_loss(similarity_pos, similarity_neg, label=None)# fill 
            
            # print(loss, acc)
            cumulative_loss += loss.item()
            cumulative_acc += acc.item()
            cnt_batch+=1

            loss.backward()
            optimizer.step()

            if batch_iter % accumulation_steps == 0:
                avg_loss = cumulative_loss / cnt_batch
                avg_acc = cumulative_acc / cnt_batch
                # print(f'batch_iter {batch_iter + 1}, Average Loss: {avg_loss}, Average Accuracy: {avg_acc}')
                wandb.log({'avg_loss':avg_loss,'avg_acc':avg_acc})
                cnt_batch=0
                # Reset accumulators
                cumulative_loss = 0.0
                cumulative_acc = 0.0
        if i%50==0 and i>1:
            with open(osp.join(out_dir, f"{time_str}_train_losses.txt"), "a") as f:
                f.write(f"{i} {loss.item()} {acc.item()}\n")
            # save model here
            torch.save(infer_similarity_metric.state_dict(), osp.join(out_dir, f"trained_model_{i}.pth"))




def main(args):
    wandb.init(project = 'cv2_3', config=args)

    # Fix random seed for reproducibility
    np.random.seed(7)
    torch.manual_seed(7)

    # Hyperparameters
    training_iterations = 1000
    batch_size = 128
    # learning_rate = 3e-4
    # learning_rate = 6e-4
    patch_size = 9
    padding = patch_size // 2
    max_disparity = 50

    # Shortcuts for directories
    root_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(root_dir, "KITTI_2015_subset")
    out_dir = osp.join(root_dir, "output/siamese_network_lr_0001")
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # Create dataloader for KITTI training set
    dataset = KITTIDataset(
        osp.join(data_dir, "training"),
        osp.join(data_dir, "training/disp_noc_0"),
    )
    # Load patch provider
    # patches = PatchProvider(dataset, patch_size=(patch_size, patch_size))
    # patches = PatchProvider(dataset, patch_size=(patch_size, patch_size))
    patch_dataset = PatchDataset(dataset)

    # 创建数据加载器
    train_loader = DataLoader(
        patch_dataset, 
        batch_size=batch_size,      # 每个批次32个样本
        shuffle=True,       # 随机打乱数据
        num_workers=3,     # 12个子进程加载数据# for debugging
        pin_memory=True     # 加速GPU训练
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")  
    # Initialize the network
    infer_similarity_metric = StereoMatchingNetwork()
    # Set to train
    infer_similarity_metric.train()
    infer_similarity_metric.to(device)
    # print(next(infer_similarity_metric.parameters()).device)
    # for param in infer_similarity_metric.parameters():print(param.device)

    # uncomment if you don't have a gpu
    # infer_similarity_metric.to('cpu')

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            infer_similarity_metric.parameters(), lr=args.lr, momentum=0.9
        )
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            infer_similarity_metric.parameters(), lr=args.lr
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            infer_similarity_metric.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    # Start training loop
    training_loop(
        infer_similarity_metric,
        train_loader,
        optimizer,
        out_dir,
        iterations=training_iterations,
        # batch_size=batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose an optimizer")
    parser.add_argument('--optimizer', type=str, default='SGD', 
                        choices=['SGD', 'Adam', 'AdamW'], 
                        help='Optimizer to use: SGD, Adam, or AdamW')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW')
    args = parser.parse_args()
    main(args)

'''
cd /root/autodl-tmp/MKSC-20-0237-codes-data/data/amazon/CV_assignment2/task3/
CUDA_VISIBLE_DEVICES="" python train.py --optimizer SGD
python train.py --optimizer Adam --lr 0.001
python train.py --optimizer AdamW --lr 0.001 --weight_decay 0.01


python train.py 
'''