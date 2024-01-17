import argparse
import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ext.byol_a.byol_a.common import load_yaml_config
from ext.byol_a.byol_a.models import AudioNTT2020

from funnynet.dataset_hybrid import Audio_vision_dataset_v2_all, Audio_vision_dataset_v2
from funnynet.network import ContrastiveLossELI5, projection_net
from transformers import (
    AutoModel,
    VideoMAEModel,
)


def train(opt, epoch):
    clf_head.train()
    audio_net.eval()
    vision_net.eval()
    text_net.eval()
    label_len = len(training_data_loader.dataset.annotation)
    clip_sim_loss = 0
    koleo_sim_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        img, audio, input_ids, attention_mask, label = batch
        img = img.to(device)
        audio = audio.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device).long()

        # forward
        optimizer.zero_grad()
        with torch.no_grad():
            output = vision_net(img)
            vision_feat = output.last_hidden_state
            audio_feat = audio_net(audio)
            sub_feat = text_net(input_ids, attention_mask=attention_mask)
            sub_feat = sub_feat.last_hidden_state.to(torch.float32)

        feat_1, feat_2, feat_3, prob = clf_head(audio_feat, vision_feat, sub_feat)

        # loss
        bce_loss = BCE_criterion(prob, label)
        simclr_loss = Contrastive_Loss(feat_1, feat_2) + Contrastive_Loss(feat_1, feat_3) + Contrastive_Loss(feat_2, feat_3)
        loss = bce_loss + 0.03 * simclr_loss

        # backward & optimization
        loss.backward()
        optimizer.step()

        print(
            f"===> Epoch[{epoch}]({iteration}/{len(training_data_loader)}): "
            + f"bce_loss: {bce_loss.data:.3f} || simclr_loss: {simclr_loss.data:.3f}"
        )

        writer.add_scalar("bce loss", bce_loss, iteration + epoch * label_len)
        writer.add_scalar("simclr loss", simclr_loss, iteration + epoch * label_len)



def test(epoch):
    clf_head.eval()
    vision_net.eval()
    audio_net.eval()

    avg_tp = 0
    avg_tn = 0
    avg_fp = 0
    avg_fn = 0

    label_len = len(test_data_loader.dataset.annotation)
    num_iters = len(test_data_loader)
    P = torch.zeros(label_len, 2)
    T = torch.zeros(label_len)
    start = 0
    for iteration, batch in tqdm(enumerate(test_data_loader, 1), total=num_iters):
        img, audio, input_ids, attention_mask, label = batch

        img = img.to(device)
        audio = audio.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        b = img.shape[0]

        with torch.no_grad():
            output = vision_net(img)
            vision_feat = output.last_hidden_state
            audio_feat = audio_net(audio)
            sub_feat = text_net(input_ids, attention_mask=attention_mask)
            sub_feat = sub_feat.last_hidden_state.to(torch.float32)
            _, _, _, predict = clf_head(audio_feat, vision_feat, sub_feat)

        _, index = torch.topk(predict, 1)
        index = index.data.cpu().squeeze(1)
        tp = (label * index) * 1.0
        tn = ((1 - label) * (1 - index)) * 1.0
        fp = ((1 - label) * index) * 1.0
        fn = (label * (1 - index)) * 1.0

        avg_tp = avg_tp + torch.sum(tp)
        avg_tn = avg_tn + torch.sum(tn)
        avg_fp = avg_fp + torch.sum(fp)
        avg_fn = avg_fn + torch.sum(fn)
        score = F.softmax(predict, dim=1)

        P[start : start + b, :] = score.cpu()
        T[start : start + b] = label
        start = start + b

    epsilon = 1e-7

    precision = avg_tp / (avg_tp + avg_fp + epsilon)
    recall = avg_tp / (avg_tp + avg_fn + epsilon)
    accuracy = avg_tp + avg_tn
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    accuracy = accuracy / label_len

    writer.add_scalar("F1 score", f1, epoch)
    writer.add_scalar("accuracy", accuracy, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", type=str)
    parser.add_argument("llama2_dir", type=str)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--embed_model", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="output/loss/v1")
    parser.add_argument("--data_augmentation", type=bool, default=True)
    parser.add_argument("--img_size", type=int, default=224, help="crop image size")
    parser.add_argument(
        "--num_frame", type=int, default=16, help="number of input frames"
    )
    parser.add_argument(
        "--time_step", type=int, default=8, help="time slot of input` frames"
    )
    parser.add_argument('--use_clip', action='store_true', help='if use clip loss')
    parser.add_argument('--use_koleo', action='store_true', help='if use koleo loss')
    parser.add_argument(
        "--threads", type=int, default=8, help="number of threads for data loader to use"
    )
    parser.add_argument("--batchSize", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--NUM_GPUS", type=int, default=1, help="number of gpus to use")
    parser.add_argument(
        "--NUM_SHARDS", type=int, default=1, help="number of shards to use"
    )
    parser.add_argument(
        "--save_interval", type=int, default=1, help="checkpoint save interval"
    )
    parser.add_argument(
        "--nEpochs", type=int, default=100, help="number of epochs to train for"
    )
    parser.add_argument("--snapshots", type=int, default=1, help="Snapshots")
    parser.add_argument("--start_iter", type=int, default=1, help="Starting Epoch")

    # ################ PREPARATIONS #################
    opt = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml_config("ext/byol_a/config.yaml")
    writer = SummaryWriter(
        log_dir=opt.log_dir, comment="_scalars", filename_suffix="123"
    )

    # ################ DATASET #################
    print("===> Loading vision_audio datasets")
    # ################ train set #################
    train_set = Audio_vision_dataset_v2_all(
        data_dir=opt.data_dir,
        llama2_dir=opt.llama2_dir,
        time_step=opt.time_step,
        cfg=cfg,
    )
    weights = torch.tensor([1, 1], dtype=torch.float)
    train_targets = train_set.get_classes_for_all_imgs()
    sample_weights = weights[train_targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
    )
    # ################ val set #################
    test_set = Audio_vision_dataset_v2(
        video_dir=osp.join(opt.data_dir, "Validation_manual/video/validation_final_8s"),
        audio_dir=osp.join(opt.data_dir, "Validation_manual/audio_new/validation_final_8s"),
        label_path=osp.join(opt.data_dir, "laughter_manual/validation_final.pk"),
        sub_path=osp.join(opt.data_dir, "Validation_manual/sub/validation_final_8s/sub_validation_8s.pk"),
        time_step=opt.time_step,
        cfg=cfg,
    )
    test_data_loader = DataLoader(
        dataset=test_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=False,
    )

    # ################ audio network #################
    audio_net = AudioNTT2020(d=cfg.feature_d)
    audio_net.load_weight("models/AudioNTT2020-BYOLA-64x96d2048.pth", device)
    # ################ vision network #################
    vision_net = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large")
    # ################ language network #################
    text_net = AutoModel.from_pretrained(opt.llama2_dir, torch_dtype=torch.bfloat16)
    # ################ projection network #############
    clf_head = projection_net(N=text_net.config.hidden_size, n_embedding_dim=4096)
    # ################ MODEL #################
    if opt.pretrained:
        if os.path.exists(opt.embed_model):
            pretrained_dict = torch.load(
                opt.embed_model, map_location=lambda storage, loc: storage
            )
            model_dict = clf_head.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            clf_head.load_state_dict(model_dict)
            print("pretrained feat embedding model is loaded!")
    for param in vision_net.parameters():
        param.requires_grad = False
    for param in text_net.parameters():
        param.requires_grad = False
    for param in audio_net.parameters():
        param.requires_grad = False

    # ################ LOSS & OPTIMIZER #################
    weights = [1, 1]
    class_weights = torch.FloatTensor(weights).cuda()
    Contrastive_Loss = ContrastiveLossELI5(opt.batchSize)
    BCE_criterion = torch.nn.CrossEntropyLoss(size_average=True, weight=class_weights)
    optimizer = torch.optim.Adam(clf_head.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,80,100], gamma=0.5)

    # ################ GPU  #################
    clf_head.to(device)
    vision_net.to(device)
    audio_net.to(device)
    text_net.to(device)
    BCE_criterion.to(device)
    Contrastive_Loss.to(device)

    # ################ TRAINING #################
    for epoch in range(opt.nEpochs):
        train(opt, epoch)
        test(epoch)
        scheduler.step()
        if (epoch + 1) % opt.save_interval == 0:
            torch.save(clf_head.state_dict(), "output/v1_epoch_%d.pth" % (epoch + 1))
