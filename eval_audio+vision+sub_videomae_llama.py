import argparse
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.functional import average_precision

from ext.byol_a.byol_a.common import load_yaml_config
from ext.byol_a.byol_a.models import AudioNTT2020

from funnynet.dataset_hybrid import Audio_vision_dataset_v2
from funnynet.network import projection_net
from transformers import (
    AutoModel,
    VideoMAEModel,
)


def test(f):
    clf_head.eval()
    vision_net.eval()
    audio_net.eval()
    text_net.eval()

    avg_tp = 0
    avg_tn = 0
    avg_fp = 0
    avg_fn = 0
    count_0 = 0
    count_1 = 0
    label_len = len(testing_data_loader.dataset.annotation)
    P = torch.zeros(label_len, 2)
    T = torch.zeros(label_len)
    start = 0
    for iteration, batch in enumerate(testing_data_loader, 1):
        img, audio, input_ids, attention_mask, label = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
            batch[4],
        )

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

        count_1 = count_1 + torch.sum(label)
        count_0 = count_0 + (label.shape[0] - torch.sum(label))

    epsilon = 1e-7

    precision = avg_tp / (avg_tp + avg_fp + epsilon)
    recall = avg_tp / (avg_tp + avg_fn + epsilon)
    accuracy = avg_tp + avg_tn
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    score = T
    T = T.long()
    for p in range(P.shape[0]):
        score[p : p + 1] = P[p : p + 1, T[p]]
    mAp = average_precision(score, T, pos_label=1)

    avg_tp = avg_tp / count_1
    avg_tn = avg_tn / count_0
    accuracy = accuracy / label_len

    print(
        " || ".join(
            [
                f"F1 score: {f1.data:.4f}",
                f"" f"accuracy: {accuracy.data:.4f}"
            ]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("llama2_dir", type=str)
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument(
        "--model_file", type=str, default="output/sub/final_models/audio+vision_v17.pth"
    )
    parser.add_argument("--data_augmentation", type=bool, default=True)
    parser.add_argument("--img_size", type=int, default=224, help="crop image size")
    parser.add_argument("--frame_rate", type=float, default=24, help="crop image size")
    parser.add_argument(
        "--num_frame", type=int, default=8, help="number of input frames"
    )
    parser.add_argument(
        "--time_step", type=int, default=8, help="time slot of input frames"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="number of threads for data loader to use"
    )
    parser.add_argument("--batchSize", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--NUM_GPUS", type=int, default=2, help="number of gpus to use")
    parser.add_argument(
        "--NUM_SHARDS", type=int, default=1, help="number of shards to use"
    )
    parser.add_argument(
        "--save_interval", type=int, default=5000, help="checkpoint save interval"
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

    # ################ DATASET #################
    test_set = Audio_vision_dataset_v2(
        video_dir=osp.join(opt.data_dir, "Test_manual/video/test_final_8s"),
        audio_dir=osp.join(opt.data_dir, "Test_manual/audio_new/test_final_8s"),
        label_path=osp.join(opt.data_dir, "laughter_manual/test_final.pk"),
        sub_path=osp.join(opt.data_dir, "Test_manual/sub/test_final_8s/sub_test_8s.pk"),
        time_step=opt.time_step,
        cfg=cfg,
    )
    testing_data_loader = DataLoader(
        dataset=test_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=False,
        drop_last=False,
    )

    # ################ audio network #################
    audio_net = AudioNTT2020(d=cfg.feature_d)
    audio_net.load_weight("models/AudioNTT2020-BYOLA-64x96d2048.pth", device)
    # ################ vision network #################
    vision_net = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large")
    # ################ language network #################
    text_net = AutoModel.from_pretrained('/mnt/5d6c1bb2-d428-4b49-a52a-3a777f4888e1/llama/llama-2-hf/7B', torch_dtype=torch.bfloat16)
    # ################ projection network #############
    clf_head = projection_net(N=text_net.config.hidden_size, n_embedding_dim=4096)
    # ################ MODEL #################
    for param in vision_net.parameters():
        param.requires_grad = False

    # ################ LOSS & OPTIMIZER #################
    L1_criterion_avg = torch.nn.L1Loss(size_average=True)
    BCE_criterion = torch.nn.BCEWithLogitsLoss()
    audio_net.to(device)
    vision_net.to(device)
    text_net.to(device)
    clf_head.to(device)
    BCE_criterion.to(device)

    with open("output/result.txt", "w") as f:
        if opt.pretrained:
            if os.path.exists(opt.model_file):
                clf_head.load_state_dict(torch.load(opt.model_file))
                print("pretrained model is loaded!")
                test(f)
