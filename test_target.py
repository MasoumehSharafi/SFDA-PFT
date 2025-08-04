import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import TargetDataset
from models import ResBase, feat_bottleneck, feat_classifier, FeatureTranslator, FullTranslator
from utils import compute_accuracy
from sklearn.metrics import f1_score
import argparse

#################### MLP-based model ######################
class T_full(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ResBase("resnet18")
        self.translator = FeatureTranslator(feature_dim=512)

    def forward(self, x):
        feat, _ = self.feature_extractor(x)
        return self.translator(feat)

@torch.no_grad()
def evaluate_subject(subject_path, ckpt_dir, pretrained_B, pretrained_C):
    dataset = TargetDataset(subject_path)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    B = feat_bottleneck(feature_dim=512, bottleneck_dim=256).cuda()
    C = feat_classifier(class_num=7, bottleneck_dim=256, type='wn').cuda()
    T = T_full().cuda()

    B.load_state_dict(torch.load(pretrained_B))
    C.load_state_dict(torch.load(pretrained_C))
    B.eval(); C.eval()

    best_T_path = os.path.join(ckpt_dir, 'best_acc', 'T_best_acc.pt')
    if not os.path.exists(best_T_path):
        raise FileNotFoundError(f"Best T checkpoint not found at {best_T_path}")
    T.load_state_dict(torch.load(best_T_path))
    T.eval()

    all_preds = []
    all_labels = []
    total_acc = 0

    for img, label, _ in loader:
        img, label = img.cuda(), label.cuda()
        feat_trans = T(img)
        feat_trans = B(feat_trans)
        logits = C(feat_trans)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(label.cpu().tolist())
        total_acc += compute_accuracy(logits, label)

    avg_acc = total_acc / len(loader)
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    avg_f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_acc, micro_f1, avg_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_root', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--pretrained_B', type=str, required=True)
    parser.add_argument('--pretrained_C', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--subject_name', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    acc_file = os.path.join(args.output_dir, "Accuracy.txt")
    microf1_file = os.path.join(args.output_dir, "MicroF1.txt")
    avgf1_file = os.path.join(args.output_dir, "AvgF1.txt")

    # Write headers only once
    if not os.path.exists(acc_file):
        with open(acc_file, 'w') as f: f.write("Subject,acc\n")
        with open(microf1_file, 'w') as f: f.write("Subject,MicroF1\n")
        with open(avgf1_file, 'w') as f: f.write("Subject,AvgF1\n")

    acc, micro_f1, avg_f1 = evaluate_subject(args.test_root, args.ckpt_dir, args.pretrained_B, args.pretrained_C)

    # Append per-subject result
    with open(acc_file, 'a') as f: f.write(f"{args.subject_name},{acc:.4f}\n")
    with open(microf1_file, 'a') as f: f.write(f"{args.subject_name},{micro_f1:.4f}\n")
    with open(avgf1_file, 'a') as f: f.write(f"{args.subject_name},{avg_f1:.4f}\n")

    print(f"[{args.subject_name}] Acc={acc:.4f}, MicroF1={micro_f1:.4f}, AvgF1={avg_f1:.4f}")
