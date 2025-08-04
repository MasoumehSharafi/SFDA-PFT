import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from data_loader import PairDataset
from models_fine import ResBase, feat_bottleneck, feat_classifier, FeatureTranslator
from losses import get_emotion_loss_fn
from utils import compute_accuracy
from sklearn.metrics import f1_score
import argparse

class T_full(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ResBase("resnet18")
        self.translator = FeatureTranslator(feature_dim=512)

    def forward(self, x):
        feat, _ = self.feature_extractor(x)
        return self.translator(feat)

def train(folder1, folder2, epochs=10, batch_size=16, lr=1e-4,
          ckpt_dir='./checkpoints',
          log_dir='./runs',
          pretrained_F=None, pretrained_B=None, pretrained_C=None, pretrained_T=None,
          early_stop_patience=10,
          emotion_loss_type='mse'):

    dataset = PairDataset(folder1, folder2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    F = ResBase("resnet18").cuda()
    B = feat_bottleneck(feature_dim=512, bottleneck_dim=256).cuda()
    C = feat_classifier(class_num=2, bottleneck_dim=256, type='wn').cuda()

    F.load_state_dict(torch.load(pretrained_F))
    B.load_state_dict(torch.load(pretrained_B))
    C.load_state_dict(torch.load(pretrained_C))

    for param in F.parameters(): param.requires_grad = False
    for param in B.parameters(): param.requires_grad = False
    for param in C.parameters(): param.requires_grad = False
    F.eval(); B.eval(); C.eval()

    T = T_full().cuda()
    if pretrained_T:
        print(f"Loading pretrained T from: {pretrained_T}")
        T.load_state_dict(torch.load(pretrained_T))

    optimizer = optim.Adam(T.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    ce_loss_fn = nn.CrossEntropyLoss()
    emotion_loss_fn = get_emotion_loss_fn(emotion_loss_type)

    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc, early_stop_counter = 0, 0

    for epoch in range(epochs):
        T.train(); total_loss, total_acc = 0, 0

        for img1, img2, labels, _ in train_loader:
            img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()

            with torch.no_grad():
                feat1, _ = F(img1)
                feat1 = B(feat1).view(feat1.size(0), -1)
                feat2_target, _ = F(img2)
                feat2_target = B(feat2_target).view(feat2_target.size(0), -1)
                logits2_target = C(feat2_target)

            feat2_trans = T(img2)
            feat2_trans = B(feat2_trans).view(feat2_trans.size(0), -1)
            logits2_trans = C(feat2_trans)

            ce_loss = ce_loss_fn(logits2_trans, labels)
            eloss = emotion_loss_fn(logits2_target, logits2_trans)
            loss = ce_loss + eloss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += compute_accuracy(logits2_trans, labels)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_train_acc, epoch)

        # Validation
        T.eval(); val_loss, val_acc = 0, 0; all_preds, all_labels = [], []
        with torch.no_grad():
            for img1, img2, labels, _ in val_loader:
                img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()
                feat1, _ = F(img1)
                feat1 = B(feat1).view(feat1.size(0), -1)
                feat2_target, _ = F(img2)
                feat2_target = B(feat2_target).view(feat2_target.size(0), -1)
                logits2_target = C(feat2_target)

                feat2_trans = T(img2)
                feat2_trans = B(feat2_trans).view(feat2_trans.size(0), -1)
                logits2_trans = C(feat2_trans)

                ce_loss = ce_loss_fn(logits2_trans, labels)
                eloss = emotion_loss_fn(logits2_target, logits2_trans)
                loss = ce_loss + eloss

                val_loss += loss.item()
                val_acc += compute_accuracy(logits2_trans, labels)
                all_preds.extend(torch.argmax(logits2_trans, dim=1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_val_acc = val_acc / len(val_loader)
        wf1 = f1_score(all_labels, all_preds, average='weighted')
        avgf1 = f1_score(all_labels, all_preds, average='macro')

        writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)
        writer.add_scalar('Accuracy/val', avg_val_acc, epoch)
        writer.add_scalar('WF1/val', wf1, epoch)
        writer.add_scalar('AvgF1/val', avgf1, epoch)

        scheduler.step(avg_val_acc)

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            early_stop_counter = 0
            os.makedirs(os.path.join(ckpt_dir, 'best_acc'), exist_ok=True)
            torch.save(T.state_dict(), os.path.join(ckpt_dir, 'best_acc', 'T_best_acc.pt'))
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Acc: {avg_val_acc:.4f}, W-F1: {wf1:.4f}, Avg-F1: {avgf1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder1', type=str, required=True)
    parser.add_argument('--folder2', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pretrained_F', type=str, required=True)
    parser.add_argument('--pretrained_B', type=str, required=True)
    parser.add_argument('--pretrained_C', type=str, required=True)
    parser.add_argument('--pretrained_T', type=str, default='')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./runs')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--emotion_loss', type=str, default='ce', choices=['mse', 'ce', 'kl'])
    args = parser.parse_args()

    train(args.folder1, args.folder2, args.epochs, args.batch_size, args.lr,
          ckpt_dir=args.ckpt_dir, log_dir=args.log_dir,
          pretrained_F=args.pretrained_F, pretrained_B=args.pretrained_B, pretrained_C=args.pretrained_C,
          pretrained_T=args.pretrained_T,
          early_stop_patience=args.patience,
          emotion_loss_type=args.emotion_loss)
