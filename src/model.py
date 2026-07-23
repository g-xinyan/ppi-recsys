"""
双塔推荐模型模块
Two-Tower Model with bias correction, hybrid sampling, and self-supervised learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm.auto import tqdm


class ItemDataset(Dataset):
    """商品数据集"""

    def __init__(self, user_items, item_features, neg_sample_ratio=0.5):
        self.user_items = user_items
        self.item_features = item_features
        self.item_list = list(item_features.keys()) if item_features else []
        self.neg_sample_ratio = neg_sample_ratio
        self.pairs = []

        # 构建正样本对
        for user_id, items in user_items.items():
            for item in items[:50]:
                if item in item_features:
                    self.pairs.append((user_id, item, 1))

        # 添加负样本
        n_neg = int(len(self.pairs) * neg_sample_ratio)
        for _ in range(n_neg):
            user_id = np.random.choice(list(user_items.keys()))
            neg_item = np.random.choice(self.item_list)
            self.pairs.append((user_id, neg_item, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user_id, item_id, label = self.pairs[idx]
        item_feat = self.item_features.get(item_id, np.zeros(32))
        return torch.tensor(item_feat, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class TwoTowerModel(nn.Module):
    """
    双塔模型

    包含用户塔 (Embedding)、物品塔 (MLP)、偏差纠正层和自监督学习头。
    """

    def __init__(self, input_dim=32, emb_dim=64, dropout=0.2):
        super().__init__()

        self.user_embedding = nn.Embedding(200000, emb_dim)

        self.item_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, emb_dim),
        )

        self.bias_layer = nn.Linear(emb_dim * 2, 1)

        self.ssl_head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def encode_user(self, user_id):
        if isinstance(user_id, int):
            user_id = torch.tensor([user_id])
        return self.user_embedding(user_id)

    def encode_item(self, item_features):
        return self.item_encoder(item_features)

    def forward(self, user_id, item_features, return_embeddings=False):
        user_emb = self.encode_user(user_id)
        item_emb = self.encode_item(item_features)

        similarity = (user_emb * item_emb).sum(dim=1, keepdim=True)

        concat_emb = torch.cat([user_emb, item_emb], dim=1)
        bias = self.bias_layer(concat_emb)

        score = similarity + bias

        if return_embeddings:
            return score, user_emb, item_emb
        return score

    def ssl_loss(self, item_emb):
        """自监督学习损失 (对比学习)"""
        noise = torch.randn_like(item_emb) * 0.1
        item_emb_aug = item_emb + noise
        pred = self.ssl_head(item_emb)
        return nn.MSELoss()(pred, item_emb_aug)


class TwoTowerRecommender:
    """
    双塔推荐器

    包含偏差纠正、混合采样和自监督学习的完整推荐流程。
    """

    def __init__(self, input_dim=32, emb_dim=64, lr=0.001, epochs=10, batch_size=256):
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.item_features = {}
        self.user_items = defaultdict(list)
        self.is_trained = False

    def _build_item_features(self, sessions, all_items):
        """构建商品特征 (基于共现矩阵 + SVD 降维)"""
        from sklearn.decomposition import TruncatedSVD

        print("Building item features...")
        item_cooccurrence = defaultdict(Counter)
        for session in tqdm(sessions, desc="Building co-occurrence"):
            aids = [int(ev["aid"]) for ev in session['events']]
            unique_aids = list(set(aids))
            for i, aid1 in enumerate(unique_aids):
                for aid2 in unique_aids[i + 1:]:
                    item_cooccurrence[aid1][aid2] += 1
                    item_cooccurrence[aid2][aid1] += 1

        all_items_list = list(all_items)
        item_to_idx = {item: i for i, item in enumerate(all_items_list)}

        n_items = len(all_items_list)
        cooccurrence_matrix = np.zeros((n_items, min(100, n_items)))
        for i, item in enumerate(tqdm(all_items_list[:10000], desc="Building feature matrix")):
            neighbors = item_cooccurrence[item].most_common(50)
            for j, (neighbor, count) in enumerate(neighbors):
                if neighbor in item_to_idx and j < cooccurrence_matrix.shape[1]:
                    cooccurrence_matrix[i][j] = count

        if cooccurrence_matrix.shape[0] > 10 and cooccurrence_matrix.shape[1] > 10:
            svd = TruncatedSVD(n_components=self.input_dim, random_state=42)
            features = svd.fit_transform(cooccurrence_matrix)
        else:
            features = np.random.randn(len(all_items_list), self.input_dim)

        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        for i, item in enumerate(all_items_list[:len(features)]):
            self.item_features[item] = features[i]
        for item in all_items_list[len(features):]:
            self.item_features[item] = np.random.randn(self.input_dim) * 0.1

        print(f"Built features for {len(self.item_features)} items")
        return self.item_features

    def _build_user_items(self, sessions):
        """构建用户-商品交互矩阵"""
        print("Building user-item matrix...")
        for idx, session in enumerate(tqdm(sessions, desc="Building user-items")):
            events = session['events']
            aids = [int(ev["aid"]) for ev in events]
            for aid in set(aids):
                weight = 1
                for ev in events:
                    if int(ev["aid"]) == aid:
                        from .utils import event_type_to_int
                        t = event_type_to_int(ev["type"])
                        if t == 2:
                            weight = 3
                        elif t == 1:
                            weight = 2
                        break
                self.user_items[idx].extend([aid] * weight)
        print(f"Built interactions for {len(self.user_items)} users")
        return self.user_items

    def _train_model(self):
        """训练双塔模型"""
        print("Training Two-Tower model...")
        dataset = ItemDataset(self.user_items, self.item_features, neg_sample_ratio=0.3)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = TwoTowerModel(input_dim=self.input_dim, emb_dim=self.emb_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        pos_weight = 2.0

        for epoch in range(self.epochs):
            total_loss = 0
            total_ssl_loss = 0
            for item_feats, labels in dataloader:
                optimizer.zero_grad()
                user_ids = torch.randint(0, len(self.user_items), (len(item_feats),))
                scores = self.model(user_ids, item_feats)
                weights = torch.where(labels == 1,
                                      torch.tensor(pos_weight),
                                      torch.tensor(1.0))
                bce_loss = nn.BCEWithLogitsLoss(weight=weights)
                loss = bce_loss(scores.squeeze(), labels)

                _, _, item_emb = self.model(user_ids, item_feats, return_embeddings=True)
                ssl_loss = self.model.ssl_loss(item_emb)
                total_loss_combined = loss + 0.1 * ssl_loss

                total_loss_combined.backward()
                optimizer.step()
                total_loss += loss.item()
                total_ssl_loss += ssl_loss.item()

            print(f"Epoch {epoch + 1}/{self.epochs} - "
                  f"Loss: {total_loss / len(dataloader):.4f}, "
                  f"SSL Loss: {total_ssl_loss / len(dataloader):.4f}")

        self.is_trained = True
        print("Two-Tower model training complete")
