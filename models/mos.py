import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import MOSNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, target2onehot
from torch.distributions.multivariate_normal import MultivariateNormal
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix
import os
from collections import defaultdict
from utils.losses import FocalLoss, CELoss, LDAMLoss, BalancedSoftmaxLoss

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        self._network = MOSNet(args, True)
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.ensemble = args["ensemble"]

        for n, p in self._network.backbone.named_parameters():
            if 'adapter' not in n and 'head' not in n:
                p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))
    
    def replace_fc(self):       
        model = self._network.to(self._device)
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader_for_protonet):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.forward_orig(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task
        
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.data_manager = data_manager
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        self.replace_fc()
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)

        optimizer = self.get_optimizer(self._network.backbone)
        scheduler = self.get_scheduler(optimizer)
        
        self._init_train(train_loader, test_loader, optimizer, scheduler)
        self._network.backbone.adapter_update()

        self._compute_mean(self._network.backbone)
        if self._cur_task > 0:
            self.classifer_align(self._network.backbone)

    def get_optimizer(self, model):
        base_params = [p for name, p in model.named_parameters() if 'adapter' in name and p.requires_grad]
        base_fc_params = [p for name, p in model.named_parameters() if 'adapter' not in name and p.requires_grad]
        base_params = {'params': base_params, 'lr': self.init_lr, 'weight_decay': self.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': self.init_lr *0.1, 'weight_decay': self.weight_decay}
        network_params = [base_params, base_fc_params]
        
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                network_params, 
                momentum=0.9,
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                network_params,
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                network_params,
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train_old(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        loss_type = self.args.get("loss_type", "ce")
        logging.info(f"Using loss_type: {loss_type}")
        
        # Compute class counts from training labels
        
        cur_classes = np.arange(self._known_classes, self._total_classes)
        labels = np.array(self.train_dataset.labels)
        class_counts = np.bincount(labels, minlength=self._total_classes)
        class_counts = torch.tensor(class_counts, dtype=torch.float32, device=self._device)
        
        class_counts[:self._known_classes] = 0 # to ignore known classes in the loss computation

        # Compute CB-loss weights
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights[class_counts == 0] = 0  # zero weight for old classes
        weights = weights / weights.sum() * len(cur_classes)
        
        if loss_type == "ce":
            criterion = nn.CrossEntropyLoss()

        elif loss_type == "cb":
            criterion = CELoss(weight=weights)
        elif loss_type == "balanced_softmax":
            criterion = BalancedSoftmaxLoss(class_counts=torch.tensor(safe_class_counts, device=self._device))

        elif loss_type == "focal":
            gamma = self.args.get("focal_gamma", 1.5)
            criterion = FocalLoss(weight=weights, gamma=gamma)

        elif loss_type == "ldam":
            max_m = self.args.get("ldam_max_m", 0.5)
            s = self.args.get("ldam_s", 30)
            safe_class_counts = class_counts.clone()
            safe_class_counts[safe_class_counts == 0] = 1.0
            criterion = LDAMLoss(cls_num_list=safe_class_counts.tolist(), max_m=max_m, weight=weights, s=s)

            #criterion = LDAMLoss(cls_num_list=class_counts.tolist(), max_m=max_m, weight=weights, s=s)

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                output = self._network(inputs, adapter_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                #loss = F.cross_entropy(logits, targets.long())
                loss = criterion(logits, targets).mean()
                loss += self.orth_loss(output['pre_logits'], targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # # using EMA method to merge adapters
                if self.args["adapter_momentum"] > 0:
                    self._network.backbone.adapter_merge()
                
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
        
        
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))

        # === Compute class counts for this task ===
        labels = np.array(self.train_dataset.labels)
        class_counts = np.bincount(labels, minlength=self._total_classes)

        def is_long_tail(class_counts, threshold=15):
            counts = class_counts[class_counts > 0]
            if len(counts) == 0:
                return False
            return counts.max() / (counts.min() + 1e-5) > threshold

        has_imbalance = is_long_tail(class_counts)
        imb_ratio = class_counts.max() / (class_counts[class_counts > 0].min() + 1e-5)
        print(f"Imbalance ratio for Task {self._cur_task}: {imb_ratio:.2f}")

        total_epochs = self.args['tuned_epoch']

        # === Precompute CB weights if needed ===
        if has_imbalance:
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, class_counts)
            weights = (1.0 - beta) / (effective_num + 1e-8)
            weights[class_counts == 0] = 0
            active = class_counts > 0
            weights = weights / weights[active].sum() * active.sum()
            weights = torch.tensor(weights, dtype=torch.float32, device=self._device)
        else:
            weights = None

        safe_class_counts = class_counts.copy()
        safe_class_counts[safe_class_counts == 0] = 1

        for epoch in prog_bar:
            self._network.backbone.train()

            # === DRW: choose loss dynamically ===
            if not has_imbalance or epoch < total_epochs // 2:
                criterion = nn.CrossEntropyLoss()
                if epoch == 0:
                    print("DRW: Using CE during warm-up phase")
            else:
                loss_type = self.args.get("loss_type", "cb")  # fallback
                print(f"DRW: Using imbalance-aware loss: {loss_type}")

                if loss_type == "cb":
                    criterion = CELoss(weight=weights)
                elif loss_type == "focal":
                    gamma = self.args.get("focal_gamma", 1.5)
                    criterion = FocalLoss(weight=weights, gamma=gamma)
                elif loss_type == "ldam":
                    max_m = self.args.get("ldam_max_m", 0.5)
                    s = self.args.get("ldam_s", 30)
                    criterion = LDAMLoss(cls_num_list=safe_class_counts.tolist(), max_m=max_m, weight=weights, s=s)
                elif loss_type == "balanced_softmax":
                    criterion = BalancedSoftmaxLoss(class_counts=torch.tensor(safe_class_counts, device=self._device))
                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}")

            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                output = self._network(inputs, adapter_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                loss = criterion(logits, targets).mean()
                loss += self.orth_loss(output['pre_logits'], targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.args.get("adapter_momentum", 0) > 0:
                    self._network.backbone.adapter_merge()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = (
                f"Task {self._cur_task}, Epoch {epoch+1}/{total_epochs} "
                f"=> Loss {losses/len(train_loader):.3f}, Train_acc {train_acc:.2f}, "
                f"Imb_ratio: {imb_ratio:.2f}"
            )
            prog_bar.set_description(info)
        logging.info(info)
        
    @torch.no_grad()
    def _compute_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size*3, shuffle=False, num_workers=4
            )
            
            vectors = []
            for _, _inputs, _targets in idx_loader:
                _vectors = model(_inputs.to(self._device), adapter_id=self._cur_task, train=True)["features"]
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)

            if self.args["ca_storage_efficient_method"] == 'covariance':
                features_per_cls = vectors
                # print(features_per_cls.shape)
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)
            elif self.args["ca_storage_efficient_method"] == 'variance':
                features_per_cls = vectors
                # print(features_per_cls.shape)
                self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
                self.cls_cov[class_idx] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device))
            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                from sklearn.cluster import KMeans
                n_clusters = self.args["n_centroids"] # 10
                features_per_cls = vectors.cpu().numpy()
                kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
                kmeans.fit(features_per_cls)
                cluster_lables = kmeans.labels_
                cluster_means = []
                cluster_vars = []
                for i in range(n_clusters):
                    cluster_data = features_per_cls[cluster_lables == i]
                    cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                    cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(self._device)
                    cluster_means.append(cluster_mean)
                    cluster_vars.append(cluster_var)
                
                self.cls_mean[class_idx] = cluster_means
                self.cls_cov[class_idx] = cluster_vars

    @staticmethod
    def ensure_positive_definite(cov, epsilon=1e-4, max_attempts=5):
        identity = torch.eye(cov.size(0), device=cov.device)
        for _ in range(max_attempts):
            try:
                torch.linalg.cholesky(cov + epsilon * identity)
                return cov + epsilon * identity
            except RuntimeError:
                epsilon *= 10

        diag = torch.clamp(torch.diag(cov), min=1e-3)   # fallback: force diagonal
        return torch.diag(diag)

    def classifer_align(self, model):
        model.train()
        
        run_epochs = self.crct_epochs
        param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'adapter' not in n]
        network_params = [{'params': param_list, 'lr': self.ca_lr, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        prog_bar = tqdm(range(run_epochs))
        for epoch in prog_bar:

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ['covariance', 'variance']:
                for class_idx in range(self._total_classes):
                    mean = self.cls_mean[class_idx].to(self._device)
                    cov = self.cls_cov[class_idx].to(self._device)
                    if self.args["ca_storage_efficient_method"] == 'variance':
                        cov = torch.diag(cov)
                    cov = self.ensure_positive_definite(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([class_idx] * num_sampled_pcls)

            elif self.args["ca_storage_efficient_method"] == 'multi-centroid':
                for class_idx in range(self._total_classes):
                    for cluster in range(len(self.cls_mean[class_idx])):
                        mean = self.cls_mean[class_idx][cluster]
                        var = self.cls_cov[class_idx][cluster]
                        if var.mean() == 0:
                            continue
                        m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([class_idx] * num_sampled_pcls)
            else:
                raise NotImplementedError


            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            if epoch == 0:
                print("sampled data shape: ", sampled_data.shape)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            losses = 0.0
            correct, total = 0, 0
            for _iter in range(self._total_classes):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                outputs = model(inp, fc_only=True)
                logits = outputs['logits'][:, :self._total_classes]

                loss = F.cross_entropy(logits, tgt)
                
                _, preds = torch.max(logits, dim=1)
                
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                losses / self._total_classes,
                ca_acc,
            )
            prog_bar.set_description(info)
         
        logging.info(info)

    def orth_loss(self, features, targets):
        if self.cls_mean:
            # orth loss of this batch
            sample_mean = []
            for k, v in self.cls_mean.items():
                if isinstance(v, list):
                    sample_mean.extend(v)
                else:
                    sample_mean.append(v)
            sample_mean = torch.stack(sample_mean, dim=0).to(self._device, non_blocking=True)
            M = torch.cat([sample_mean, features], dim=0)
            sim = torch.matmul(M, M.t()) / 0.8
            loss = torch.nn.functional.cross_entropy(sim, torch.arange(0, sim.shape[0]).long().to(self._device))
            # print(loss)
            return self.args["reg"] * loss
            # return 0.1 * loss
        else:
            sim = torch.matmul(features, features.t()) / 0.8
            loss = torch.nn.functional.cross_entropy(sim, torch.arange(0, sim.shape[0]).long().to(self._device))
            return self.args["reg"] * loss
            # return 0.0
    
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        orig_y_pred = []
        MAX_ITER = 4
        class_stats = defaultdict(lambda: {
            "count": 0,
            "correct": 0,
            "wrong_adapter_ids": Counter(),
            "correct_iter_hist": Counter(),
            "wrong_iter_hist": Counter(),
        })

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                orig_logits = self._network.forward_orig(inputs)["logits"][:, :self._total_classes]
                orig_preds = torch.max(orig_logits, dim=1)[1].cpu().numpy()
                orig_idx = torch.tensor([self.cls2task[v] for v in orig_preds], device=self._device)
                
                # test the accuracy of the original model
                orig_y_pred.append(orig_preds)
                
                all_features = torch.zeros(len(inputs), self._cur_task + 1, self._network.backbone.out_dim, device=self._device)
                for t_id in range(self._cur_task + 1):
                    t_features = self._network.backbone(inputs, adapter_id=t_id, train=False)["features"]
                    all_features[:, t_id, :] = t_features
                
                # self-refined
                final_logits = []
                
                for x_id in range(len(inputs)):
                    loop_num = 0
                    prev_adapter_idx = orig_idx[x_id]
                    while True:
                        loop_num += 1
                        cur_feature = all_features[x_id, prev_adapter_idx].unsqueeze(0) # shape=[1, 768]
                        cur_logits = self._network.backbone(cur_feature, fc_only=True)["logits"][:, :self._total_classes]
                        cur_pred = torch.max(cur_logits, dim=1)[1].cpu().numpy()
                        cur_adapter_idx = torch.tensor([self.cls2task[v] for v in cur_pred], device=self._device)[0]
                        
                        if loop_num >= MAX_ITER or cur_adapter_idx == prev_adapter_idx:
                            break
                        else:
                            prev_adapter_idx = cur_adapter_idx
                        
                    final_logits.append(cur_logits)

                    # logging refinement statistics
                    label = targets[x_id].item()
                    pred = cur_pred[0]

                    stat = class_stats[label]
                    stat["count"] += 1
                    if pred == label:
                        stat["correct"] += 1
                        stat["correct_iter_hist"][loop_num] += 1
                    else:
                        stat["wrong_iter_hist"][loop_num] += 1
                        stat["wrong_adapter_ids"][prev_adapter_idx.item()] += 1

                final_logits = torch.cat(final_logits, dim=0).to(self._device)

                if self.ensemble:
                    final_logits = F.softmax(final_logits, dim=1)
                    orig_logits = F.softmax(orig_logits / (1/(self._cur_task+1)), dim=1)
                    outputs = final_logits + orig_logits
                else:
                    outputs = final_logits
                
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        orig_acc = (np.concatenate(orig_y_pred) == np.concatenate(y_true)).sum() * 100 / len(np.concatenate(y_true))
        logging.info("the accuracy of the original model:{}".format(np.around(orig_acc, 2)))

        # logging class-wise refinement statistics
        print(f"{'Class':<6} {'Count':<6} {'Acc':<7} {'CorrectIterHist':<35} {'WrongIterHist':<35} {'WrongAdapters':<20}")
        print("-" * 160)
        logging.info(f"{'Class':<6} {'Count':<6} {'Acc':<7} {'CorrectIterHist':<35} {'WrongIterHist':<35} {'WrongAdapters':<20}")
        logging.info("-" * 160)
        for cls in sorted(class_stats.keys()):
            entry = class_stats[cls]
            acc = 100 * entry["correct"] / entry["count"] if entry["count"] > 0 else 0
            correct_hist = dict(entry["correct_iter_hist"])
            wrong_hist = dict(entry["wrong_iter_hist"])
            wrong_adapter_summary = dict(entry["wrong_adapter_ids"])

            print(f"{cls:<6} {entry['count']:<6} {acc:<7.2f} {str(correct_hist):<35} {str(wrong_hist):<35} {str(wrong_adapter_summary):<20}")
            logging.info(f"{cls:<6} {entry['count']:<6} {acc:<7.2f} {str(correct_hist):<35} {str(wrong_hist):<35} {str(wrong_adapter_summary):<20}")

        y_pred_flat = np.concatenate(y_pred)[:, 0]  # Take top-1 prediction
        y_true_flat = np.concatenate(y_true)
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=np.arange(self._total_classes))

        init_cls = 0 if self.args ["init_cls"] == self.args["increment"] else self.args["init_cls"]
        class_order_mode = self.args.get("class_order_mode", "random")
        logs_dir = os.path.join(
            "logs",
            self.args["model_name"],
            self.args["dataset"],
            str(init_cls),
            str(self.args["increment"]),
            class_order_mode,
            "confusions"
        )
        os.makedirs(logs_dir, exist_ok=True)
        cm_save_path = os.path.join(logs_dir, f"cm_task_{self._cur_task}.npy")
        np.save(cm_save_path, cm)

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    
    
    def _eval_cnn_imb(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        orig_y_pred = []
        MAX_ITER = 4
        class_stats = defaultdict(lambda: {
            "count": 0,
            "correct": 0,
            "wrong_max_iter": 0,
            "loop_list": []
        })

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                orig_logits = self._network.forward_orig(inputs)["logits"][:, :self._total_classes]
                orig_preds = torch.max(orig_logits, dim=1)[1].cpu().numpy()
                orig_idx = torch.tensor([self.cls2task[v] for v in orig_preds], device=self._device)
            
                orig_y_pred.append(orig_preds.reshape(-1))
                
                all_features = torch.zeros(len(inputs), self._cur_task + 1, self._network.backbone.out_dim, device=self._device)
                for t_id in range(self._cur_task + 1):
                    t_features = self._network.backbone(inputs, adapter_id=t_id, train=False)["features"]
                    all_features[:, t_id, :] = t_features
                
                final_logits = []
                
                for x_id in range(len(inputs)):
                    loop_num = 0
                    prev_adapter_idx = orig_idx[x_id]
                    while True:
                        loop_num += 1
                        cur_feature = all_features[x_id, prev_adapter_idx].unsqueeze(0)
                        cur_logits = self._network.backbone(cur_feature, fc_only=True)["logits"][:, :self._total_classes]
                        cur_pred = torch.max(cur_logits, dim=1)[1].cpu().numpy()
                        cur_adapter_idx = torch.tensor([self.cls2task[v] for v in cur_pred], device=self._device)[0]
                        
                        if loop_num >= MAX_ITER or cur_adapter_idx == prev_adapter_idx:
                            break
                        else:
                            prev_adapter_idx = cur_adapter_idx
                        
                    final_logits.append(cur_logits)

                    label = targets[x_id].item()
                    pred = cur_pred[0]

                    stat = class_stats[label]
                    stat["count"] += 1
                    stat["loop_list"].append(loop_num)
                    if pred == label:
                        stat["correct"] += 1
                    else:
                        if loop_num >= MAX_ITER:
                            stat["wrong_max_iter"] += 1

                final_logits = torch.cat(final_logits, dim=0).to(self._device)

                if self.ensemble:
                    final_logits = F.softmax(final_logits, dim=1)
                    orig_logits = F.softmax(orig_logits / (1/(self._cur_task+1)), dim=1)
                    outputs = final_logits + orig_logits
                else:
                    outputs = final_logits
                
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
        # === Original acc (overall) ===
        orig_acc = (np.concatenate(orig_y_pred) == np.concatenate(y_true)).sum() * 100 / len(np.concatenate(y_true))
        logging.info("the accuracy of the original model:{}".format(np.around(orig_acc, 2)))

        print(f"{'Class':<6} {'Count':<6} {'Acc':<7} {'AvgIter':<8} {'MaxIterWrong':<14}")
        print("-" * 60)
        logging.info(f"{'Class':<6} {'Count':<6} {'Acc':<7} {'AvgIter':<8} {'MaxIterWrong':<14}")
        logging.info("-" * 60)
        for cls in sorted(class_stats.keys()):
            entry = class_stats[cls]
            acc = 100 * entry["correct"] / entry["count"] if entry["count"] > 0 else 0
            avg_iter = np.mean(entry["loop_list"])
            print(f"{cls:<6} {entry['count']:<6} {acc:<7.2f} {avg_iter:<8.2f} {entry['wrong_max_iter']:<14}")
            logging.info(f"{cls:<6} {entry['count']:<6} {acc:<7.2f} {avg_iter:<8.2f} {entry['wrong_max_iter']:<14}")
        
        # === Head/Mid/Tail acc ===
        labels = np.array(self.train_dataset.labels)
        class_counts = np.bincount(labels, minlength=self._total_classes)

        counts = class_counts.copy()

        head_cls = np.where(counts > 100)[0]
        mid_cls  = np.where((counts >= 20) & (counts <= 100))[0]
        tail_cls = np.where(counts < 20)[0]

        print(f"Head classes: {head_cls.tolist()}")
        print(f"Mid classes: {mid_cls.tolist()}")›
        print(f"Tail classes: {tail_cls.tolist()}")
        
        y_pred_ = np.concatenate(y_pred)[:, 0]  # top-1
        y_true_ = np.concatenate(y_true)

        head_mask = np.isin(y_true_, head_cls)
        mid_mask = np.isin(y_true_, mid_cls)
        tail_mask = np.isin(y_true_, tail_cls)

        head_acc = (y_pred_[head_mask] == y_true_[head_mask]).mean() * 100 if np.sum(head_mask) > 0 else 0
        mid_acc = (y_pred_[mid_mask] == y_true_[mid_mask]).mean() * 100 if np.sum(mid_mask) > 0 else 0
        tail_acc = (y_pred_[tail_mask] == y_true_[tail_mask]).mean() * 100 if np.sum(tail_mask) > 0 else 0

        print(f"Head Acc: {head_acc:.2f}, Mid Acc: {mid_acc:.2f}, Tail Acc: {tail_acc:.2f}")
        logging.info(f"Head Acc: {head_acc:.2f}, Mid Acc: {mid_acc:.2f}, Tail Acc: {tail_acc:.2f}")    

        return np.concatenate(y_pred), np.concatenate(y_true), head_acc, mid_acc, tail_acc
    
    
    def is_long_tail(class_counts, threshold=1.5):
        counts = class_counts[class_counts > 0]
        if len(counts) == 0:
            return False
        return counts.max() / (counts.min() + 1e-5) > threshold

