import logging
import numpy as np
import torch
import os
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
from utils.longtail_split_specs import LONGTAIL_SPLIT_SPEC

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
                embedding = model.backbone(data, adapter_id=self._cur_task, train=False)['features']
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

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        cur_task_cls_freq = Counter()

        for _, _, targets in train_loader:
            cur_task_cls_freq.update(targets.tolist())

        self._network.backbone.set_hierarchical_routing(dict(cur_task_cls_freq))

        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()

            for adapter in self._network.backbone.cur_adapter:
                adapter.reset_expert_usage()

            losses = 0.0
            router_losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                output = self._network(inputs, adapter_id=self._cur_task, train=True, targets=targets, return_loss=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                loss = F.cross_entropy(logits, targets.long())
                router_loss = output.get('router_loss', 0) * 0.5
                # Expert orthogonality loss
                orth_loss = 0
                # if i % 10 == 0:
                    # orth_loss = self._network.backbone.compute_expert_orthogonality_loss() * 0.05
                    # orth_loss = self._network.backbone.compute_cross_task_expert_orthogonality_loss() * 0.05

                loss += self.orth_loss(output['pre_logits'], targets) + router_loss + orth_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # using EMA method to merge adapters
                if self.args["adapter_momentum"] > 0:
                    self._network.backbone.adapter_merge()

                losses += loss.item()
                if isinstance(router_loss, torch.Tensor):
                    router_losses += router_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Router KLD Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                router_losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)
            self._log_expert_utilization(epoch)

        self._log_expert_utilization_summary()
        logging.info(info)

    def _log_expert_utilization(self, epoch):
        """Log expert utilization statistics for current epoch"""
        # Aggregate stats across all layers
        total_usage = torch.zeros(self.args['num_experts'])
        num_layers = len(self._network.backbone.cur_adapter)

        for _, adapter in enumerate(self._network.backbone.cur_adapter):
            layer_usage = adapter.get_expert_usage()
            total_usage += layer_usage

        # Average across layers
        avg_usage = total_usage / num_layers

        # Convert to percentages
        usage_percentages = (avg_usage * 100).numpy()

        # Log the statistics
        logging.info(f"Task {self._cur_task} => Epoch {epoch+1} Expert Utilization:")
        logging.info(f"General Expert (0):  {usage_percentages[0]:.2f}%")
        logging.info(f"Majority Expert (1): {usage_percentages[1]:.2f}%")
        logging.info(f"Minority Expert (2): {usage_percentages[2]:.2f}%")

        # Check for imbalance
        max_usage = usage_percentages.max()
        min_usage = usage_percentages.min()
        imbalance_ratio = max_usage / (min_usage + 1e-8)

        if imbalance_ratio > 3.0:  # If one expert is used 3x more than another
            logging.warning(f"High imbalance detected! Ratio: {imbalance_ratio:.2f}")

    def _log_expert_utilization_summary(self):
        """Log final summary of expert utilization across all epochs"""
        logging.info(f"TASK {self._cur_task} EXPERT UTILIZATION SUMMARY")
        logging.info("="*50)

        # Get layer-wise statistics
        for layer_idx in [0, len(self._network.backbone.cur_adapter)//2, -1]:  # First, middle, last layer
            adapter = self._network.backbone.cur_adapter[layer_idx]
            usage = adapter.get_expert_usage()
            usage_pct = (usage * 100).numpy()

            layer_name = "First" if layer_idx == 0 else "Middle" if layer_idx > 0 else "Last"
            logging.info(f"{layer_name} Layer (#{layer_idx})=> General: {usage_pct[0]:.2f}% | Majority: {usage_pct[1]:.2f}% | Minority: {usage_pct[2]:.2f}%")

    def _compute_semantic_clusters(self, train_loader, num_clusters=3):
        self._network.backbone.eval()
        class_features = defaultdict(list)

        with torch.no_grad():
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                features = self._network.backbone(inputs, adapter_id=self._cur_task, train=False)['features']
                for feat, label in zip(features, targets):
                    class_features[label.item()].append(feat)

        # Compute class prototypes with stability checks
        class_prototypes = {}

        for cls, feats in class_features.items():
            class_prototypes[cls] = torch.stack(feats).mean(dim=0)

        # Normalize prototypes
        prototypes_tensor = torch.stack([class_prototypes[cls] for cls in sorted(class_prototypes.keys())])
        prototypes_normalized = F.normalize(prototypes_tensor, dim=1)

        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(num_clusters, len(class_prototypes)), n_init=10)
        cluster_assignments = kmeans.fit_predict(prototypes_normalized.cpu().numpy())

        cls2expert = {}
        for idx, cls in enumerate(sorted(class_prototypes.keys())):
            cls2expert[cls] = cluster_assignments[idx]

        # Log clustering statistics
        expert_counts = Counter(cls2expert.values())
        logging.info(f"Expert distribution: {dict(expert_counts)}")
        logging.info(f"Class-to-expert mapping: {cls2expert}")

        self._network.backbone.train()
        return cls2expert

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

        class_stats = defaultdict(lambda: {"count": 0, "correct": 0})

        # Pre-compute prototype information
        prototype_matrix = []
        class_indices = sorted(self.cls_mean.keys())
        for class_idx in class_indices:
            prototype_matrix.append(self.cls_mean[class_idx])
        prototype_matrix = torch.stack(prototype_matrix).to(self._device)  # [num_classes, feature_dim]

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # Extract features from all adapters for all samples
                all_adapter_features = []
                for task_id in range(self._cur_task + 1):
                    features = self._network.backbone(inputs, adapter_id=task_id, train=False)["features"]
                    all_adapter_features.append(features)  # Each is [B, feature_dim]

                final_logits = []

                # Process each sample
                for sample_idx in range(inputs.size(0)):
                    # Collect features from all adapters for this sample
                    sample_features = [feature[sample_idx] for feature in all_adapter_features]

                    # Find nearest prototype
                    min_dist = float('inf')
                    best_task_id = 0

                    for idx, class_idx in enumerate(class_indices):
                        # Use features from the adapter that matches the prototype's task
                        prototype = prototype_matrix[idx].unsqueeze(0)
                        proto_task = self.cls2task[class_idx]
                        sample_task_features = sample_features[proto_task].unsqueeze(0)

                        # Compute distance
                        sample_task_features = F.normalize(sample_task_features, dim=1)
                        prototype = F.normalize(prototype, dim=1)
                        dist = 1 - F.cosine_similarity(sample_task_features, prototype, dim=1).item()

                        if dist < min_dist:
                            min_dist = dist
                            best_task_id = proto_task

                    # Get final prediction using best adapter
                    best_features = sample_features[best_task_id].unsqueeze(0)
                    sample_logits = self._network.backbone(best_features, fc_only=True)["logits"][:, :self._total_classes]
                    final_logits.append(sample_logits)

                final_logits = torch.cat(final_logits, dim=0)
                outputs = final_logits

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            batch_predictions = predicts[:, 0].cpu().numpy()  # Top-1 predictions
            batch_targets = targets.cpu().numpy()

            for pred, true_label in zip(batch_predictions, batch_targets):
                class_stats[true_label]["count"] += 1
                if pred == true_label:
                    class_stats[true_label]["correct"] += 1

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        y_pred_all = np.concatenate(y_pred)[:, 0]  # Top-1 predictions
        y_true_all = np.concatenate(y_true)
        acc = (y_pred_all == y_true_all).sum() * 100 / len(y_true_all)
        logging.info("Prototype matched inference accuracy: {:.2f}%".format(acc))

        # logging long-tail accuracies
        if "lt" in self.args["dataset"] and self.args["dataset"] in LONGTAIL_SPLIT_SPEC:
            thresholds = LONGTAIL_SPLIT_SPEC[self.args["dataset"]]
            head_threshold = thresholds["head_threshold"]
            tail_threshold = thresholds["tail_threshold"]

            current_train_class_counts = Counter(self.train_dataset.labels)
            self._known_classes_histogram.update(current_train_class_counts)
            all_train_class_counts = self._known_classes_histogram

            head_classes = {cls for cls, count in all_train_class_counts.items() if count > head_threshold}
            tail_classes = {cls for cls, count in all_train_class_counts.items() if count < tail_threshold}
            mid_classes = set(all_train_class_counts.keys()) - head_classes - tail_classes

            head_accs, mid_accs, tail_accs = [], [], []
            head_correct, head_total = 0, 0
            mid_correct, mid_total = 0, 0
            tail_correct, tail_total = 0, 0

            for cls, stats in class_stats.items():
                n = stats["count"]
                correct = stats["correct"]
                acc = 100 * correct / n if n > 0 else 0
                if cls in head_classes:
                    head_accs.append(acc)
                    head_correct += correct
                    head_total += n
                elif cls in tail_classes:
                    tail_accs.append(acc)
                    tail_correct += correct
                    tail_total += n
                elif cls in mid_classes:
                    mid_accs.append(acc)
                    mid_correct += correct
                    mid_total += n

            # logging average accuracies
            head_avg = np.mean(head_accs) if head_accs else 0.0
            mid_avg = np.mean(mid_accs) if mid_accs else 0.0
            tail_avg = np.mean(tail_accs) if tail_accs else 0.0

            logging.info(f"\n{'='*60}")
            logging.info(f"[Task {self._cur_task}] Long-tail Performance Analysis:")
            logging.info(f"{'='*60}")
            logging.info(f"Head Classes ({len(head_classes)} classes, {head_total} samples): Class-averaged accuracy: {head_avg:.2f}%")
            logging.info(f"Mid Classes ({len(mid_classes)} classes, {mid_total} samples): Class-averaged accuracy: {mid_avg:.2f}%")
            logging.info(f"Tail Classes ({len(tail_classes)} classes, {tail_total} samples): Class-averaged accuracy: {tail_avg:.2f}%")
            logging.info(f"{'='*60}\n")

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
