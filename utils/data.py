import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from datasets import load_dataset
import os
from collections import Counter


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100LT(iData):
    def __init__(self, args): 
        super().__init__()
        self.args = args
        #self.imbalance_ratio = args.imbalance_ratio #"r-10", "r-50", or "r-100"
        self.use_path = False
        #self.class_order_mode = args.class_order_mode # 'random','inc','dec'

        # Follow the same pattern as iCIFAR224
        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        
        self.common_trsf = [
            # transforms.ToTensor() is already included in build_transform
        ]

        self.class_order = []

    def download_data(self):
        train_dataset = load_dataset("tomas-gajarsky/cifar100-lt", self.args["imbalance_ratio"], split="train")
        test_dataset = load_dataset("tomas-gajarsky/cifar100-lt", self.args["imbalance_ratio"], split="test")
        
        self.train_data = np.stack([np.array(x["img"]) for x in train_dataset])
        self.train_targets = np.array([x["fine_label"] for x in train_dataset])
        self.test_data = np.stack([np.array(x["img"]) for x in test_dataset])
        self.test_targets = np.array([x["fine_label"] for x in test_dataset])
        
        class_counts = {cls: 0 for cls in range(100)}
        for label in self.train_targets:
            class_counts[label] += 1

        class_order_mode = self.args["class_order_mode"]
        if class_order_mode == "dec":
            sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
            self.class_order = [cls for cls, _ in sorted_classes]
        elif class_order_mode == "inc":
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
            self.class_order = [cls for cls, _ in sorted_classes]
        elif class_order_mode == "random":
            np.random.seed(1993)
            sorted_classes = [(cls, class_counts[cls]) for cls in np.random.permutation(100)]
            self.class_order = [cls for cls, _ in sorted_classes]
        elif class_order_mode =="first":
            self.class_order = create_head_first_inc_ordering(class_counts, 10)
        elif class_order_mode == "last":
            self.class_order = create_tail_first_dec_ordering(class_counts, 10) 
        else:
            raise ValueError("Invalid class_order_mode: choose from ['random', 'inc', 'dec', 'first', 'last']")
        


def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t


class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root_dir = os.getenv("IMAGENETR_ROOT")
        train_dir = os.path.join(root_dir, "train")
        test_dir = os.path.join(root_dir, "test")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNetR_Longtail(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = []

        self.class_order = np.arange(200).tolist()  # sorted label IDs (0–199)

    def _load_txt_split(self, root_dir, txt_file):
        data, targets = [], []
        with open(txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(root_dir, path)
                data.append(full_path)
                targets.append(int(label))
        return np.array(data), np.array(targets)

    def download_data(self):
        root_dir = os.getenv("IMAGENETR_ROOT")
        train_root_dir = os.path.join(root_dir, "train")
        test_root_dir = os.path.join(root_dir, "test")
        splits_dir = os.path.join(root_dir, "splits")

        train_txt = os.path.join(splits_dir, "train_longtail.txt")
        test_txt = os.path.join(splits_dir, "test.txt")

        self.train_data, self.train_targets = self._load_txt_split(train_root_dir, train_txt)
        self.test_data, self.test_targets = self._load_txt_split(test_root_dir, test_txt)
        
        class_counts = Counter(self.train_targets)
        mode = self.args.get("class_order_mode", "default")
        
        if mode == "dec":
            sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
            self.class_order = [cls for cls, _ in sorted_classes]
        elif mode == "inc":
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
            self.class_order = [cls for cls, _ in sorted_classes]
        elif mode == "random":
            np.random.seed(self.args.get("seed", 1993))
            self.class_order = list(np.random.permutation(len(class_counts)))
        elif mode == "first":
            self.class_order = create_head_first_inc_ordering(class_counts, num_head_classes=20)
        elif mode == "last":
            self.class_order = create_tail_first_dec_ordering(class_counts, num_tail_classes=20)
        else:
            raise ValueError("Invalid class_order_mode: choose from ['random', 'inc', 'dec', 'first', 'last']")


class iImageNetA(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-a/train/"
        test_dir = "./data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class CUB(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def _load_split(self, root_dir, split_file):
        data, targets = [], []
        with open(split_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(root_dir, path)
                data.append(full_path)
                targets.append(int(label))
        return np.array(data), np.array(targets)

    def download_data(self):
        root_dir = os.getenv("CUB_ROOT")
        split_dir = os.path.join(root_dir, "splits")

        train_split = os.path.join(split_dir, "train.txt")
        test_split = os.path.join(split_dir, "test.txt")

        self.train_data, self.train_targets = self._load_split(root_dir, train_split)
        self.test_data, self.test_targets = self._load_split(root_dir, test_split)

class CUB_Longtail(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def _load_split(self, root_dir, split_file):
        data, targets = [], []
        with open(split_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(root_dir, path)
                data.append(full_path)
                targets.append(int(label))
        return np.array(data), np.array(targets)

    def download_data(self):
        root_dir = os.getenv("CUB_ROOT")
        split_dir = os.path.join(root_dir, "splits")

        train_split = os.path.join(split_dir, "train_longtail.txt")
        test_split = os.path.join(split_dir, "test.txt")

        self.train_data, self.train_targets = self._load_split(root_dir, train_split)
        self.test_data, self.test_targets = self._load_split(root_dir, test_split)



class objectnet(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/objectnet/train/"
        test_dir = "./data/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/omnibenchmark/train/"
        test_dir = "./data/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class vtab(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/vtab-cil/vtab/train/"
        test_dir = "./data/vtab-cil/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
  
        
def create_head_first_inc_ordering(class_counts, num_head_classes=10):
    """
    Create custom ordering: top N head classes first, then remaining in increasing order
    
    Args:
        class_counts: dict {class_id: sample_count}
        num_head_classes: number of head classes to put first (default 10)
    
    Returns:
        list: ordered class IDs
    """
    # Sort all classes by sample count (decreasing)
    sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
    # Get the top N head classes
    head_classes = [cls for cls, _ in sorted_classes[:num_head_classes]]
    # Get remaining classes and sort them in increasing order
    remaining_classes = sorted_classes[num_head_classes:]
    tail_classes = [cls for cls, _ in sorted(remaining_classes, key=lambda x: x[1])]
    
    # Combine: head classes first, then increasing tail classes
    final_order = head_classes + tail_classes
    return final_order

def create_tail_first_dec_ordering(class_counts, num_tail_classes=10):
    """
    Create custom ordering: bottom N tail classes first, then remaining in decreasing order
    
    Args:
        class_counts: dict {class_id: sample_count}
        num_tail_classes: number of tail classes to put first (default 10)
    
    Returns:
        list: ordered class IDs
    """
    # Sort all classes by sample count (increasing)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    
    # Get the bottom N tail classes (smallest sample counts)
    tail_classes = [cls for cls, _ in sorted_classes[:num_tail_classes]]
    
    # Get remaining classes and sort them in decreasing order
    remaining_classes = sorted_classes[num_tail_classes:]
    head_classes = [cls for cls, _ in sorted(remaining_classes, key=lambda x: -x[1])]
    
    # Combine: tail classes first, then decreasing head classes
    final_order = tail_classes + head_classes
    return final_order