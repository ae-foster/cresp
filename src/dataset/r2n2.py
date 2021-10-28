import json
import math
from os import path, listdir
from pathlib import Path
from typing import Dict
from collections import defaultdict

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tabulate import tabulate
from glob import iglob
import logging

from .augmentation import *

log = logging.getLogger(__name__)


SYNSET_DICT_DIR = Path(__file__).resolve().parent
MAX_CAMERA_DISTANCE = 1.75  # Constant from R2N2.


class R2N2(torch.utils.data.Dataset):
    """
    This class loads the R2N2 dataset from a given directory into a Dataset object.
    The R2N2 dataset contains 13 categories that are a subset of the ShapeNetCore v.1
    dataset. The R2N2 dataset also contains its own 24 renderings of each object and
    voxelized models. Most of the models have all 24 views in the same split, but there
    are eight of them that divide their views between train and test splits.

    Warning: use of this class within a `DataLoader` requires the use of a custom `collate_fn`, see
    `collator.py`.

    """

    def __init__(
        self,
        root,
        splits_file=None,
        print_classes: bool = True,
        crop_size: int = 64,
        crop_area: tuple = (0.08, 1),
        crop_ratio: tuple = (1.0, 1.0),
        limit_num_views: int = 24,
        subsample: bool = False,
        transform=None,
        cov_transform=None,
        **kwargs
    ):
        """
        Store each object's synset id and models id the given directories.

        Args:
            root (path): Path to the R2N2 dataset.
            splits_file (path): File containing the train/val/test splits.
            num_random_views (int): number of views to randomly sample, return all if 0

        """
        super().__init__()
        """
        Set up lists of synset_ids and model_ids.
        """
        self.synset_ids = []
        self.model_ids = []
        self.synset_inv = {}
        self.synset_start_idxs = {}
        self.synset_num_models = {}

        self.limit_num_views = limit_num_views
        self.subsample = subsample
        self.root = root
        self.transform = eval(transform) if transform else None
        self.cov_transform = eval(cov_transform) if cov_transform else None
        self.is_preloaded = False

        self.crop_size = crop_size
        self.crop_area = crop_area
        self.crop_ratio = crop_ratio

        # Check that path is valid
        if not (path.exists(root) and path.isdir(root) and listdir(root)):
            raise FileExistsError("Dataset is empty or not located at {}".format(root))

        # Synset dictionary mapping synset offsets in R2N2 to corresponding labels.
        with open(path.join(SYNSET_DICT_DIR, "r2n2_synset_dict.json"), "r") as read_dict:
            self.synset_dict = json.load(read_dict)
        # Inverse dicitonary mapping synset labels to corresponding offsets.
        self.synset_inv = {label: offset for offset, label in self.synset_dict.items()}

        # If a split file is used, we select the train split
        # Default: no split file
        # Store synset and model ids of objects mentioned in the splits_file.
        if splits_file:
            with open(splits_file) as splits:
                split_dict = json.load(splits)["train"]
        else:
            split_dict = self.get_split_dict()

        synset_set = set()
        # Store lists of views of each model in a list.
        self.views_per_model_list = []
        # Store tuples of synset label and total number of views in each category in a list.
        synset_num_instances = []
        for synset in split_dict.keys():

            synset_set.add(synset)
            self.synset_start_idxs[synset] = len(self.synset_ids)
            # Start counting total number of views in the current category.
            synset_view_count = 0
            models = list(split_dict[synset].keys())
            # Randomization should be handled by the sampler
            # random.shuffle(models)
            for i, model in enumerate(models):
                if self.subsample > 0 and i >= self.subsample:
                    break

                self.synset_ids.append(synset)
                self.model_ids.append(model)

                model_views = np.array(split_dict[synset][model][: self.limit_num_views])
                self.views_per_model_list.append(model_views)
                synset_view_count += len(model_views)

            synset_num_instances.append((self.synset_dict[synset], len(models), synset_view_count))
            self.synset_num_models[synset] = len(models)

        if print_classes:
            headers = ["category", "#objects", "#images"]
            synset_num_instances.append(
                ("total", sum(n for _, n, _ in synset_num_instances), sum(n for _, _, n in synset_num_instances))
            )
            log.info(
                "Dataset information\n" + tabulate(synset_num_instances, headers, numalign="left", stralign="center")
            )

    def get_split_dict(self):
        """Returns a dictionary describing the dataset of the following form.

        Top level keys are synsets (class labels).
        Values of top level keys are themselves dictionaries.
        Keys of the low level dictionaries are hashes signifying particular models (objects).
        Values of low level dictionaries are lists of integers representing allowed views.

        """
        split_dict = defaultdict(dict)
        paths = filter(path.isdir, iglob("{}/*/*".format(self.root)))
        for model_path in paths:
            synset, model = tuple(model_path.split("/")[-2:])
            split_dict[synset][model] = list(range(24))
        return split_dict

    def __len__(self):
        """
        Return number of total models in the loaded dataset.
        """
        return len(self.model_ids)

    def load_model(self, index):
        model_views = self.views_per_model_list[index]
        synset_id, model_id = self._get_item_ids(index)

        # Retrieve R2N2's renderings if required.
        rendering_path = path.join(self.root, synset_id, model_id, "rendering")
        # Read metadata file to obtain params for calibration matrices.
        with open(path.join(rendering_path, "rendering_metadata.txt"), "r") as f:
            metadata_lines = f.readlines()

        keys = tuple(self.synset_dict.keys())
        labels = torch.tensor([keys.index(synset_id)]).squeeze()

        return R2N2Object(
            len(model_views),
            rendering_path,
            self.transform,
            self.cov_transform,
            metadata_lines,
            labels,
            self.crop_size,
            self.crop_area,
            self.crop_ratio,
        )

    def __getitem__(self, index: int):
        """
        Returns a `R2N2Object` instance corresponding to the given object.
        To extract views from the given object, either in a random or deterministic fashion, you should
        use a suitable `collate_fn` within the `DataLoader`.
        If calling `__getitem__` directly, use the `.make_views` method of the resulting object to extract views.
        To request all views, use `o.make_views(list(range(o.total_views)))`.
        """
        if self.is_preloaded:
            views, covariates, labels = self.views[index], self.covariates[index], self.labels[index]
            return (views, covariates), labels
        else:
            return self.load_model(index)

    def _get_item_ids(self, idx):
        """
        Read a model by the given index.

        Args:
            idx: The idx of the model to be retrieved in the dataset.

        Returns:
            dictionary with following keys:
            - synset_id (str): synset id
            - model_id (str): model id
        """
        return self.synset_ids[idx], self.model_ids[idx]


class R2N2Object:
    def __init__(
        self,
        total_views,
        rendering_path,
        transform,
        cov_transform,
        metadata_lines,
        labels,
        crop_size,
        crop_area,
        crop_ratio,
    ):
        self.total_views = total_views
        self.rendering_path = rendering_path
        self.transform = transform
        self.cov_transform = cov_transform
        self.labels = labels
        self.metadata_lines = metadata_lines
        self.crop_size = crop_size
        self.crop_area = crop_area
        self.crop_ratio = crop_ratio
        self.array = None

    def make_views_boxes(self, idxs, boxes):
        views, covariates = [], []
        for i, bounding_box in zip(idxs, boxes):
            # Read images.
            img = self.load_image(i)
            # Cropping now forms part of the covariate / rendering process
            img = transforms.functional.resized_crop(img, *bounding_box, self.crop_size, Image.LANCZOS)
            if self.transform:
                img = self.transform(img)
            views.append(img)

            # Get camera calibration.
            azim, elev, yaw, dist_ratio, fov = [float(v) for v in self.metadata_lines[i].strip().split(" ")]
            azim = azim / 360.0
            elev = elev / 360.0  # (elev - 25.0) / (30.0 - 25.0)
            dist = dist_ratio  # * MAX_CAMERA_DISTANCE  # (dist - 0.65) / (0.95 - 0.65)

            featurized_covariates = torch.tensor(self.featurization(azim, elev, dist, *bounding_box))
            featurized_covariates = self.cov_transform(featurized_covariates)
            covariates.append(featurized_covariates)

        covariates = torch.stack(covariates)
        views = torch.stack(views)
        return (views, covariates), self.labels

    def make_views(self, idxs):
        views, covariates = [], []
        for i in idxs:
            # Read images.
            img = self.load_image(i)
            # Cropping now forms part of the covariate / rendering process
            bounding_box = list(transforms.RandomResizedCrop.get_params(img, self.crop_area, self.crop_ratio))
            img = transforms.functional.resized_crop(img, *bounding_box, self.crop_size, Image.LANCZOS)
            if self.transform:
                img = self.transform(img)
            views.append(img)

            # Get camera calibration.
            azim, elev, yaw, dist_ratio, fov = [float(v) for v in self.metadata_lines[i].strip().split(" ")]
            azim = azim / 360.0
            elev = elev / 360.0  # (elev - 25.0) / (30.0 - 25.0)
            dist = dist_ratio  # * MAX_CAMERA_DISTANCE  # (dist - 0.65) / (0.95 - 0.65)

            featurized_covariates = torch.tensor(self.featurization(azim, elev, dist, *bounding_box))
            featurized_covariates = self.cov_transform(featurized_covariates)
            covariates.append(featurized_covariates)

        covariates = torch.stack(covariates)
        views = torch.stack(views)
        return (views, covariates), self.labels

    @staticmethod
    def featurization(azim, elev, dist, i, j, h, w):
        """
        Performs a range of feature expansion operations on raw covariates.
        *Note*: we assume that features will be normalized using batch norm before further processing, to control
        their scale

        Args:
            - the unprocesses covariates: azimuth, elevation, distance, y-coord, x-coord, height, width of box

        Returns:
            - featurized_covariates: a list of  items: the featurized covariates
        """

        # Harmonics of the azim (6 features)
        azim_features = [math.sin(2 * math.pi * n * azim) for n in range(1, 4)] + [
            math.cos(2 * math.pi * n * azim) for n in range(1, 4)
        ]
        # elev features (1 feature)
        elev_features = [elev]
        # dist features (1 feature)
        dist_features = [dist]
        # Box features: raw, plus area and centre (7 features)
        box_features = [i, j, h, w, h * w, j + w / 2, i + h / 2]
        # All features (15 features)
        all_features = azim_features + elev_features + dist_features + box_features
        return all_features

    def load_image(self, i):
        if self.array is None:
            npz_path = path.join(self.rendering_path, "all.npz")
            try:
                archive = np.load(npz_path)
                self.array = archive["arr_0"]
                archive.close()
            except (FileNotFoundError, ValueError):
                self.array = False
        if self.array is False:
            # Falling back on the old png method
            image_path = path.join(self.rendering_path, "%02d.png" % i)
            return Image.open(image_path).convert("RGB")
        else:
            return Image.fromarray(self.array[i, :, :, :]).convert("RGB")
