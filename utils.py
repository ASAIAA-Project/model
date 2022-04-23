import json
import logging
import shutil
from tabnanny import check
import torch
import random

import numpy as np
import os


def set_all_random_seed(seed, rank=0):
    """Set random seed.
    Args:
        seed (int): Nonnegative integer.
        rank (int): Process rank in the distributed training. Defaults to 0.
    """
    assert seed >= 0, f"Got invalid seed value {seed}."
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    if not json_path.parent.exists():
        json_path.parent.mkdir(parents=True)
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = checkpoint / 'last.pth.tar'
    if not checkpoint.exists():
        print("Checkpoint Directory does not exist! Making directory {}".format(
            checkpoint))
        checkpoint.mkdir(parent=True)

    filepath = str(filepath)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, checkpoint / 'best.pth.tar')


def load_best_checkpoint(save_dir, model):
    checkpoint_path = save_dir / 'best.pth.tar'
    checkpoint = str(checkpoint_path)
    if not checkpoint_path.exists():
        print("File doesn't exist {}".format(checkpoint))
    else:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

    return


def load_checkpoint(checkpoint_path, model, optimizer_R=None, optimizer_D=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not checkpoint_path.exists():
        raise ("File doesn't exist {}".format(checkpoint_path))
    checkpoint_path = str(checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path))
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer_R:
        optimizer_R.load_state_dict(checkpoint['optim_R_dict'])
    if optimizer_D:
        optimizer_D.load_state_dict(checkpoint['optim_D_dict'])

    return checkpoint

# helper functions for CAAM
def norm_att_map(att_map):
    _min = torch.min(att_map)
    _max = torch.max(att_map)
    att_norm = (att_map - _min) / (_max - _min)
    return att_norm


def cammed_image(image, mask, require_norm=False):
    if require_norm:
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return heatmap * 255., cam * 255.


def intensity_to_rgb(intensity, normalize=False):
    """
    Convert a 1-channel matrix of intensities to an RGB image employing a colormap.
    This function requires matplotlib. See `matplotlib colormaps
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_ for a
    list of available colormap.
    Args:
        intensity (np.ndarray): array of intensities such as saliency.
        cmap (str): name of the colormap to use.
        normalize (bool): if True, will normalize the intensity so that it has
            minimum 0 and maximum 1.
    Returns:
        np.ndarray: an RGB float32 image in range [0, 255], a colored heatmap.
    """
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = 'jet'
    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0

