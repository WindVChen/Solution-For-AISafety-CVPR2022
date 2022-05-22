from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import CustomDataset

from .transforms import build_transformer, TwoCropsTransform, GaussianBlur
from .auto_augmentation import ImageNetPolicy
from .sampler import build_sampler
from .metrics import build_evaluator
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa


def custom_autoAug():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            sometimes(iaa.ChannelShuffle(0.35)),
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 11)),
                               # blur image using local medians with kernel sizes between 2 and 7
                               iaa.MotionBlur(),
                               iaa.imgcorruptlike.ZoomBlur(),
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           iaa.SimplexNoiseAlpha(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           ])),
                           iaa.OneOf([
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               iaa.SaltAndPepper(p=(0.03, 0.1), per_channel=0.5),
                           ]),
                           # add gaussian noise to images
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               iaa.Cutout(nb_iterations=2),
                               iaa.imgcorruptlike.Snow(severity=(1, 3)),
                               iaa.imgcorruptlike.Spatter(severity=(1, 4)),
                               iaa.Rain(),
                               iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03)),
                               iaa.Fog(),
                           ]),
                           iaa.Invert(0.05, per_channel=True),  # invert color channels
                           iaa.Add((-10, 10), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.OneOf([
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.FrequencyNoiseAlpha(
                                   exponent=(-4, 0),
                                   first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                   second=iaa.LinearContrast((0.5, 2.0))
                               )
                           ]),
                           iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           iaa.Grayscale(alpha=(0.0, 1.0)),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           # move pixels locally around (with random strengths)
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # sometimes move parts of the image around
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    ).augment_image
    return seq

def fix_custom_autoAug():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            sometimes(iaa.ChannelShuffle(0.35)),
            iaa.Sometimes(0.3, iaa.JpegCompression(compression=(0, 90))),
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                # shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 3),
                       [
                           # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 11)),
                               # blur image using local medians with kernel sizes between 2 and 7
                               iaa.MotionBlur(),
                               iaa.imgcorruptlike.ZoomBlur(),
                               iaa.imgcorruptlike.DefocusBlur(severity=(1, 3)),
                           ]),
                           # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           # iaa.SimplexNoiseAlpha(iaa.OneOf([
                           #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                           #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           # ])),
                           iaa.OneOf([
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               iaa.SaltAndPepper(p=(0.03, 0.1), per_channel=0.5),
                               iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.05), per_channel=0.5)
                           ]),
                           # add gaussian noise to images
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               iaa.Cutout(nb_iterations=2),
                               iaa.imgcorruptlike.Snow(severity=(1, 3)),
                               iaa.imgcorruptlike.Spatter(severity=(1, 4)),
                               iaa.Rain(),
                               iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03)),
                               iaa.Fog(),
                           ]),
                           # iaa.Invert(0.05, per_channel=True),  # invert color channels
                           iaa.Add((-10, 10), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           # iaa.OneOf([
                           #     iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           #     iaa.FrequencyNoiseAlpha(
                           #         exponent=(-4, 0),
                           #         first=iaa.Multiply((0.5, 1.5), per_channel=True),
                           #         second=iaa.LinearContrast((0.5, 2.0))
                           #     )
                           # ]),
                           iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           iaa.Grayscale(alpha=(0.0, 1.0)),
                           # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           # move pixels locally around (with random strengths)
                           # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # sometimes move parts of the image around
                           # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    ).augment_image
    return seq

class AddSaltPepperNoise(object):

    def __init__(self, density=0.):
        self.density = density

    def __call__(self, img):

        img = np.array(img)                                                             # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)                                               # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0                                                              # 椒
        img[mask == 1] = 255                                                            # 盐
        img= Image.fromarray(img.astype('uint8')).convert('RGB')                        # numpy转图片
        return img

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

def build_custom_dataloader(data_type, cfg_dataset):
    """
    arguments:
        - data_type: 'train', 'test', 'val'
        - cfg_dataset: configurations of dataset
    """
    assert data_type in cfg_dataset
    # build transformer
    image_reader = cfg_dataset[data_type].get('image_reader', {})
    if isinstance(cfg_dataset[data_type]['transforms'], list):
        transformer = build_transformer(cfgs=cfg_dataset[data_type]['transforms'],
                                        image_reader=image_reader)
    else:
        transformer = build_common_augmentation(cfg_dataset[data_type]['transforms']['type'])

    # build evaluator
    evaluator = None
    if data_type == 'test' and cfg_dataset[data_type].get('evaluator', None):
        evaluator = build_evaluator(cfg_dataset[data_type]['evaluator'])
    # build dataset
    if cfg_dataset['type'] == 'custom':
        CurrDataset = CustomDataset
    elif cfg_dataset['type'] == 'multiclass':
        raise NotImplementedError
    else:
        raise NotImplementedError

    if cfg_dataset['read_from'] == 'osg':
        dataset = CurrDataset(
            root_dir='',
            meta_file=cfg_dataset[data_type]['meta_file'],
            transform=transformer,
            read_from='osg',
            evaluator=evaluator,
            image_reader_type=image_reader.get('type', 'pil'),
            osg_server=cfg_dataset[data_type]['osg_server'],
        )
    else:
        dataset = CurrDataset(
            root_dir=cfg_dataset[data_type]['root_dir'],
            meta_file=cfg_dataset[data_type]['meta_file'],
            transform=transformer,
            read_from=cfg_dataset['read_from'],
            evaluator=evaluator,
            image_reader_type=image_reader.get('type', 'pil')
        )
    # initialize kwargs of sampler
    cfg_dataset[data_type]['sampler']['kwargs'] = {}
    cfg_dataset['dataset'] = dataset
    # build sampler
    sampler = build_sampler(cfg_dataset[data_type]['sampler'], cfg_dataset)
    if data_type == 'train' and cfg_dataset['last_iter'] >= cfg_dataset['max_iter']:
        return {'loader': None}
    # build dataloader
    loader = DataLoader(dataset=dataset,
                        batch_size=cfg_dataset['batch_size'],
                        shuffle=False if sampler is not None else True,
                        num_workers=cfg_dataset['num_workers'],
                        pin_memory=cfg_dataset['pin_memory'],
                        sampler=sampler)
    return {'type': data_type, 'loader': loader}

def build_common_augmentation(aug_type):
    """
    common augmentation settings for training/testing ImageNet
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    grayscale_normalize = transforms.Normalize(mean=[0.449], std=[0.226])
    if aug_type == 'STANDARD':
        augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'GRAYSCALE':
        augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            grayscale_normalize,
        ]
    elif aug_type == 'AUTOAUG':
        augmentation = [
            transforms.RandomResizedCrop(224),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ]

    elif aug_type == 'CUSTOMAUTOAUG':
        augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.Compose([
                np.asarray,
                custom_autoAug(),
                np.uint8,
                Image.fromarray
            ]),
            transforms.ToTensor(),
            normalize,
        ]

    elif aug_type == 'MORECUSTOMAUTOAUG':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
            transforms.Compose([
                np.asarray,
                fix_custom_autoAug(),
                np.uint8,
                Image.fromarray
            ]),
            transforms.ToTensor(),
            normalize,
        ]

    elif aug_type == 'MOCOV1':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'MOCOV2' or aug_type == 'SIMCLR':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'LINEAR':
        augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'ONECROP':
        augmentation = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'JUSTNORM':
        augmentation = [
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'ONECROPGRAYSCALE':
        augmentation = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            grayscale_normalize,
        ]
    elif aug_type == 'CUSTOM':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomOrder([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply([AddSaltPepperNoise(0.01)], p=0.5),
            transforms.RandomApply([AddGaussianNoise(mean=0, variance=1, amplitude=5)], p=0.5),
            ]),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'ADVGEN':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.)),
            transforms.RandomOrder([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            ]),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'MORENOISECUSTOM':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.)),
            transforms.RandomOrder([
                transforms.Compose([
                    np.asarray,
                    transforms.RandomChoice([
                        transforms.RandomApply([iaa.MotionBlur(k=11).augment_image], p=0.4),
                        transforms.RandomApply([iaa.imgcorruptlike.ZoomBlur(severity=4).augment_image], p=0.4)
                    ]),
                    np.uint8,
                    Image.fromarray
                ]),
                transforms.Compose([
                    np.asarray,
                    transforms.RandomApply([iaa.Cutout(fill_mode="gaussian", fill_per_channel=True).augment_image], p=0.4),
                    np.uint8,
                    Image.fromarray
                ]),
                transforms.Compose([
                    np.asarray,
                    transforms.RandomApply([iaa.imgcorruptlike.Snow(severity=1).augment_image], p=0.4),
                    np.uint8,
                    Image.fromarray
                ]),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.4),
                transforms.RandomApply([transforms.RandomRotation(degrees=[90, 90])], p=0.4),
                transforms.RandomApply([AddSaltPepperNoise(0.01)], p=0.4),
                transforms.RandomApply([AddGaussianNoise(mean=0, variance=1, amplitude=5)], p=0.4),
            ]),
            transforms.ToTensor(),
            normalize
        ]
    else:
        raise RuntimeError("undefined augmentation type for ImageNet!")

    if aug_type in ['MOCOV1', 'MOCOV2', 'SIMCLR']:
        return TwoCropsTransform(transforms.Compose(augmentation))
    else:
        return transforms.Compose(augmentation)