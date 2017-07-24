""" Routines to fetch MS COCO data.

Krzysztof Chalupka, 2017.
"""

import skimage.transform
import skimage.color
import skimage.io as io
import numpy as np
from pycocotools.coco import COCO

# Tell Python where the data is.
home = '/home/kchalupk/'
dataDir = home + 'projects/COCO/coco/'
dataType = 'train2014'
imageDir = home + 'projects/COCO/' + dataType
annFile = dataDir + 'annotations/instances_{}.json'.format(dataType)

# Initialize COCO api for instance annotations.
coco = COCO(annFile)

def get_img_and_mask(img_ids, im_size,  cat_id):
    """ Get an image and corresponding mask.
    
    Choose an id at random from image_ids. Check whether corresponding mask
    contains at least 1% positive class. If so, crop out a random patch of
    im_size and return it. If not, return good_image=False.

    Args:
        img_ids: Array of ids to choose from. Should correspond to cat_id.
        im_size: Desired image size. Images will be resized to fit.
        cat_id: Category id, obtained by coco.getCatIds.

    Returns:
        image (im_size[0], im_size[1], 3): An image of the appropriate category,
            choosen at random from img_ids, resized and converted to 3 channels.
        mask (im_size[0], im_size[1], 1): A binary mask.
        success (bool): If True, image and mask are valid. If False, then image
            and mask are None, as the randomly chosen image's pixels contain 
            less than 1% of the desired category.
    """
    img_id = np.random.choice(img_ids)
    image = coco.loadImgs(int(img_id))[0]
    annIds = coco.getAnnIds(imgIds=image['id'], catIds=[cat_id], iscrowd=None)
    image = io.imread('{}/{}'.format(imageDir, image['file_name']))
    ann = coco.loadAnns(annIds)
    mask = np.zeros((image.shape[0], image.shape[1]))
    for ann_single in ann:
        mask += coco.annToMask(ann_single)
    mask[mask > 1] = 1
    image = image / 255.
    if mask.sum() > mask.size / 100:
        image = skimage.transform.resize(image, im_size)
        mask = skimage.transform.resize(mask, im_size)
        return image, mask, True
    return None, None, False


def get_coco_batch(category, batch_size, im_size, data_type='train'):
    """ Get a batch of MS COCO data.

    Args:
        category (str): Category to choose from, e.g. 'person'.
        batch_size (int): Number of images to fetch.
        im_size (array): Image height and width.
        data_type (str): 'train', 'val' or 'test'.

    Returns:
        ims (batch_size, im_size[0], im_size[1], 3): Images,
            rescaled to the (0, 1) range on each channel.
        masks (batch_size, im_size[0], im_size[1], 1): Masks.
    """
    # Get ids of current category.
    cat_id = coco.getCatIds(catNms=[category])[0]

    # Split into train/validation/test.
    st0 = np.random.get_state()
    np.random.seed(1)
    img_ids = np.random.permutation(coco.getImgIds(catIds=cat_id))
    np.random.set_state(st0)
    n_train = int(len(img_ids) * .8)
    n_valid = int(len(img_ids) * .1)
    if data_type == 'train':
        img_ids = img_ids[:n_train]
    elif data_type == 'val':
        img_ids = img_ids[n_train:n_train+n_valid]
    elif data_type == 'test':
        img_ids = img_ids[n_train+n_valid:]

    # Fetch the batch.
    ims = np.zeros((batch_size, im_size[0], im_size[1], 3))
    masks = np.zeros((batch_size, im_size[0], im_size[1], 1))
    for img_id in range(batch_size):
        good_image = False
        while not good_image:
            image, mask, good_image = get_img_and_mask(
                img_ids, im_size, cat_id)
        if len(image.shape) == 2:
            ims[img_id] = skimage.color.gray2rgb(image)
        else:
            ims[img_id] = image
        masks[img_id, :, :, 0] = mask

    return ims, masks
