import numpy as np
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    size = 256, 256
    im = Image.open(image)
    im.thumbnail(size, Image.ANTIALIAS)
    im = im.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npIm = np.array(im)
    npIm = npIm/255.
        
    imgA = npIm[:,:,0]
    imgB = npIm[:,:,1]
    imgC = npIm[:,:,2]
    
    imgA = (imgA - 0.485)/(0.229) 
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)
        
    npIm[:,:,0] = imgA
    npIm[:,:,1] = imgB
    npIm[:,:,2] = imgC
    
    npIm = np.transpose(npIm, (2,0,1))
    
    return npIm


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array(img_normal_means)
    std = np.array(img_normal_std)
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax