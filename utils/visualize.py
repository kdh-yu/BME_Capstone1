import matplotlib.pyplot as plt
import numpy as np

def visualize_retrieval(retrieval, 
                        target = None,
                        figsize: tuple = (10, 3)):
    image = retrieval['image']
    mask = retrieval['mask']
    plt.figure(figsize=figsize)
    
    l = len(image)
    if target is not None:
        l += 1
        image = [target['image']] + image
        mask = [target['mask']] + mask
        
    for i in range(l):
        plt.subplot(2, l, i+1)
        img = image[i]
        #img = denormalize(img)
        plt.imshow(img.permute(1, 2, 0))
        plt.axis('off')
        if (target is not None) and (i == 0):
            plt.title('Target')
        else:
            if target is not None:
                plt.title(f'Retrieval {i}')
            else:
                plt.title(f'Retrieval {i+1}')
        
    for i in range(l):
        plt.subplot(2, l, i+1+l)
        if type(mask[i]) == np.ndarray:
            plt.imshow(mask[i], cmap='gray')
        else:
            plt.imshow(mask[i].permute(1, 2, 0), cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(hspace=0.0)
    plt.tight_layout()
    plt.show()