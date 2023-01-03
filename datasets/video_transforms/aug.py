    
import numpy as np
import PIL

class Compose:
    def __init__(self, aug_list):
        self.aug_list = aug_list
    
    def __call__(self, img_list):
        for aug in self.aug_list:
            img_list = aug(img_list)
        return img_list
    
class RandomHorizontalFlip:
    def __init__(self):
        self.p = 0.5
    
    def __call__(self, img_list):
        if np.random.uniform() < self.p:
            return img_list
        return [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in img_list]
        
class RandomRotation:
    def __init__(self, angle):
        self.min_angle = -angle
        self.max_angle = angle
    
    def __call__(self, img_list):
        angle = np.random.uniform(self.min_angle, self.max_angle)
        return [img.rotate(angle) for img in img_list]
    
class ColorJitter:
    def __init__(self, brightness=0, contrast=0, ):
        self.min_brightness = 1.0 - brightness
        self.max_brightness = 1.0 + brightness
        self.min_contrast = 1.0 - contrast
        self.max_contrast = 1.0 + contrast
    
    def __call__(self, img_list):
        brightness = np.random.uniform(self.min_brightness, self.max_brightness)
        def _brightness_func(img):
            enhancer = PIL.ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
            return img
        
        contrast = np.random.uniform(self.min_contrast, self.max_contrast)
        def _contrast_func(img):
            enhancer = PIL.ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
            return img
        
        if np.random.uniform() > 0.5:
            img_list = [_brightness_func(img) for img in img_list]
            img_list = [_contrast_func(img) for img in img_list]
        else:
            img_list = [_contrast_func(img) for img in img_list]
            img_list = [_brightness_func(img) for img in img_list]
            
        return img_list
    
class RandomCrop:
    def __init__(self, image_size):
        self.height = image_size[0]
        self.width = image_size[1]
    
    def __call__(self, img_list):
        img_width, img_height = img_list[0].size
        x1 = np.random.randint(0, img_width - self.width)
        y1 = np.random.randint(0, img_height - self.height)
        x2 = x1 + self.width
        y2 = y1 + self.height
        return [img.crop((x1, y1, x2, y2)) for img in img_list]
    
class Resize:
    def __init__(self, image_size):
        self.height = image_size[0]
        self.width = image_size[1]
    
    def __call__(self, img_list):
        return [img.resize((self.width, self.height), PIL.Image.BILINEAR) for img in img_list]