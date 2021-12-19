import random


class ImagePool(object):
    def __init__(self, pool_size=10):
        self.pool_size = pool_size
        if pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def model_name(self):
        return self.__class__.__name__

    def query(self, image):
        if self.pool_size == 0:
            return image

        if self.num_imgs < self.pool_size:
            self.num_imgs += 1
            # TODO might be necessary to do copy
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                random_idx = random.randint(0, len(self.images) - 1)
                # TODO might be necessary to do copy
                tmp = self.images[random_idx]
                self.images[random_idx] = image
                return tmp
            else:
                return image
