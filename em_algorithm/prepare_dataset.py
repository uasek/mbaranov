import numpy as np
from tqdm.notebook import tqdm

class PictureNoiser:
    def __init__(self, F, B, A):
        self.B = B.copy()
        self.F = F.copy()
        self.A = A.copy()

        self.H, self.W = B.shape
        self.h, self.w = F.shape

    def allocate_face(self):
        ind = np.unravel_index(
            np.random.choice(
                a=np.arange(self.A.size),
                p=self.A.ravel(),
                size=1
            )[0],
            shape=self.A.shape
        )

        image = self.B.copy()
        image[ind[0]:ind[0] + self.h, ind[1]:ind[1] + self.w] = self.F

        return image

    def noise_image(self, img, s):
        
        noise = np.random.randn(*img.shape) * s
        return img + noise


    def generate_sample(self, n, s):

        images = []

        for i in tqdm(range(n)):
            img = self.allocate_face()
            img = self.noise_image(img, s)

            images.append(img)

        images = np.array(images).transpose(1, 2, 0)

        return images