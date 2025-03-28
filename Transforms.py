# Transforms : argumentation들을 다르게 부르는것
#              data를 요리조리 변경하니 Transform이 맞는듯

from torchvision.transforms import v2 as T
import torch

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True)) #필수적인 전처리 : unit8[0,255] -> float32[0.0, 1.0] 정규화
    transforms.append(T.ToPureTensor()) #PIL.Image or numpy.ndarray를 torch.Tensor로 변환함
    return T.Compose(transforms)

finish = 1