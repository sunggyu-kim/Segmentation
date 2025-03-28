import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        #모든 이미지파일을 로드시, 얼라인이 맞게 load해야할것
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        
    def __getitem__(self, idx):
        # 이미지와 마스크 불러오기
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # mask된 객체들은 각기 다른색으로 지정되어있음. 배경은 보통 0
        # unique함수를 써서 mask안에 라벨링값(pixel을 가져옴)
        obj_ids = torch.unique(mask) 
        # 배경 아이디를 제거해서 mask객체를 저장
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        
        #색상 구분 마스크 이미지를 객체별로 나눈 이진마스크로 변경
        #두개의 차원을 추가해줘서 mask 객체와 비교가능하도록 변경
        #브로드 캐스팅 기법이라고 하는데, 아직 명확하게 와닫지 않는다.
        #아 마스크값은 각각 픽셀마다 1,2 이런식으로 마스킹을 해놨기 때문에, mask랑 object_ids(1~4)랑 같은값이 나오면 그 객체만 저장하는 방식이 
        masks = (mask == obj_ids[:,None, None]).to(dtype=torch.unit8)
        
        #각각의 마스크와 bounding box를 일치와 시킴
        boxes = masks_to_boxes(masks)
        
        #there is only one class(human 유무)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        image_id = idx
        area = (boxes[:,3] - boxes[:,1]*(boxes[:,2] - boxes[:,0]))
        
        #모든 객체(사람)가 없을때 : 배경 == obj[0]
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  
        
        #image를 torchtensor로 wrapping 함 : 연산의 편의 tv_tensors
        img = tv_tensors.Image(img)
        
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    def __len__(self):
        return len(self.imgs)