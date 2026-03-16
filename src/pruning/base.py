import torch
import torch.nn as nn
from src.models import find_prunable_blocks

class BasePruner:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        # 모델의 파라미터가 위치한 장치(CPU/CUDA)를 자동으로 감지
        self.device = next(model.parameters()).device

    def get_prunable_layers(self):
        """
        모델 내에서 프루닝 타겟이 되는 블록/레이어들을 반환.
        src.models.find_prunable_blocks와 연동하여 모델별 특성을 반영함.
        """
        model_name = self.config['model']['name']
        # 모델별로 정의된 프루닝 포인트(BasicBlock, Bottleneck, Conv2d 등)를 가져옴
        blocks_dict = find_prunable_blocks(self.model, model_name)
        
        # 블록 객체들만 리스트로 반환
        return list(blocks_dict.values())

    @torch.no_grad()
    def apply_mask_to_weights(self):
        """
        PDT(In-training) 방식에서 주로 사용.
        등록된 버퍼 'mask'를 실제 가중치와 편향에 곱하여 0으로 만듦.
        """
        for m in self.model.modules():
            # 모델 생성 시(get_model) 등록된 mask 버퍼가 있는지 확인
            if hasattr(m, 'mask'):
                # mask는 [Out_channels] 형태이므로 브로드캐스팅을 위해 view 조정
                mask = m.mask.view(-1, 1, 1, 1)
                m.weight.data.mul_(mask)
                
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.mul_(m.mask)