import torch
from .base import BasePruner
from src.models import find_prunable_blocks

EPS = 1e-8

class PATPruner(BasePruner):
    def __init__(self, model, config, sensitivity_si):
        super().__init__(model, config)
        self.sensitivity_si = sensitivity_si  # 사전 계산된 민감도 {block_name: si}
        self.global_target = config['strategy']['channel_keep_ratio']
        
        # 모델별 프루닝 가능 블록 찾기
        self.prunable_blocks = find_prunable_blocks(model, config['model']['name'])
        
        # fi(파라미터 비중) 계산을 위한 블록별 파라미터 수 측정
        self.param_counts = {
            name: sum(p.numel() for p in blk.parameters()) 
            for name, blk in self.prunable_blocks.items()
        }
        self.total_params = sum(self.param_counts.values())

    def compute_all_keep_indices(self, round_idx=1):
        """
        main.py에서 호출하는 핵심 함수.
        계산된 비율을 바탕으로 실제 남길 필터 인덱스를 추출.
        """
        # 1. 수식 전략에 따라 레이어별 프루닝 비율(%) 계산
        ratios = self.compute_all_ratios()
        
        keep_indices_dict = {}
        
        with torch.no_grad():
            for name, block in self.prunable_blocks.items():
                # 해당 라운드에서 깎아야 할 비율 (예: 40.0)
                # 반복 프루닝(n_rounds > 1)일 경우 round_idx에 따라 스케줄링 가능
                ratio = ratios.get(name, 0.0)
                
                # 수치가 0~100 사이일 경우 0~1 사이로 변환
                channel_keep_ratio = min(ratio / 100.0, 0.99) if ratio > 1 else ratio
                
                # 2. 중요도(L1-norm) 기반 필터 선택
                # ResNet/VGG 등 모델 구조에 맞춰 가중치 텐서 추출
                if hasattr(block, 'conv2'):
                    w = block.conv2.weight.data
                elif hasattr(block, 'bn1'):
                    w = block.conv1.weight.data
                else:
                    w = block.weight.data
                
                # 필터별 절댓값 합 계산 (L1-norm)
                importance = w.view(w.size(0), -1).abs().sum(dim=1).cpu()
                
                # 남길 채널 개수 계산 (최소 1개는 유지)
                num_channels = importance.numel()
                num_keep = max(1, int(num_channels * (1 - channel_keep_ratio)))
                
                # 중요도가 높은 순서대로 인덱스 추출
                keep_idx = importance.argsort(descending=True)[:num_keep].tolist()
                keep_indices_dict[name] = sorted(keep_idx)
                
        return keep_indices_dict

    def compute_all_ratios(self):
        """YAML 설정에 따른 전략 분기 및 최종 레이어별 프루닝 비율 반환"""
        st_type = self.config['strategy']['type']
        
        if st_type == "normalization":
            return self._normalization()
        elif st_type == "amplification":
            return self._amplification(p=self.config['strategy'].get('p', 2.5))
        elif st_type == "weighted_sum":
            return self._weighted_sum(beta=self.config['strategy'].get('beta', 0.5))
        else:
            raise ValueError(f"Unknown PAT strategy type: {st_type}")

    # --- 내부 수식 로직 ---
    def _normalization(self):
        wi = self._get_wi(p=1.0)
        fi = {n: c / (self.total_params + EPS) for n, c in self.param_counts.items()}
        return self._apply_score(fi, wi)

    def _amplification(self, p):
        wi = self._get_wi(p=p)
        fi = {n: c / (self.total_params + EPS) for n, c in self.param_counts.items()}
        return self._apply_score(fi, wi)

    def _weighted_sum(self, beta):
        wi = self._get_wi(p=1.0)
        fi = {n: c / (self.total_params + EPS) for n, c in self.param_counts.items()}
        
        pr_temp = {}
        for n in fi.keys():
            pr_temp[n] = self.global_target * (beta * fi[n] + (1 - beta) * wi[n])
        return self._balance(pr_temp)

    def _get_wi(self, p):
        wi = {}
        sum_inv = sum((1.0 / (abs(s) + EPS)) ** p for s in self.sensitivity_si.values())
        for n, s in self.sensitivity_si.items():
            wi[n] = ((1.0 / (abs(s) + EPS)) ** p) / (sum_inv + EPS)
        return wi

    def _apply_score(self, fi, wi):
        sum_fw = sum(fi[n] * wi[n] for n in fi.keys())
        pr_temp = {n: self.global_target * fi[n] * (wi[n] / (sum_fw + EPS)) for n in fi.keys()}
        return self._balance(pr_temp)

    def _balance(self, pr_dict):
        actual_sum = sum(pr_dict.values())
        if abs(actual_sum - self.global_target) > EPS and actual_sum > 0:
            scale = self.global_target / actual_sum
            return {n: r * scale for n, r in pr_dict.items()}
        return pr_dict