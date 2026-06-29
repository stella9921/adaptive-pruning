import torch
from .base import BasePruner
from src.models import find_prunable_blocks

EPS = 1e-8

class PATPruner(BasePruner):
    def __init__(self, model, config, sensitivity_si, topology_groups=None):
        super().__init__(model, config)
        self.sensitivity_si = sensitivity_si  # 사전 계산된 민감도 {block_name: si}
        self.global_pruning_percent = config['strategy']['pruning_ratio'] * 100.0
        self.topology_groups = topology_groups
        
        # 모델별 프루닝 가능 블록 찾기
        self.prunable_blocks = find_prunable_blocks(model, config['model']['name'])
        
        self.group_si = {}
        if self.topology_groups:
            for i, group in enumerate(self.topology_groups):
                # 그룹에 속한 레이어들의 SI 값들만 추출
                group_si_values = [self.sensitivity_si.get(name, 0.0) for name in group]
                # 그룹 SI = 평균 민감도 (레이어가 여러 개 묶인 경우 대응)
                self.group_si[i] = sum(group_si_values) / len(group_si_values) if group_si_values else 0.0
        # fi(파라미터 비중) 계산을 위한 블록별 파라미터 수 측정
        self.param_counts = {
            name: sum(p.numel() for p in blk.parameters()) 
            for name, blk in self.prunable_blocks.items()
        }
        self.total_params = sum(self.param_counts.values())

    # def compute_all_keep_indices(self, round_idx=1):
    #     """
    #     main.py에서 호출하는 핵심 함수.
    #     계산된 비율을 바탕으로 실제 남길 필터 인덱스를 추출.
    #     """
    #     # 1. 수식 전략에 따라 레이어별 프루닝 비율(%) 계산
    #     ratios = self.compute_all_ratios()
        
    #     keep_indices_dict = {}
        
    #     with torch.no_grad():
    #         for name, block in self.prunable_blocks.items():
    #             # 해당 라운드에서 깎아야 할 비율 (예: 40.0)
    #             # 반복 프루닝(n_rounds > 1)일 경우 round_idx에 따라 스케줄링 가능
    #             ratio = ratios.get(name, 0.0)
                
    #             # 수치가 0~100 사이일 경우 0~1 사이로 변환
    #             pruning_ratio = min(ratio / 100.0, 0.99) if ratio > 1 else ratio
                
    #             # 2. 중요도(L1-norm) 기반 필터 선택
    #             # ResNet/VGG 등 모델 구조에 맞춰 가중치 텐서 추출
    #             if hasattr(block, 'conv2'):
    #                 w = block.conv2.weight.data
    #             elif hasattr(block, 'bn1'):
    #                 w = block.conv1.weight.data
    #             else:
    #                 w = block.weight.data
                
    #             # 필터별 절댓값 합 계산 (L1-norm)
    #             importance = w.view(w.size(0), -1).abs().sum(dim=1).cpu()
                
    #             # 남길 채널 개수 계산 (최소 1개는 유지)
    #             num_channels = importance.numel()
    #             num_keep = max(1, int(num_channels * (1 - pruning_ratio)))
                
    #             # 중요도가 높은 순서대로 인덱스 추출
    #             keep_idx = importance.argsort(descending=True)[:num_keep].tolist()
    #             keep_indices_dict[name] = sorted(keep_idx)
                
    #     return keep_indices_dict

    def compute_all_keep_indices(self, round_idx=1):
        ratios = self.compute_all_ratios()
        keep_indices_dict = {}
        
        with torch.no_grad():
            for group_layers in self.topology_groups:
                rep_name = group_layers[0]
                block = self.prunable_blocks.get(rep_name)
                if block is None: continue

                ratio = ratios.get(rep_name, 0.0)
                # 라운드가 반복될수록 누적해서 더 많이 깎아야 하므로 
                # round_idx를 비율에 곱해줍니다 (스케줄링)
                current_ratio = min((ratio * round_idx) / 100.0, 0.99)
                
                # 가중치 추출
                if hasattr(block, 'conv2'): w = block.conv2.weight.data
                elif hasattr(block, 'bn1'): w = block.conv1.weight.data
                else: w = block.weight.data
                
                # [핵심 수정] 현재 가중치의 실제 크기를 가져옵니다 (63개면 63개)
                actual_channels = w.size(0)
                importance = w.view(actual_channels, -1).abs().sum(dim=1).cpu()
                
                # 남길 개수 계산
                num_keep = max(1, int(actual_channels * (1 - current_ratio / round_idx))) 
                # 위 식은 round_idx에 따라 매 라운드 조금씩 더 쳐내게 설계하거나,
                # 단순히 현재 채널에서 일정 비율을 유지하게 합니다.
                
                # 현재 살아있는 채널 개수 안에서만 인덱스 추출
                keep_idx = sorted(importance.argsort(descending=True)[:num_keep].tolist())
                
                for name in group_layers:
                    keep_indices_dict[name] = keep_idx
                    
        return keep_indices_dict

    def _get_wi(self, p):
        """
        [Group-aware 수정한 버전]
        개별 레이어 SI 대신 __init__에서 계산한 group_si(평균 민감도)를 사용하여 wi를 계산.
        """
        wi = {}
        # 1. 모든 그룹의 (1/SI)^p 합산 (Normalization factor)
        # self.group_si는 __init__에서 {group_idx: mean_si} 형태로 저장되어 있어야 함
        sum_inv = sum((1.0 / (abs(s) + EPS)) ** p for s in self.group_si.values())
        
        # 2. 각 그룹별 비중 계산 후 소속 레이어들에 배분
        for i, group in enumerate(self.topology_groups):
            s = self.group_si.get(i, 0.0)
            # 해당 그룹의 비중 계산
            group_val = ((1.0 / (abs(s) + EPS)) ** p) / (sum_inv + EPS)
            
            # 그룹 내 모든 레이어(예: features.0, features.3...)에 동일한 wi 할당
            for name in group:
                wi[name] = group_val
                
        return wi
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
            pr_temp[n] = self.global_pruning_percent * (beta * fi[n] + (1 - beta) * wi[n])
        return self._balance(pr_temp)

    # def _get_wi(self, p):
    #     wi = {}
    #     sum_inv = sum((1.0 / (abs(s) + EPS)) ** p for s in self.sensitivity_si.values())
    #     for n, s in self.sensitivity_si.items():
    #         wi[n] = ((1.0 / (abs(s) + EPS)) ** p) / (sum_inv + EPS)
    #     return wi

    def _apply_score(self, fi, wi):
        sum_fw = sum(fi[n] * wi[n] for n in fi.keys())
        pr_temp = {
            n: self.global_pruning_percent * fi[n] * (wi[n] / (sum_fw + EPS))
            for n in fi.keys()
        }
        return self._balance(pr_temp)

    def _balance(self, pr_dict):
        actual_sum = sum(pr_dict.values())
        if abs(actual_sum - self.global_pruning_percent) > EPS and actual_sum > 0:
            scale = self.global_pruning_percent / actual_sum
            return {n: r * scale for n, r in pr_dict.items()}
        return pr_dict
