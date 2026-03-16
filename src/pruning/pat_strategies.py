import torch
from .base import BasePruner
from src.models import find_prunable_blocks
import torch.nn as nn
EPS = 1e-8

class PATPruner(BasePruner):
    def __init__(self, model, config, sensitivity_si, topology_groups=None):
        super().__init__(model, config)
        self.sensitivity_si = sensitivity_si  # 사전 계산된 민감도 {block_name: si}
        self.global_target = config['strategy']['channel_keep_ratio']
        self.topology_groups = topology_groups
        
        # 1. 자동 수집 시도
        model_name_raw = config['model']['name'].lower()
        search_name = 'resnet32' if ('resnet32' in model_name_raw or 'cifar' in model_name_raw) else config['model']['name']
        self.prunable_blocks = find_prunable_blocks(model, search_name)
        
        # 2. 자동 수집 실패 시 강제 수동 수집 (Aggressive Mode)
        if not self.prunable_blocks:
            print("🔎 [Debug] Automatic collection failed. Starting Aggressive Manual Search...")
            self.prunable_blocks = {}
            for n, m in model.named_modules():
                # nn.Conv2d이면서 ResNet 블록 이름(layer1, 2, 3)을 포함하는 모든 모듈 대상
                if isinstance(m, nn.Conv2d) and any(x in n for x in ['layer1', 'layer2', 'layer3']):
                    parts = n.split('.')
                    if len(parts) >= 2:
                        # 'layer3.4.conv1' -> 'layer3.4' 형식으로 블록 이름 생성
                        short_name = f"{parts[0]}.{parts[1]}"
                        # 해당 블록의 대표 모듈로 등록 (이미 등록된 경우 skip)
                        if short_name not in self.prunable_blocks:
                            self.prunable_blocks[short_name] = m
            
            # 수집된 키들을 정렬해서 출력하여 layer3.4가 있는지 확인용
            collected_keys = sorted(list(self.prunable_blocks.keys()))
            print(f"✅ Manually collected blocks: {collected_keys}")

        # 3. 그룹별 민감도(SI) 계산
        self.group_si = {}
        if self.topology_groups:
            for i, group in enumerate(self.topology_groups):
                # 그룹 내 레이어(긴 이름)들의 SI 값 추출
                group_si_values = [self.sensitivity_si.get(name, 0.0) for name in group]
                self.group_si[i] = sum(group_si_values) / len(group_si_values) if group_si_values else 0.0
        
        # 4. 파라미터 수 및 비중 계산을 위한 준비
        # self.prunable_blocks가 비어있지 않아야 KeyError가 안 납니다.
        self.param_counts = {
            name: sum(p.numel() for p in blk.parameters()) 
            for name, blk in self.prunable_blocks.items()
        }
        self.total_params = sum(self.param_counts.values())
        print(f"🚀 PATPruner Ready: {len(self.prunable_blocks)} blocks identified.")
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
    #             channel_keep_ratio = min(ratio / 100.0, 0.99) if ratio > 1 else ratio
                
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
    #             num_keep = max(1, int(num_channels * (1 - channel_keep_ratio)))
                
    #             # 중요도가 높은 순서대로 인덱스 추출
    #             keep_idx = importance.argsort(descending=True)[:num_keep].tolist()
    #             keep_indices_dict[name] = sorted(keep_idx)
                
    #     return keep_indices_dict

    def compute_all_keep_indices(self, round_idx=1):
        ratios = self.compute_all_ratios()
        keep_indices_dict = {}
        
        with torch.no_grad():
            for group_layers in self.topology_groups:
                # [수정] 현재 그룹에 속한 레이어 이름들(긴 이름)을 
                # 수집된 prunable_blocks의 키(짧은 이름)로 변환하여 매칭
                block = None
                rep_name = ""
                for name in group_layers:
                    # 'layer1.0.bn1' -> 'layer1.0'
                    short_name = ".".join(name.split('.')[:2])
                    if short_name in self.prunable_blocks:
                        block = self.prunable_blocks[short_name]
                        rep_name = short_name
                        break
                
                # 매칭되는 블록이 없으면 건너뜀
                if block is None: continue

                # 해당 블록의 프루닝 비율 계산
                ratio = ratios.get(rep_name, 0.0)
                current_prune_ratio = min(ratio / 100.0, 0.99)
                
                # 가중치 텐서에서 필터 중요도 계산
                if hasattr(block, 'conv2'): w = block.conv2.weight.data
                elif hasattr(block, 'conv1'): w = block.conv1.weight.data
                else: w = block.weight.data
                
                actual_channels = w.size(0)
                importance = w.view(actual_channels, -1).abs().sum(dim=1).cpu()
                num_keep = max(1, int(actual_channels * (1 - current_prune_ratio)))
                
                # 중요도 순으로 남길 인덱스 추출
                keep_idx = sorted(importance.argsort(descending=True)[:num_keep].tolist())
                
                # 그룹 내 모든 레이어(bn 등 포함)에 인덱스 전파
                for name in group_layers:
                    keep_indices_dict[name] = keep_idx
                    
        # [최종 핵심] Linear(FC) 레이어 차원 맞추기
        if keep_indices_dict:
            # 전체 레이어 중 가장 마지막에 등록된 (즉, layer3의 마지막) 인덱스를 사용
            # resnet32 구조상 layer3의 마지막 bn 출력이 linear의 입력이 됨
            last_key = list(keep_indices_dict.keys())[-1]
            keep_indices_dict['linear'] = keep_indices_dict[last_key]
            print(f"✅ Final Pruning Map: {len(keep_indices_dict)} layers matched.")
        else:
            print("❗ Critical Error: No layers were matched for pruning!")
                    
        return keep_indices_dict

    def _get_wi(self, p):
        """
        [Group-aware 수정 버전]
        모든 이름을 short_name(예: layer1.0)으로 통일하여 fi와 Key를 맞춤.
        """
        wi = {}
        # 1. 모든 그룹의 (1/SI)^p 합산
        sum_inv = sum((1.0 / (abs(s) + EPS)) ** p for s in self.group_si.values())
        
        # 2. 각 그룹별 비중 계산 후 short_name으로 배분
        for i, group in enumerate(self.topology_groups):
            s = self.group_si.get(i, 0.0)
            group_val = ((1.0 / (abs(s) + EPS)) ** p) / (sum_inv + EPS)
            
            for name in group:
                # 'layer1.0.bn1' -> 'layer1.0' 형식으로 변환하여 fi의 키와 일치시킴
                short_name = ".".join(name.split('.')[:2])
                # 해당 short_name이 prunable_blocks에 있을 때만 할당
                if short_name in self.param_counts:
                    wi[short_name] = group_val
                
        return wi
    def compute_all_ratios(self):
        """YAML 설정에 따른 전략 분기 및 최종 레이어별 프루닝 비율 반환"""
        st_type = self.config['strategy']['type']
        # st_type = "weighted_sum"

        print(f"🧠 PAT Strategy Type: {st_type}")
        
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

    # def _get_wi(self, p):
    #     wi = {}
    #     sum_inv = sum((1.0 / (abs(s) + EPS)) ** p for s in self.sensitivity_si.values())
    #     for n, s in self.sensitivity_si.items():
    #         wi[n] = ((1.0 / (abs(s) + EPS)) ** p) / (sum_inv + EPS)
    #     return wi

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