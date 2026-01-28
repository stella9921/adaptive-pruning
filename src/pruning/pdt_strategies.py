import torch
import torch.nn as nn
from .base import BasePruner
from .engine.hessian_free import SNOWSEngine
import numpy as np
from .optimizer import lagrangian_optimization 

class PDTPruner(BasePruner):
    def __init__(self, model, config, topology_groups=None):
        super().__init__(model, config)
        self.ema_decay = config['strategy'].get('ema_decay', 0.95)
        self.lambda_h = config['strategy'].get('lambda_h', 0.5) 
        self.engine = SNOWSEngine(n_iter=config['strategy'].get('hessian_iter', 5))
        
        self.topology_groups = topology_groups
        self.layers_dict = nn.ModuleDict()
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.layers_dict[name.replace('.', '_')] = m
        
        self.layers = list(self.layers_dict.values())
        self._check_buffers()

    def _check_buffers(self):
        for m in self.layers:
            n_f = m.weight.shape[0]
            if not hasattr(m, 'mask'):
                m.register_buffer("mask", torch.ones(n_f, device=self.device))
            if not hasattr(m, 'grad_ema'):
                m.register_buffer("grad_ema", torch.zeros(n_f, device=self.device))
            if not hasattr(m, 'hessian_score'):
                m.register_buffer("hessian_score", torch.zeros(n_f, device=self.device))

    def update_ema_and_mask_grad(self):
        with torch.no_grad():
            for m in self.layers:
                if hasattr(m, 'weight') and m.weight.grad is not None:
                    # 마스크 적용하여 그래디언트 마스킹
                    m_view = m.mask.view(-1, 1, 1, 1) if m.weight.dim()==4 else m.mask.view(-1, 1)
                    m.weight.grad.mul_(m_view)
                    g = m.weight.grad.pow(2).reshape(m.weight.shape[0], -1).mean(1)
                    m.grad_ema.mul_(self.ema_decay).add_(g, alpha=1 - self.ema_decay)

    def apply_mask_to_weights(self, optimizer=None):
        with torch.no_grad():
            for m in self.layers:
                mask = m.mask
                m_view = mask.view(-1, 1, 1, 1) if m.weight.dim() == 4 else mask.view(-1, 1)
                m.weight.data.mul_(m_view)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.mul_(mask)
                if optimizer is not None:
                    for p in [m.weight, m.bias]:
                        if p is not None and p in optimizer.state:
                            state = optimizer.state[p]
                            if "momentum_buffer" in state:
                                # Blackwell 대응: 브로드캐스팅 뷰 적용
                                mask_view = mask.view(-1, *(1 for _ in range(p.dim() - 1)))
                                state["momentum_buffer"].mul_(mask_view)

    def step_pruning(self, loss, memory_budget=None):
        """
        [Conditional Pruning Logic]
        1. 위상 하위 그룹 선별 (Target Groups)
        2. 타겟 그룹 내 유닛들만 헤시안 계산 및 경쟁
        3. 상위 그룹(Protected)은 100% 생존 보장
        """
        all_modules = dict(self.model.named_modules())
        def find_layer(name):
            if name in all_modules: return all_modules[name]
            return all_modules.get(name.replace('_', '.'))

        # 1. 위상 정보 수집
        group_info_list = []
        processed_layers = set()

        if self.topology_groups:
            for group in self.topology_groups:
                layers = [find_layer(ln) for ln in group if isinstance(find_layer(ln), (nn.Conv2d, nn.Linear))]
                if not layers: continue
                for m in layers: processed_layers.add(m)
                w_g = torch.mean(torch.stack([m.grad_ema.mean() for m in layers])).item()
                group_info_list.append({'layers': layers, 'w_g': w_g, 'is_group': True})

        for m in self.layers:
            if m not in processed_layers:
                group_info_list.append({'layers': [m], 'w_g': m.grad_ema.mean().item(), 'is_group': False})

        # 2. 하위 그룹 선별
        group_info_list.sort(key=lambda x: x['w_g'])
        h_target_ratio = self.config['strategy'].get('hessian_target_ratio', 0.5)
        num_targets = int(len(group_info_list) * h_target_ratio)
        
        target_groups = group_info_list[:num_targets]
        protected_groups = group_info_list[num_targets:]

        # 3. 타겟 그룹만 헤시안 계산
        target_params = []
        for g in target_groups:
            for m in g['layers']:
                target_params.append(m.weight)

        K = self.config['strategy'].get('k_horizon', 10)
        hv_list = self.engine.get_k_step_hessian_selective(loss, target_params, K)

        # 4. 타겟 유닛들만 점수 리스트 구성 (경쟁 대상)
        target_unit_scores = []
        target_unit_costs = []
        target_unit_metadata = []

        hv_idx = 0
        for g in target_groups:
            for m in g['layers']:
                hv = hv_list[hv_idx]
                h_energy = hv.pow(2).reshape(hv.shape[0], -1).mean(1)
                if h_energy.max() > 0:
                    h_energy /= (h_energy.max() + 1e-8)
                m.hessian_score.copy_(h_energy)
                hv_idx += 1

            num_channels = g['layers'][0].weight.shape[0]
            for i in range(num_channels):
                s_gc = sum(m.hessian_score[i].item() for m in g['layers'])
                # 점수 = 그룹 위상(W_g) * 헤시안 기여도(s_gc)
                target_unit_scores.append(g['w_g'] * s_gc) 
                # 비용 = 유닛당 파라미터 수
                target_unit_costs.append(sum(m.weight.nelement()/m.weight.shape[0] for m in g['layers']))
                target_unit_metadata.append((g['layers'], i))

        # 5. 타겟 그룹 내부 최적화 (보호 그룹 제외)
        scores_np = np.array(target_unit_scores)
        costs_np = np.array(target_unit_costs)

        if memory_budget is None:
            # 설정된 target_ratio를 '타겟 그룹 내 삭제 비율'로 적용
            t_ratio = self.config['strategy'].get('target_ratio', 0.5)
            # 예산: 타겟 그룹의 총 용량 중 t_ratio) 만큼 남기기
            memory_budget = np.sum(costs_np) * (t_ratio)

        # 타겟 유닛들끼리만 서바이벌 게임
        optimal_mask_flags = lagrangian_optimization(scores_np, costs_np, memory_budget)

        # 6. 마스크 최종 적용
        with torch.no_grad():
            # [A] 보호 그룹: 전원 생존 (1.0)
            for g in protected_groups:
                for m in g['layers']:
                    m.mask.fill_(1.0)
            
            # [B] 타겟 그룹: 최적화 결과 반영
            active_count = 0
            for idx, is_alive in enumerate(optimal_mask_flags):
                layers_list, channel_idx = target_unit_metadata[idx]
                for layer_obj in layers_list:
                    layer_obj.mask[channel_idx] = 1.0 if is_alive else 0.0
                if is_alive: active_count += 1

        print(f"[PDT] Conditional Pruning Done.")
        print(f"[*] Target Groups: {num_targets} | Protected: {len(protected_groups)}")
        print(f"[*] Target Units Result: {active_count}/{len(target_unit_scores)} alive")
        torch.cuda.empty_cache()

    def get_current_sparsity(self):
        total_params = 0
        active_params = 0
        for m in self.model.modules():
            if hasattr(m, "mask"):
                total_params += m.mask.numel()
                active_params += m.mask.sum().item()
        if total_params == 0: return 0.0
        return (1.0 - (active_params / total_params)) * 100.0