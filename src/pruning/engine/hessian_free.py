import torch
import gc

class SNOWSEngine:
    def __init__(self, n_iter=3, tolerance=1e-5):
        self.n_iter = n_iter 
        self.tolerance = tolerance 

    def get_k_step_hessian_selective(self, loss, target_params, K_horizon=5):
        if not target_params:
            return []

        final_hv_list = []
        num_params = len(target_params)
        
        # 1. 메모리 사전 청소
        torch.cuda.empty_cache()
        gc.collect()

        # 2. 파라미터별 순차 Hessian 계산
        for i, param in enumerate(target_params):
            # 마지막 파라미터가 아니면 연산 그래프를 유지(True)해야 다음 레이어 미분이 가능함
            is_last = (i == num_params - 1)
            
            try:
                # 1차 그래디언트 (여기서 retain_graph=True는 필수)
                grad = torch.autograd.grad(
                    loss, param, 
                    create_graph=True, 
                    retain_graph=True
                )[0]
                
                v = torch.randn_like(grad)
                dot_product = (grad * v).sum()
                
                # 2차 미분 (HVP)
                # [수정 핵심] 마지막 레이어 전까지는 loss 그래프를 살려둬야 함
                hv = torch.autograd.grad(
                    dot_product, param, 
                    retain_graph=not is_last 
                )[0]
                
                final_hv_list.append(hv.detach().clone())
                
                # 메모리 정리
                del grad, v, dot_product, hv
                if i % 5 == 0: # 5개 레이어마다 캐시 비우기
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                raise e

        del loss # loss 자체도 이제 필요 없으니 삭제
        gc.collect()
        torch.cuda.empty_cache()

        return final_hv_list

    def get_channel_scores(self, hv_list, params_with_name):
        scores = {}
        for (name, param), hv in zip(params_with_name, hv_list):
            if hv.dim() >= 2: 
                energy = torch.norm(hv, p=2, dim=0)**2 
                scores[name] = energy
            else:
                scores[name] = hv**2
        return scores