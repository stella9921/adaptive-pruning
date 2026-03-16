import torch
import gc

class SNOWSEngine:
    def __init__(self, n_iter=3, tolerance=1e-5):
        self.n_iter = n_iter 
        self.tolerance = tolerance 

    def get_k_step_hessian_selective(self, loss, target_params, K_horizon=5):
        if not target_params:
            return []

        # ViT의 flash/mem-efficient attention은 2차 미분 미지원 → math 모드로 fallback
        flash_was_enabled = torch.backends.cuda.flash_sdp_enabled()
        mem_eff_was_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        final_hv_list = []
        num_params = len(target_params)
        torch.cuda.empty_cache()
        gc.collect()

        for i, param in enumerate(target_params):
            is_last = (i == num_params - 1)
            try:
                grad = torch.autograd.grad(
                    loss, param,
                    create_graph=True,
                    retain_graph=True
                )[0]

                v = torch.randn_like(grad)
                dot_product = (grad * v).sum()

                hv = torch.autograd.grad(
                    dot_product, param,
                    retain_graph=not is_last
                )[0]

                final_hv_list.append(hv.detach().clone())

                del grad, v, dot_product, hv
                if i % 5 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                raise e

        # 원래 설정 복원
        torch.backends.cuda.enable_flash_sdp(flash_was_enabled)
        torch.backends.cuda.enable_mem_efficient_sdp(mem_eff_was_enabled)

        del loss
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