import torch
import torch.nn as nn
from .quantizer import Quantizer

class GPTQ:
    def __init__(self, model, layer_names, device):
        self.model = model
        self.layer_names = layer_names
        self.device = device
        self.quantized_weights = {}
        
        # 层类型关键词定义
        self.attention_keywords = {"q_proj", "k_proj", "v_proj", "o_proj"}
        self.ffn_keywords = {"gate_proj", "up_proj", "down_proj"}

    def _get_layer_type(self, layer_name):
        """识别层类型（注意力层/前馈层）"""
        for kw in self.attention_keywords:
            if kw in layer_name:
                return "attention"
        for kw in self.ffn_keywords:
            if kw in layer_name:
                return "ffn"
        return "default"

    def fasterquant(self, layer_name, W, block_size=128, damp=0.01, group_size=-1, act_order=True):
        """动态精度量化函数"""
        layer_type = self._get_layer_type(layer_name)
        W = W.clone().to(self.device)
        if W.ndim == 4:  # 处理卷积层权重（如存在）
            W = W.flatten(1)

        # 根据层类型配置量化精度
        if layer_type == "attention":
            bits = 4  # 注意力层4bit量化
        elif layer_type == "ffn":
            bits = 2  # 前馈层2bit量化
        else:
            bits = 4  # 默认4bit

        # 初始化量化器
        quantizer = Quantizer(
            bits=bits,
            perchannel=True,
            sym=False,
            mse=False
        )

        # 计算海森矩阵（误差建模）
        H = torch.zeros(W.shape[1], W.shape[1], device=self.device)
        nsamples = 0

        def add_batch(x):
            nonlocal nsamples, H
            x = x.to(self.device)
            x = x.float()
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            x = x.reshape(-1, x.shape[-1])
            n = x.shape[0]
            h = x.t() @ x
            H *= nsamples / (nsamples + n)
            nsamples += n
            H += h / nsamples

        # 注册前向钩子收集激活值计算海森矩阵
        handles = []
        for name, module in self.model.named_modules():
            if layer_name in name and "weight" in name:
                handles.append(module.register_forward_hook(
                    lambda m, i, o: add_batch(o[0])
                ))

        # 执行校准数据推理（需外部传入校准数据加载逻辑）
        self.model(*self.calib_data)  # 假设calib_data已在外部设置
        for h in handles:
            h.remove()

        # 海森矩阵处理（乔列斯基分解）
        H += damp * torch.eye(H.shape[0], device=self.device)
        H = torch.linalg.cholesky(H)

        # 分组建模与量化
        Q = torch.zeros_like(W)
        scale = torch.zeros(W.shape[0], device=self.device)
        zero = torch.zeros(W.shape[0], device=self.device)

        for i in range(0, W.shape[0], block_size):
            end = min(i + block_size, W.shape[0])
            w = W[i:end, :]
            # 基于海森矩阵的误差补偿量化
            q, s, z = quantizer.quantize(w, H)
            Q[i:end, :] = q
            scale[i:end] = s
            zero[i:end] = z

        return Q.reshape(W.shape), scale, zero

    def quantize_layers(self, calib_data):
        """量化所有指定层"""
        self.calib_data = calib_data  # 保存校准数据
        for name in self.layer_names:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                if hasattr(layer, "weight"):
                    W = layer.weight.data
                    q_weight, scale, zero = self.fasterquant(name, W)
                    self.quantized_weights[name] = (q_weight, scale, zero)
                    # 更新模型权重为量化后权重
                    layer.weight.data = q_weight
        return self.model
