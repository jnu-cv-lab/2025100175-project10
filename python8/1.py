import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np

class PositionalEncodingIntegrator:
    def __init__(self, d_model=64, max_len=100):
        self.d_model = d_model
        self.max_len = max_len
        
        
        self.sinusoidal_pe = self._generate_sinusoidal()
        
        
        self.rope_cos, self.rope_sin = self._generate_rope()

    def _generate_sinusoidal(self):
        """生成传统的 Sinusoidal Positional Encoding"""
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) 

    def _generate_rope(self):
        """预计算 RoPE 所需的 cos 和 sin 张量"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_model, 2).float() / self.d_model))
        t = torch.arange(self.max_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq) 
        return freqs.cos().unsqueeze(0), freqs.sin().unsqueeze(0) 

 
    def apply_sinusoidal(self, x, pos):
        """
        x: [batch, seq_len, d_model]
        pos: 起始位置索引
        """
        seq_len = x.size(1)
       
        return x + self.sinusoidal_pe[:, pos:pos+seq_len]

    
    def apply_rotary_pos_emb(self, x, cos, sin):
        """
        x: [batch, seq_len, d_model]
        cos/sin: [1, seq_len, d_model//2]
        """
      
        x1, x2 = x.chunk(2, dim=-1)
        
      
        return torch.cat((-x2, x1), dim=-1) * sin + x * cos

    
    def verify_relative_property(self):
        print("🚀 开始验证 RoPE 的相对位置性质...")
        print("原理：若 m-n = i-j，则 <q_m, k_n> 应该约等于 <q_i, k_j>\n")
        
        
        d_model = self.d_model
        q_vec = torch.randn(1, 1, d_model)
        k_vec = torch.randn(1, 1, d_model)
        
      
        pos_q_a, pos_k_a = 5, 10
        cos_a_q = self.rope_cos[:, pos_q_a:pos_q_a+1]
        sin_a_q = self.rope_sin[:, pos_q_a:pos_q_a+1]
        cos_a_k = self.rope_cos[:, pos_k_a:pos_k_a+1]
        sin_a_k = self.rope_sin[:, pos_k_a:pos_k_a+1]
        
        q_a = self.apply_rotary_pos_emb(q_vec, cos_a_q, sin_a_q)
        k_a = self.apply_rotary_pos_emb(k_vec, cos_a_k, sin_a_k)
        dot_a = torch.sum(q_a * k_a).item()
        
       
        pos_q_b, pos_k_b = 20, 25
        cos_b_q = self.rope_cos[:, pos_q_b:pos_q_b+1]
        sin_b_q = self.rope_sin[:, pos_q_b:pos_q_b+1]
        cos_b_k = self.rope_cos[:, pos_k_b:pos_k_b+1]
        sin_b_k = self.rope_sin[:, pos_k_b:pos_k_b+1]
        
        q_b = self.apply_rotary_pos_emb(q_vec, cos_b_q, sin_b_q)
        k_b = self.apply_rotary_pos_emb(k_vec, cos_b_k, sin_b_k)
        dot_b = torch.sum(q_b * k_b).item()
        
        diff = abs(dot_a - dot_b)
        
        print(f"实验组 1 (位置 {pos_q_a} vs {pos_k_a}): 点积 = {dot_a:.4f}")
        print(f"实验组 2 (位置 {pos_q_b} vs {pos_k_b}): 点积 = {dot_b:.4f}")
        print(f"差异值: {diff:.6f}")
        
        if diff < 1e-5:
            print("✅ 验证通过！点积结果几乎一致，证明 RoPE 具有完美的相对位置编码能力。")
        else:
            print("❌ 验证失败，请检查代码逻辑。")

   
    def visualize_encoding(self):
        plt.figure(figsize=(12, 5))

        
        plt.subplot(1, 2, 1)
        plt.imshow(self.sinusoidal_pe[0, :50].numpy(), cmap='viridis', aspect='auto')
        plt.title('Sinusoidal Positional Encoding (E + pos)')
        plt.xlabel('Embedding Dim')
        plt.ylabel('Position')
        plt.colorbar()

       
        plt.subplot(1, 2, 2)
        plt.imshow(self.rope_cos[0, :50, :32].numpy(), cmap='viridis', aspect='auto') 
        plt.title('RoPE Cos Table (Rotary Base)')
        plt.xlabel('Embedding Dim (Half)')
        plt.ylabel('Position')
        plt.colorbar()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
   
    encoder = PositionalEncodingIntegrator(d_model=64, max_len=100)
    
    encoder.verify_relative_property()
    
   
    encoder.visualize_encoding()