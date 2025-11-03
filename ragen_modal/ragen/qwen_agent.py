import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import numpy as np

class QwenRAGENAgent(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B", device="cuda"):
        super().__init__()
        self.device = device
        
        print(f"加载Qwen Base模型: {model_name}")
        # 加载模型但不冻结 - 允许梯度计算
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Qwen智能体初始化完成")
    
    def forward(self, input_ids, attention_mask):
        """前向传播用于训练"""
        outputs = self.llm(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.logits, outputs.hidden_states[-1]
    
    def generate_webshop_response(self, observation, instruction):
        """生成WebShop任务的思考和动作 - 训练时使用"""
        # 简化的prompt
        prompt = f"""网页: {observation}
任务: {instruction}

请思考并行动:
<think>"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
        
        # 训练时使用模型生成
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        full_response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # 提取内容
        think_content = self._extract_between_tags(full_response, "think")
        action_content = self._extract_between_tags(full_response, "action")
        
        # 计算对数概率
        log_prob = self._calculate_log_prob(outputs, inputs.input_ids.size(1))
        
        # 如果没有有效动作，提供默认值
        if not action_content or not any(x in action_content for x in ['search[', 'click[', 'buy[']):
            action_content = "search[product]"
        if not think_content:
            think_content = "分析任务需求并搜索合适商品"
        
        return think_content, action_content, log_prob, full_response
    
    def _extract_between_tags(self, text, tag):
        """提取标签间的内容"""
        if not text:
            return ""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _calculate_log_prob(self, outputs, input_length):
        """计算生成序列的对数概率"""
        try:
            # 获取生成的token IDs（排除输入部分）
            generated_sequences = outputs.sequences[:, input_length:]
            scores = outputs.scores
            
            log_probs = []
            for i, score in enumerate(scores):
                # 计算每个位置的对数概率
                log_prob = torch.log_softmax(score, dim=-1)
                # 获取实际生成token的对数概率
                token_log_prob = log_prob[0, generated_sequences[0, i]]
                log_probs.append(token_log_prob)
            
            return torch.stack(log_probs).mean().item()
        except:
            return 0.0

class SimplePolicyHead(nn.Module):
    """简单的策略头用于动作选择"""
    def __init__(self, hidden_size, action_size=10):
        super().__init__()
        self.action_head = nn.Linear(hidden_size, action_size)
        
    def forward(self, hidden_states):
        return torch.softmax(self.action_head(hidden_states.mean(dim=1)), dim=-1)
