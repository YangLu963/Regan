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
        # 加载Base Model用于文本生成
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 冻结LLM参数，只用于推理
        for param in self.llm.parameters():
            param.requires_grad = False
            
        print("Qwen智能体初始化完成")
    
    def generate_webshop_response(self, observation, instruction):
        """生成WebShop任务的思考和动作"""
        prompt = f"""基于网页内容和任务要求，请先思考再行动。

网页内容: {observation}
任务: {instruction}

请按以下格式响应:
<think>
你的推理过程...
</think>
<action>
你的动作（search[关键词], click[ID], 或 buy[ID]）
</action>

请开始:
<think>
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        full_response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # 提取思考和动作
        think_content = self._extract_between_tags(full_response, "think")
        action_content = self._extract_between_tags(full_response, "action")
        
        # 计算对数概率
        log_prob = self._calculate_response_log_prob(full_response, inputs)
        
        return think_content, action_content, log_prob, full_response
    
    def _extract_between_tags(self, text, tag):
        """提取标签间的内容"""
        if not text:
            return ""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _calculate_response_log_prob(self, response, inputs):
        """计算生成响应的对数概率"""
        try:
            # 编码完整响应
            response_tokens = self.tokenizer.encode(response, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # 计算每个位置的logits
                outputs = self.llm(response_tokens, labels=response_tokens)
                # 使用交叉熵损失计算平均对数概率
                log_prob = -outputs.loss * response_tokens.size(1)
                
            return log_prob.item()
        except Exception as e:
            print(f"对数概率计算错误: {e}")
            return 0.0
    
    def get_text_embedding(self, text):
        """获取文本的嵌入表示（用于缓存键）"""
        if not text:
            return torch.zeros(512)
            
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.llm(**inputs, output_hidden_states=True)
                # 使用最后一层隐藏状态的均值
                embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
                return embedding.cpu()
                
        except Exception as e:
            print(f"文本嵌入错误: {e}")
            return torch.zeros(512)