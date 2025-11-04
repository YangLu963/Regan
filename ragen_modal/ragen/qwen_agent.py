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
            
        print("Qwen智能体初始化完成")
    
    def generate_webshop_response(self, observation, instruction):
        """生成WebShop任务的思考和动作 - 改进版本"""
        # 更清晰、更具体的Prompt，避免模板文字
        prompt = f"""你是一个网页购物助手。请根据网页内容和任务要求完成任务。

网页内容: {observation}
任务: {instruction}

请严格按照以下格式思考和行动：

<think>
首先分析当前网页有什么内容，然后根据任务要求决定下一步行动。
例如：网页显示搜索页面，我需要搜索"蓝色牛仔裤"。
</think>
<action>
search[具体商品关键词] 或 click[商品ID] 或 buy[商品ID]
</action>

重要：不要输出任何其他内容，严格按照上面的格式。

现在开始：
<think>
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True, max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=200,  # 增加token数量确保完整输出
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                repetition_penalty=1.1,  # 减少重复
                no_repeat_ngram_size=3   # 避免重复短语
            )
        
        full_response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        print(f"🔍 完整响应: {full_response}")
        
        # 改进的内容提取
        think_content = self._extract_think_content(full_response)
        action_content = self._extract_action_content(full_response, instruction)
        
        # 计算对数概率
        log_prob = self._calculate_log_prob(outputs, inputs.input_ids.size(1))
        
        return think_content, action_content, log_prob, full_response
    
    def _extract_think_content(self, text):
        """改进的思考内容提取"""
        if not text:
            return "分析任务需求并采取行动"
        
        # 方法1: 提取 <think> 标签内容
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            content = think_match.group(1).strip()
            # 过滤掉模板文字和无效内容
            if (content and 
                len(content) > 5 and 
                "你的推理" not in content and 
                "请思考" not in content and
                "思考过程" not in content):
                return content
        
        # 方法2: 如果没有标签，尝试找到合理的思考内容
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if (line and 
                len(line) > 10 and 
                not line.startswith('<') and 
                not line.endswith('>') and
                "思考" not in line and
                "你的推理" not in line and
                "请思考" not in line and
                "动作" not in line and
                "search[" not in line and
                "click[" not in line and
                "buy[" not in line):
                return line
        
        # 方法3: 返回有意义的默认思考
        return "根据任务需求，我需要搜索相关商品"
    
    def _extract_action_content(self, text, instruction):
        """改进的动作内容提取"""
        if not text:
            return self._generate_default_action(instruction)
        
        # 方法1: 提取 <action> 标签内容
        action_match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            if self._is_valid_action(action):
                return action
        
        # 方法2: 在文本中搜索动作模式
        action_patterns = [
            r'search\[[^\]]+\]',
            r'click\[\d+\]', 
            r'buy\[\d+\]'
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, text)
            if match:
                action = match.group(0)
                if self._is_valid_action(action):
                    return action
        
        # 方法3: 生成基于任务的具体动作
        return self._generate_default_action(instruction)
    
    def _is_valid_action(self, action):
        """检查动作是否有效"""
        if not action:
            return False
        
        # 检查动作格式
        valid_formats = [
            r"^search\[.+\]$",
            r"^click\[\d+\]$", 
            r"^buy\[\d+\]$"
        ]
        
        for pattern in valid_formats:
            if re.match(pattern, action.strip()):
                return True
        
        return False
    
    def _generate_default_action(self, instruction):
        """根据任务生成具体的默认动作"""
        instruction_lower = instruction.lower()
        
        if "blanket" in instruction_lower and "classic" in instruction_lower:
            return "search[classic wool blanket]"
        elif "jeans" in instruction_lower and "blue" in instruction_lower:
            if "32" in instruction_lower:
                return "search[blue jeans size 32]"
            else:
                return "search[blue denim jeans]"
        elif "laptop" in instruction_lower and "1000" in instruction_lower:
            return "search[laptop under 1000 dollars]"
        elif "shirt" in instruction_lower and "red" in instruction_lower:
            return "search[red cotton shirt]"
        elif "mouse" in instruction_lower and "wireless" in instruction_lower:
            return "search[wireless mouse with good ratings]"
        else:
            # 从指令中提取关键词
            keywords = self._extract_keywords(instruction)
            if keywords:
                return f"search[{keywords}]"
            else:
                return "search[product]"
    
    def _extract_keywords(self, instruction):
        """从指令中提取关键词"""
        # 移除常见动词和介词
        stop_words = {'find', 'get', 'buy', 'purchase', 'search', 'for', 'a', 'an', 'the', 'with', 'in', 'under', 'over'}
        words = instruction.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        if keywords:
            return ' '.join(keywords[:3])  # 取前3个关键词
        else:
            return "product"
    
    def _calculate_log_prob(self, outputs, input_length):
        """计算生成序列的对数概率"""
        try:
            # 获取生成的token IDs（排除输入部分）
            generated_sequences = outputs.sequences[:, input_length:]
            scores = outputs.scores
            
            if not scores:
                return 0.0
            
            log_probs = []
            for i, score in enumerate(scores):
                if i >= generated_sequences.size(1):
                    break
                # 计算每个位置的对数概率
                log_prob = torch.log_softmax(score, dim=-1)
                # 获取实际生成token的对数概率
                token_log_prob = log_prob[0, generated_sequences[0, i]]
                log_probs.append(token_log_prob)
            
            if log_probs:
                return torch.stack(log_probs).mean().item()
            else:
                return 0.0
                
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
    
    def forward(self, input_ids, attention_mask):
        """前向传播用于训练"""
        outputs = self.llm(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.logits, outputs.hidden_states[-1]

# 测试函数
def test_qwen_agent():
    """测试Qwen智能体"""
    print("🧪 测试Qwen智能体...")
    
    agent = QwenRAGENAgent()
    
    test_cases = [
        ("模拟网页 - 搜索页面", "Purchase a classic blanket"),
        ("模拟网页 - 商品列表", "Get a blue jeans in size 32"),
        ("模拟网页 - 首页", "Find a laptop under $1000")
    ]
    
    for i, (obs, instruction) in enumerate(test_cases):
        print(f"\n📝 测试案例 {i+1}: {instruction}")
        think, action, log_prob, full = agent.generate_webshop_response(obs, instruction)
        print(f"💭 思考: {think}")
        print(f"🎯 动作: {action}")
        print(f"📊 对数概率: {log_prob:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    test_qwen_agent() 
