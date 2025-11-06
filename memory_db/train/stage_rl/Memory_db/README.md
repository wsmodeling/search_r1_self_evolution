# Memory Database System for Reason-RFT

## 使用方法

- 这里, 我们先用update_db.py中的VLMInferenceManager类来处理推理和数据库更新。
- 然后, 我们就建立好了初步的数据库. database.py里面 get_llm_responses_sorted_by_average_score 可以将选定的样本按平均分排序, 我们就可以做后续的操作了, 比如选取正负例子和添加reflexion(change_reflexion)

## 概述

Memory Database是Reason-RFT项目中的一个关键组件，专门用于管理强化学习阶段的训练数据和VLM（视觉语言模型）响应。该系统结合了两种记忆机制：**短期记忆（STM）**和**长期记忆（LTM）**，以及一个基于JSON的数据库，用于存储和管理训练样本、模型响应及其评分。

## 系统架构

```
Memory_db/
├── database.py          # JSON数据库核心组件
├── update_db.py         # VLM推理和数据库更新管理器
├── vlm_agent.py         # VLM代理和提示模板
└── README.md           # 文档说明（本文件）
```

### 关联文件
- `../memory.py` - 记忆管理器，实现STM/LTM机制

## 核心组件

### 1. JSONDatabase (`database.py`)

基于JSON文件的轻量级数据库，专门用于存储训练数据和LLM响应。

#### 数据结构
```json
{
  "training_data": {
    "trance-001": {
      "id": "trance-001",
      "problem": "问题描述",
      "answer": "正确答案",
      "round": 1,
      "image": [...],
      "llm_answers_and_score": [
        {
          "ans": "模型响应",
          "reflexion": "反思过程",
          "accuracy": 0.85,
          "format": 0.90,
          "reason": 0.75,
          "length": 0.80
        }
      ]
    }
  }
}
```

#### 主要功能
- **数据插入/更新**: `insert_training_data()`
- **响应添加**: `add_llm_response()`
- **数据检索**: `get_training_data()`, `get_all_training_data()`
- **按轮次筛选**: `get_training_data_by_round()`
- **搜索功能**: `search_by_problem_text()`
- **统计信息**: `get_statistics()`
- **排序功能**: `get_llm_responses_sorted_by_average_score()`
- **数据导出**: `export_to_json()`

#### 特性
- 🔒 **线程安全**: 使用锁机制确保并发安全
- 💾 **原子操作**: 通过临时文件确保数据完整性
- 📊 **评分系统**: 支持多维度评分（准确性、格式、推理、长度）
- 🔍 **高效检索**: 支持按ID、轮次、问题文本检索

### 2. VLMInferenceManager (`update_db.py`)

管理VLM推理过程和数据库更新的核心管理器。

#### 核心功能

##### 多次推理
```python
def multiple_inference(sample, image_dir, k=5, temperature=0.7, max_tokens=768)
```
- 对同一个样本执行k次推理
- 支持温度采样控制多样性
- 错误处理和恢复机制

##### 奖励计算
```python
def calculate_rewards(responses, ground_truth, current_step=0)
```
支持四种奖励函数：
- **准确性奖励**: `accuracy_reward` / `math_accuracy_reward` / `func_accuracy_reward`
- **格式奖励**: `format_reward` / `caption_format_reward`
- **推理奖励**: `reasoning_steps_reward`
- **长度奖励**: `len_reward`

##### 批处理
```python
def process_batch(training_data_list, image_dir, k=5, ...)
```
- 支持批量处理训练样本
- 进度跟踪和错误报告
- 结果统计和汇总

#### 工作流程
1. **数据预处理**: 加载训练数据和图像
2. **多次推理**: 使用VLM代理进行k次推理
3. **奖励计算**: 计算多维度奖励分数
4. **数据存储**: 将结果存储到数据库
5. **结果汇总**: 生成处理报告

### 3. VLM Agent (`vlm_agent.py`)

提供VLM推理能力和任务特定的提示模板。

#### 支持的任务类型
- **空间变换**: `trance`, `trance-left`, `trance-right`
- **数学推理**: `clevr-math`, `super-clevr`
- **几何推理**: `geometry3k`, `geoqa`
- **结构感知**: `structure-perception`

#### 评估类型
- **CoT-SFT**: Chain-of-Thought with Supervised Fine-tuning
- **Caption-CoT**: Caption-based Chain-of-Thought

#### 提示模板示例
```python
COT_TRANCE_QUESTION_PROMPT = '''
Your need to complete the spatial visual reasoning task...
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
'''
```

### 4. MemoryManager (`../memory.py`)

实现双重记忆机制的核心组件。

#### 短期记忆（STM）
- 使用`deque`存储最近几批的反馈文本
- 直接拼接，无需检索
- 快速访问最新反馈

#### 长期记忆（LTM）
- 存储"经验"（文本+嵌入向量）
- 基于余弦相似度的语义检索
- 支持经验累积和长期学习

#### 主要方法
```python
# STM操作
add_stm_feedback(feedback_text)
get_stm_context()

# LTM操作  
add_ltm_experience(experience_text, meta)
retrieve_ltm(query, k=3)
get_ltm_context(query, k=3)

# 统一接口
build_memory_prefix(query, k=3)
```

## 使用指南

### 1. 基础数据库操作

```python
from database import JSONDatabase

# 初始化数据库
db = JSONDatabase("my_memory_db.json")

# 插入训练数据
training_data = {
    'id': 'sample-001',
    'problem': '问题描述',
    'answer': '正确答案',
    'round': 1,
    'image': [{'path': 'image1.png', 'type': 'Spatial-Transformation'}]
}
db.insert_training_data(training_data)

# 添加LLM响应
scores = {'accuracy': 0.85, 'format': 0.90, 'reason': 0.75, 'length': 0.80}
db.add_llm_response('sample-001', 'LLM的响应', scores)

# 获取数据
data = db.get_training_data('sample-001')
stats = db.get_statistics()
```

### 2. VLM推理和更新

```python
from update_db import VLMInferenceManager

# 初始化管理器
manager = VLMInferenceManager(
    model_name_or_path="/path/to/model",
    db_path="memory_db.json",
    task_name="trance",
    eval_type="cot-sft"
)

# 处理单个样本
success = manager.process_and_store(
    training_data=sample_data,
    image_dir="/path/to/images",
    k=5,  # 执行5次推理
    temperature=0.7,
    round=1
)

# 批处理
results = manager.process_batch(
    training_data_list=samples,
    image_dir="/path/to/images",
    k=5,
    round=2
)
```

### 3. 命令行使用

```bash
# 处理JSON文件中的训练数据
python update_db.py \
    --model_path /path/to/qwen_model \
    --image_dir /path/to/images \
    --data_json training_data.json \
    --k 5 \
    --temperature 0.7 \
    --round 2

# 使用测试样本
python update_db.py \
    --model_path /path/to/qwen_model \
    --image_dir /path/to/images \
    --test_sample \
    --round 1
```

### 4. 记忆系统集成

```python
from memory import MemoryManager

# 初始化记忆管理器
memory = MemoryManager(stm_max_batches=3, ltm_max_items=1000)
memory.set_embedder(tokenizer, model, device="cuda")

# 添加反馈和经验
memory.add_stm_feedback("最近的训练反馈")
memory.add_ltm_experience("重要的训练经验", {"type": "error_pattern"})

# 构建记忆前缀
memory_prefix = memory.build_memory_prefix("当前查询", k=3)
```

## 配置选项

### 数据库配置
- `db_path`: 数据库文件路径
- `max_batches`: STM最大批次数
- `max_items`: LTM最大项目数

### 推理配置
- `k`: 每个样本的推理次数
- `temperature`: 采样温度
- `max_tokens`: 最大token数
- `max_image_num`: 最大图像数量

### 任务配置
- `task_name`: 任务类型 (`trance`, `clevr-math`, 等)
- `eval_type`: 评估类型 (`cot-sft`, `caption-cot`)

## 评分系统

系统支持四个维度的评分：

1. **准确性 (Accuracy)**: 答案的正确性
2. **格式 (Format)**: 输出格式的规范性
3. **推理 (Reason)**: 推理过程的质量
4. **长度 (Length)**: 回答长度的适当性

每个维度的分数范围为0.0-1.0，系统自动计算平均分数并支持按分数排序。

## 统计信息

数据库提供丰富的统计信息：

```python
stats = db.get_statistics()
# 返回:
# {
#   'total_training_entries': 100,
#   'total_llm_responses': 500,
#   'round_distribution': {1: 60, 2: 40},
#   'average_scores': {
#     'avg_accuracy': 0.75,
#     'avg_format': 0.85,
#     'avg_reason': 0.70,
#     'avg_length': 0.80
#   }
# }
```

## 性能优化

### 内存管理
- LTM向量存储在CPU上节省显存
- 支持向量L2归一化加速相似度计算
- deque提供高效的FIFO操作

### 并发安全
- 数据库操作使用线程锁
- 原子文件写入避免数据损坏
- 异常处理确保系统稳定性

### 扩展性
- 模块化设计支持新任务类型
- 插件式奖励函数架构
- 灵活的配置系统

## 故障排除

### 常见问题

1. **数据库损坏**
   - 系统会自动创建备份文件
   - 支持从损坏的JSON恢复

2. **内存不足**
   - 调整LTM最大项目数
   - 使用CPU存储向量

3. **推理失败**
   - 检查模型路径和权限
   - 验证图像文件完整性

### 日志和调试
- 详细的进度跟踪
- 错误信息和堆栈跟踪
- 性能统计和时间测量

## 扩展开发

### 添加新任务类型
1. 在`vlm_agent.py`中添加提示模板
2. 在`update_db.py`中配置奖励函数
3. 更新任务映射关系

### 自定义奖励函数
```python
def custom_reward_function(responses, solutions=None, step=0):
    """自定义奖励函数"""
    scores = []
    for response in responses:
        # 实现自定义评分逻辑
        score = calculate_custom_score(response)
        scores.append(score)
    return scores
```

### 数据库扩展
- 支持新的数据字段
- 添加索引和查询优化
- 实现数据迁移机制

## 许可证

本项目遵循相应的开源许可证。详细信息请参考项目根目录的LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request来改进这个系统。请确保：
- 代码符合项目规范
- 包含适当的测试
- 更新相关文档

## 联系信息

如有问题或建议，请通过项目的GitHub Issues联系我们。
