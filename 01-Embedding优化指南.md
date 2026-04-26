# Embedding优化指南

## 目录

1. [概述](#概述)
2. [模型选择策略](#模型选择策略)
3. [领域适配微调](#领域适配微调)
4. [工程优化技巧](#工程优化技巧)
5. [多维度Embedding](#多维度embedding)
6. [评估与监控](#评估与监控)
7. [最佳实践](#最佳实践)

---

## 概述

Embedding是将文本转换为向量表示的核心环节，其质量直接决定检索效果。优秀的Embedding应该能够：

- **语义捕捉**：将语义相似的文本映射到相近的向量空间
- **区分能力**：将不相关的文本保持足够的距离
- **鲁棒性**：对拼写错误、同义词、表述变化具有一定的容错能力
- **效率**：在合理的时间和资源消耗内完成编码

---

## 模型选择策略

### 中文场景推荐模型

#### 1. BGE系列（智源研究院）⭐ 强烈推荐

**模型对比**：

| 模型 | 维度 | 最大长度 | MTEB排名 | 适用场景 |
|------|------|----------|----------|---------|
| bge-large-zh-v1.5 | 1024 | 512 | 中文第1 | 通用场景，平衡性能和速度 |
| bge-base-zh-v1.5 | 768 | 512 | 中文前3 | 资源受限场景 |
| bge-m3 | 1024 | 8192 | 多语言Top | 国际化、长文档、多粒度 |
| bge-small-zh-v1.5 | 512 | 512 | - | 移动端、嵌入式设备 |

**优势**：
- 针对中文深度优化
- 在C-MTEB基准上表现优异
- 提供多种尺寸选择
- 开源免费，可商用

**使用示例**：
```python
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# 编码文本
texts = ["人工智能的发展", "AI技术的进步"]
embeddings = model.encode(texts, normalize_embeddings=True)

# 计算相似度
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(embeddings[0].reshape(1, -1), 
                                embeddings[1].reshape(1, -1))
print(f"相似度: {similarity[0][0]:.4f}")
```

#### 2. Text2Vec系列

**模型对比**：

| 模型 | 维度 | 特点 |
|------|------|------|
| text2vec-base-chinese | 768 | 轻量级，速度快 |
| text2vec-large-chinese | 1024 | 精度更高 |
| text2vec-bge-large-chinese | 1024 | 结合BGE优势 |

**适用场景**：
- 对推理速度要求高的场景
- 已有Text2Vec技术栈的团队

#### 3. 商业API

| 服务商 | 模型 | 维度 | 价格 | 优势 |
|--------|------|------|------|------|
| OpenAI | text-embedding-ada-002 | 1536 | $0.0001/1K tokens | 英文场景最优 |
| OpenAI | text-embedding-3-large | 3072 | $0.00013/1K tokens | 最新最强 |
| Cohere | embed-multilingual-v3 | 1024 | $0.0001/1K tokens | 多语言支持好 |
| 阿里云 | text-embedding-v1 | 1536 | ¥0.0007/1K tokens | 国内访问快 |
| 腾讯云 | Embedding | 768 | ¥0.0007/1K tokens | 生态集成好 |

**选择建议**：
- 英文为主 → OpenAI
- 多语言 → Cohere
- 国内业务 → 阿里云/腾讯云
- 成本敏感 → 自建开源模型

### 选型考虑因素

```
决策框架：
├── 1. 语言支持
│   ├── 纯中文 → BGE系列
│   ├── 纯英文 → OpenAI/Sentence-BERT
│   └── 多语言 → BGE-M3/Cohere
│
├── 2. 性能需求
│   ├── QPS要求高 → 小模型 + GPU加速
│   ├── 精度要求高 → 大模型 + 微调
│   └── 平衡型 → BGE-Large
│
├── 3. 资源限制
│   ├── 内存 < 4GB → bge-small (< 200MB)
│   ├── 内存 4-8GB → bge-base (~400MB)
│   └── 内存 > 8GB → bge-large (~1.2GB)
│
├── 4. 最大输入长度
│   ├── 短文本 (< 512) → 任意模型
│   ├── 中长文本 (512-2048) → BGE-M3
│   └── 长文档 (> 2048) → BGE-M3 + 分段策略
│
└── 5. 许可证
    ├── 开源项目 → Apache 2.0/MIT
    ├── 商业应用 → 检查商用许可
    └── 内部使用 → 相对宽松
```

### 模型性能基准测试

```python
import time
import numpy as np
from sentence_transformers import SentenceTransformer

def benchmark_model(model_name, test_texts, batch_size=32):
    """性能基准测试"""
    model = SentenceTransformer(model_name)
    
    # 预热
    _ = model.encode(test_texts[:10])
    
    # 测试推理速度
    start = time.time()
    embeddings = model.encode(test_texts, batch_size=batch_size)
    end = time.time()
    
    qps = len(test_texts) / (end - start)
    avg_latency = (end - start) / len(test_texts) * 1000  # ms
    
    # 测试内存占用
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    return {
        'model': model_name,
        'qps': qps,
        'avg_latency_ms': avg_latency,
        'memory_mb': memory_mb,
        'embedding_dim': embeddings.shape[1]
    }

# 运行测试
test_texts = ["测试文本"] * 1000
results = [
    benchmark_model('BAAI/bge-small-zh-v1.5', test_texts),
    benchmark_model('BAAI/bge-base-zh-v1.5', test_texts),
    benchmark_model('BAAI/bge-large-zh-v1.5', test_texts),
]

for r in results:
    print(f"{r['model']}: QPS={r['qps']:.1f}, "
          f"Latency={r['avg_latency_ms']:.2f}ms, "
          f"Memory={r['memory_mb']:.0f}MB")
```

---

## 领域适配微调

### 何时需要微调

✅ **需要微调的场景**：
- 企业有大量专业术语（医疗、法律、金融、技术等）
- 通用模型在业务数据上的检索效果不佳（Recall@10 < 0.7）
- 有标注的相似/不相似样本对（至少1000对）
- 查询和文档的语言风格差异大

❌ **不需要微调的场景**：
- 通用领域知识问答
- 数据量小（< 1000条）
- 没有标注数据
- 当前效果已满足业务需求

### 准备训练数据

#### 1. 正样本构建方法

**方法一：基于文档结构**
```python
# 同一章节的段落作为正样本
def build_positive_pairs_from_structure(document):
    pairs = []
    sections = document.split('\n\n## ')
    
    for section in sections:
        paragraphs = section.split('\n\n')
        # 同一段落内的句子互为正样本
        for i in range(len(paragraphs)):
            for j in range(i+1, len(paragraphs)):
                pairs.append((paragraphs[i], paragraphs[j]))
    
    return pairs
```

**方法二：基于用户行为**
```python
# 用户点击的query-doc对作为正样本
def build_positive_pairs_from_logs(user_logs):
    pairs = []
    for log in user_logs:
        if log['clicked']:  # 用户点击了该文档
            pairs.append((log['query'], log['document']))
    return pairs
```

**方法三：LLM生成**
```python
from openai import OpenAI

def generate_synthetic_pairs(documents, llm_client):
    """使用LLM生成合成查询-文档对"""
    pairs = []
    
    for doc in documents[:100]:  # 抽样100个文档
        prompt = f"""基于以下文档内容，生成3个用户可能会搜索的查询：

文档内容：{doc[:500]}

请输出JSON格式：
{{"queries": ["查询1", "查询2", "查询3"]}}
"""
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        queries = json.loads(response.choices[0].message.content)['queries']
        
        for query in queries:
            pairs.append((query, doc))
    
    return pairs
```

#### 2. 负样本挖掘

**硬负样本（Hard Negatives）**：与查询相关但不完全匹配的文档

```python
def mine_hard_negatives(query, positive_doc, all_documents, 
                       embedding_model, top_k=10):
    """挖掘硬负样本"""
    query_emb = embedding_model.encode([query])
    doc_embs = embedding_model.encode(all_documents)
    
    # 计算相似度
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_emb, doc_embs)[0]
    
    # 排除正样本，选择相似度最高的作为硬负样本
    sorted_indices = np.argsort(similarities)[::-1]
    
    hard_negatives = []
    for idx in sorted_indices:
        if all_documents[idx] != positive_doc and len(hard_negatives) < 3:
            hard_negatives.append(all_documents[idx])
    
    return hard_negatives
```

**随机负样本**：从语料库中随机采样

```python
import random

def sample_random_negatives(query, positive_doc, all_documents, n=5):
    """采样随机负样本"""
    candidates = [doc for doc in all_documents if doc != positive_doc]
    return random.sample(candidates, min(n, len(candidates)))
```

### 微调实现

#### 使用Sentence Transformers微调

```python
from sentence_transformers import (
    SentenceTransformer, 
    InputExample, 
    losses,
    evaluation
)
from torch.utils.data import DataLoader
import numpy as np

class EmbeddingFineTuner:
    def __init__(self, base_model='BAAI/bge-large-zh-v1.5'):
        self.model = SentenceTransformer(base_model)
    
    def prepare_training_data(self, pairs_with_labels):
        """
        准备训练数据
        
        Args:
            pairs_with_labels: [(text1, text2, label), ...]
                              label: 1表示相似，0表示不相似
        """
        train_examples = []
        for text1, text2, label in pairs_with_labels:
            train_examples.append(InputExample(
                texts=[text1, text2],
                label=float(label)
            ))
        return train_examples
    
    def train(self, train_examples, dev_examples=None, 
              epochs=4, batch_size=16, warmup_steps=100,
              output_path='./fine_tuned_model'):
        """
        执行微调
        
        Args:
            train_examples: 训练样本
            dev_examples: 验证样本（可选）
            epochs: 训练轮数
            batch_size: 批次大小
            warmup_steps: 预热步数
            output_path: 模型保存路径
        """
        # 创建数据加载器
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        # 选择损失函数
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # 验证器（可选）
        evaluators = []
        if dev_examples:
            dev_sentences1 = [ex.texts[0] for ex in dev_examples]
            dev_sentences2 = [ex.texts[1] for ex in dev_examples]
            dev_scores = [ex.label for ex in dev_examples]
            
            evaluator = evaluation.EmbeddingSimilarityEvaluator(
                dev_sentences1, 
                dev_sentences2, 
                dev_scores
            )
            evaluators.append(evaluator)
        
        # 开始训练
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluators[0] if evaluators else None,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True
        )
        
        print(f"模型已保存到: {output_path}")
        return self.model
    
    def evaluate(self, test_pairs):
        """评估微调后的模型"""
        sentences1 = [p[0] for p in test_pairs]
        sentences2 = [p[1] for p in test_pairs]
        labels = [p[2] for p in test_pairs]
        
        embeddings1 = self.model.encode(sentences1)
        embeddings2 = self.model.encode(sentences2)
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.diag(cosine_similarity(embeddings1, embeddings2))
        
        # 计算AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, similarities)
        
        print(f"Test AUC: {auc:.4f}")
        return auc

# 使用示例
if __name__ == '__main__':
    # 初始化
    tuner = EmbeddingFineTuner('BAAI/bge-large-zh-v1.5')
    
    # 准备数据
    train_data = [
        ("如何重置密码", "密码重置操作指南", 1),
        ("如何重置密码", "账户注册流程", 0),
        # ... 更多样本
    ]
    
    dev_data = [
        ("登录失败怎么办", "登录问题排查", 1),
        ("登录失败怎么办", "产品功能介绍", 0),
    ]
    
    # 训练
    tuner.train(train_data, dev_data, epochs=4, batch_size=16)
    
    # 评估
    test_data = [
        ("忘记密码", "密码找回步骤", 1),
        ("忘记密码", "产品价格", 0),
    ]
    tuner.evaluate(test_data)
```

#### 对比学习微调（无监督）

适用于没有标注数据的场景：

```python
from sentence_transformers import losses

def contrastive_finetune(model, documents, epochs=2, batch_size=32):
    """
    使用MultipleNegativesRankingLoss进行无监督微调
    
    原理：同一batch内，(anchor, positive)为正样本对，
         其他所有为负样本
    """
    # 构造训练数据：(anchor, positive)对
    # 这里可以使用数据增强生成positive
    train_examples = []
    for doc in documents:
        # 简单增强：裁剪、同义词替换等
        augmented = augment_text(doc)
        train_examples.append(InputExample(texts=[doc, augmented]))
    
    train_dataloader = DataLoader(train_examples, shuffle=True, 
                                  batch_size=batch_size)
    
    # 使用MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        show_progress_bar=True
    )
    
    return model

def augment_text(text):
    """简单的文本增强"""
    import jieba
    words = jieba.lcut(text)
    
    # 随机删除10%的词
    if len(words) > 10:
        num_to_remove = max(1, int(len(words) * 0.1))
        indices_to_remove = random.sample(range(len(words)), num_to_remove)
        words = [w for i, w in enumerate(words) if i not in indices_to_remove]
    
    return ''.join(words)
```

### 微调技巧

#### 1. 学习率调度

```python
from transformers import get_linear_schedule_with_warmup

# 推荐配置
learning_rate = 2e-5  # 小学习率，避免灾难性遗忘
warmup_ratio = 0.1    # 10%的步数用于预热
weight_decay = 0.01   # L2正则化
```

#### 2. 渐进式解冻

```python
# 先冻结底层，只训练顶层
for param in model.bert.parameters():
    param.requires_grad = False

# 解冻最后几层
for param in model.bert.encoder.layer[-2:].parameters():
    param.requires_grad = True

# 训练几个epoch后，再解冻更多层
```

#### 3. 早停策略

```python
from sentence_transformers.evaluation import SequentialEvaluator

# 监控验证集性能，性能不再提升时停止
evaluator = SequentialEvaluator([
    evaluation.EmbeddingSimilarityEvaluator(...),
])

model.fit(
    ...,
    evaluation_steps=500,  # 每500步评估一次
    callback=lambda score, epoch, step: score > best_score
)
```

---

## 工程优化技巧

### 批处理加速

```python
import numpy as np
from typing import List

class BatchEncoder:
    def __init__(self, model, batch_size=32, device='cuda'):
        self.model = model
        self.batch_size = batch_size
        self.model.to(device)
    
    def encode(self, texts: List[str], normalize=True) -> np.ndarray:
        """批量编码，自动分batch"""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def encode_with_cache(self, texts: List[str], cache) -> np.ndarray:
        """带缓存的批量编码"""
        uncached_texts = []
        uncached_indices = []
        embeddings = [None] * len(texts)
        
        # 检查缓存
        for i, text in enumerate(texts):
            cached_emb = cache.get(text)
            if cached_emb is not None:
                embeddings[i] = cached_emb
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 编码未缓存的文本
        if uncached_texts:
            new_embeddings = self.encode(uncached_texts)
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings[idx] = emb
                cache.set(texts[idx], emb)  # 写入缓存
        
        return np.array(embeddings)
```

### 向量归一化

```python
from sklearn.preprocessing import normalize

def normalize_embeddings(embeddings, norm='l2'):
    """
    L2归一化，确保余弦相似度计算准确
    
    重要：大多数相似度搜索都假设向量已归一化
    """
    return normalize(embeddings, norm=norm)

# 使用示例
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embeddings = model.encode(texts)
normalized_embeddings = normalize_embeddings(embeddings)

# 现在可以直接使用点积代替余弦相似度
# dot_product(normalized_a, normalized_b) == cosine_similarity(a, b)
```

### 缓存策略

#### Redis缓存实现

```python
import redis
import pickle
import hashlib

class EmbeddingCache:
    def __init__(self, host='localhost', port=6379, db=0, ttl=86400):
        self.redis = redis.Redis(host=host, port=port, db=db)
        self.ttl = ttl  # 缓存过期时间（秒），默认24小时
    
    def _get_key(self, text: str) -> str:
        """生成缓存key"""
        # 使用MD5哈希，避免key过长
        hash_obj = hashlib.md5(text.encode('utf-8'))
        return f"emb:{hash_obj.hexdigest()}"
    
    def get(self, text: str) -> np.ndarray:
        """获取缓存的embedding"""
        key = self._get_key(text)
        cached = self.redis.get(key)
        
        if cached:
            return pickle.loads(cached)
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """设置embedding缓存"""
        key = self._get_key(text)
        serialized = pickle.dumps(embedding)
        self.redis.setex(key, self.ttl, serialized)
    
    def clear(self):
        """清空缓存"""
        self.redis.flushdb()
    
    def stats(self):
        """缓存统计"""
        info = self.redis.info()
        return {
            'used_memory_human': info['used_memory_human'],
            'keys_count': self.redis.dbsize()
        }

# 使用示例
cache = EmbeddingCache(ttl=86400)  # 24小时过期
encoder = BatchEncoder(model)

# 带缓存的编码
embeddings = encoder.encode_with_cache(texts, cache)
```

#### 本地LRU缓存

```python
from functools import lru_cache

class LocalEmbeddingCache:
    def __init__(self, maxsize=10000):
        self.maxsize = maxsize
    
    @lru_cache(maxsize=10000)
    def get_or_compute(self, text_hash: str, text: str) -> tuple:
        """
        使用lru_cache装饰器
        注意：需要将text作为参数传入以保持一致性
        """
        # 这个方法需要配合外部调用
        pass

# 更实用的实现
from cachetools import TTLCache

class SimpleCache:
    def __init__(self, maxsize=10000, ttl=3600):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        self.cache[key] = value
```

### GPU加速

```python
import torch
from sentence_transformers import SentenceTransformer

def setup_gpu_acceleration(model_name, gpu_id=0):
    """配置GPU加速"""
    # 指定GPU
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型到GPU
    model = SentenceTransformer(model_name, device=device)
    
    # 启用半精度（FP16）加速
    if device.startswith('cuda'):
        model.half()  # 转换为FP16，节省显存并加速
    
    return model

# 多GPU并行
def multi_gpu_encode(model, texts, num_gpus=2):
    """使用多个GPU并行编码"""
    import multiprocessing as mp
    
    # 分割文本
    chunk_size = len(texts) // num_gpus
    text_chunks = [
        texts[i*chunk_size:(i+1)*chunk_size] 
        for i in range(num_gpus)
    ]
    
    # 并行编码
    with mp.Pool(num_gpus) as pool:
        results = pool.starmap(
            encode_on_gpu,
            [(model, chunk, i) for i, chunk in enumerate(text_chunks)]
        )
    
    return np.vstack(results)

def encode_on_gpu(model, texts, gpu_id):
    """在指定GPU上编码"""
    model.to(f'cuda:{gpu_id}')
    return model.encode(texts, batch_size=64)
```

### 量化压缩

```python
# INT8量化（减少50%内存，轻微精度损失）
from sentence_transformers.quantization import quantize_embeddings

def quantize_and_store(embeddings):
    """量化embedding并存储"""
    # 量化
    quantized = quantize_embeddings(embeddings, precision='int8')
    
    # 存储时使用更少的空间
    return quantized

def decode_quantized(quantized_embeddings):
    """反量化"""
    # 注意：需要在量化时保存统计信息
    pass

# FAISS内置量化
import faiss

def create_quantized_index(embeddings, dim):
    """创建量化的FAISS索引"""
    # IVF + PQ量化
    nlist = 100  # Voronoi cells数量
    m = 8        # PQ子空间数量
    
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
    
    # 训练
    index.train(embeddings)
    
    # 添加向量
    index.add(embeddings)
    
    return index
```

---

## 多维度Embedding

### 混合向量表示

结合多种特征获得更全面的表示：

```python
class MultiVectorEncoder:
    def __init__(self, semantic_model, keyword_extractor):
        self.semantic_model = semantic_model
        self.keyword_extractor = keyword_extractor
    
    def encode(self, text, metadata=None):
        """生成多维度向量"""
        
        # 1. 语义向量（主向量）
        semantic_emb = self.semantic_model.encode([text])[0]
        
        # 2. 关键词向量（稀疏向量）
        keywords = self.keyword_extractor.extract(text, top_k=20)
        keyword_emb = self.keywords_to_vector(keywords)
        
        # 3. 结构化向量（如果有元数据）
        if metadata:
            struct_emb = self.metadata_to_vector(metadata)
        else:
            struct_emb = np.zeros(64)
        
        # 组合方式1：拼接
        combined = np.concatenate([
            semantic_emb * 0.7,
            keyword_emb * 0.2,
            struct_emb * 0.1
        ])
        
        return {
            'semantic': semantic_emb,
            'keyword': keyword_emb,
            'structural': struct_emb,
            'combined': combined
        }
    
    def keywords_to_vector(self, keywords, vocab_size=10000):
        """将关键词转换为TF-IDF风格的稀疏向量"""
        vector = np.zeros(vocab_size)
        for kw, score in keywords:
            idx = hash(kw) % vocab_size
            vector[idx] = score
        return vector
    
    def metadata_to_vector(self, metadata):
        """将元数据转换为向量"""
        # 例如：类别one-hot、时间编码等
        category_vec = self.encode_category(metadata.get('category', ''))
        time_vec = self.encode_time(metadata.get('created_at', ''))
        
        return np.concatenate([category_vec, time_vec])
```

### 多向量检索

```python
class MultiVectorRetriever:
    def __init__(self, semantic_index, keyword_index, struct_index):
        self.semantic_index = semantic_index
        self.keyword_index = keyword_index
        self.struct_index = struct_index
    
    def search(self, query_vectors, top_k=20, weights=None):
        """多向量联合检索"""
        if weights is None:
            weights = {'semantic': 0.7, 'keyword': 0.2, 'structural': 0.1}
        
        # 各向量独立检索
        semantic_results = self.semantic_index.search(
            query_vectors['semantic'], top_k=top_k * 2
        )
        keyword_results = self.keyword_index.search(
            query_vectors['keyword'], top_k=top_k * 2
        )
        struct_results = self.struct_index.search(
            query_vectors['structural'], top_k=top_k * 2
        )
        
        # 融合结果
        fused = self.fuse_results(
            semantic_results, keyword_results, struct_results,
            weights
        )
        
        return fused[:top_k]
    
    def fuse_results(self, *result_lists, weights):
        """融合多路检索结果"""
        # 使用RRF或其他融合算法
        return reciprocal_rank_fusion(result_lists, list(weights.values()))
```

### ColBERT风格的多向量

每个token单独编码，支持细粒度匹配：

```python
# 需要安装 colbert-ai
from colbert import Indexer, Searcher

# 索引
indexer = Indexer(checkpoint='colbert-model')
indexer.index(name='my_index', collection=documents)

# 搜索
searcher = Searcher(index='my_index')
results = searcher.search(query, k=10)

# 优势：
# - 支持细粒度的token级别匹配
# - 对长文档效果更好
# - 可以解释哪些token匹配上了
```

---

## 评估与监控

### 离线评估

#### 1. 语义相似度评估

```python
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

def evaluate_semantic_similarity(model, test_pairs):
    """
    评估语义相似度质量
    
    Args:
        test_pairs: [(text1, text2, human_similarity_score), ...]
    """
    texts1 = [p[0] for p in test_pairs]
    texts2 = [p[1] for p in test_pairs]
    human_scores = [p[2] for p in test_pairs]
    
    # 计算预测相似度
    embs1 = model.encode(texts1)
    embs2 = model.encode(texts2)
    pred_scores = np.diag(cosine_similarity(embs1, embs2))
    
    # 计算斯皮尔曼相关系数
    correlation, p_value = spearmanr(human_scores, pred_scores)
    
    print(f"Spearman Correlation: {correlation:.4f} (p={p_value:.4f})")
    return correlation
```

#### 2. 检索任务评估

```python
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

def evaluate_retrieval_quality(model, queries, relevant_docs, top_k=10):
    """
    评估检索质量
    
    Args:
        queries: 查询列表
        relevant_docs: 每个查询的相关文档列表
        top_k: 召回数量
    """
    results = {
        'recall_at_k': [],
        'precision_at_k': [],
        'mrr': []
    }
    
    for query, rel_docs in zip(queries, relevant_docs):
        # 检索
        query_emb = model.encode([query])
        all_doc_embs = model.encode(all_documents)
        
        similarities = cosine_similarity(query_emb, all_doc_embs)[0]
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved_docs = [all_documents[i] for i in top_k_indices]
        
        # 计算Recall@K
        rel_set = set(rel_docs)
        ret_set = set(retrieved_docs)
        recall = len(rel_set & ret_set) / len(rel_set) if rel_set else 0
        results['recall_at_k'].append(recall)
        
        # 计算Precision@K
        precision = len(rel_set & ret_set) / len(ret_set) if ret_set else 0
        results['precision_at_k'].append(precision)
        
        # 计算MRR
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc in rel_set:
                results['mrr'].append(1.0 / rank)
                break
        else:
            results['mrr'].append(0)
    
    # 汇总
    print(f"Recall@{top_k}: {np.mean(results['recall_at_k']):.4f}")
    print(f"Precision@{top_k}: {np.mean(results['precision_at_k']):.4f}")
    print(f"MRR: {np.mean(results['mrr']):.4f}")
    
    return results
```

### 在线监控

#### 关键指标监控

```python
import time
from prometheus_client import Counter, Histogram, Gauge

# Prometheus指标
EMBEDDING_REQUESTS = Counter(
    'embedding_requests_total',
    'Total embedding requests',
    ['model', 'status']
)

EMBEDDING_LATENCY = Histogram(
    'embedding_latency_seconds',
    'Embedding latency',
    ['model']
)

EMBEDDING_CACHE_HIT = Counter(
    'embedding_cache_hits_total',
    'Cache hits'
)

class EmbeddingMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def record_request(self, success=True):
        status = 'success' if success else 'failure'
        EMBEDDING_REQUESTS.labels(model=self.model_name, status=status).inc()
    
    def record_latency(self, latency_seconds):
        EMBEDDING_LATENCY.labels(model=self.model_name).observe(latency_seconds)
    
    def record_cache_hit(self):
        EMBEDDING_CACHE_HIT.inc()
    
    def monitor_encode(self, texts):
        """带监控的编码"""
        start = time.time()
        try:
            embeddings = self.model.encode(texts)
            self.record_request(success=True)
            return embeddings
        except Exception as e:
            self.record_request(success=False)
            raise
        finally:
            latency = time.time() - start
            self.record_latency(latency)
```

#### 漂移检测

```python
def detect_distribution_drift(reference_embeddings, current_embeddings, 
                               threshold=0.1):
    """
    检测embedding分布漂移
    
    如果新数据的embedding分布与参考数据差异过大，可能需要重新训练
    """
    from scipy.spatial.distance import jensenshannon
    
    # 计算直方图分布
    ref_hist, _ = np.histogramdd(reference_embeddings, bins=50)
    cur_hist, _ = np.histogramdd(current_embeddings, bins=50)
    
    # 归一化
    ref_dist = ref_hist.flatten() / ref_hist.sum()
    cur_dist = cur_hist.flatten() / cur_hist.sum()
    
    # Jensen-Shannon散度
    js_divergence = jensenshannon(ref_dist, cur_dist)
    
    if js_divergence > threshold:
        print(f"⚠️ 检测到分布漂移！JS散度: {js_divergence:.4f}")
        return True
    else:
        print(f"✅ 分布正常。JS散度: {js_divergence:.4f}")
        return False
```

---

## 最佳实践

### ✅ Do's

1. **始终归一化向量**
   ```python
   embeddings = model.encode(texts, normalize_embeddings=True)
   ```

2. **使用批处理**
   ```python
   # 好的做法
   embeddings = model.encode(texts, batch_size=32)
   
   # 不好的做法
   embeddings = [model.encode(text) for text in texts]
   ```

3. **实施缓存**
   - 高频查询的embedding结果缓存24小时
   - 定期清理过期缓存

4. **监控性能**
   - 记录QPS、延迟、缓存命中率
   - 设置告警阈值

5. **定期评估**
   - 每月进行一次离线评估
   - 关注Recall@K和MRR指标

6. **版本管理**
   ```python
   # 保存模型版本信息
   model.save('./models/bge-large-v1.0')
   # 记录超参数和数据版本
   ```

### ❌ Don'ts

1. **不要混合使用不同模型的embedding**
   ```python
   # 错误！
   query_emb = openai_model.encode(query)
   doc_emb = bge_model.encode(document)
   similarity = cosine_similarity(query_emb, doc_emb)  # 无意义！
   ```

2. **不要忽略文本预处理**
   ```python
   # 应该清理文本
   text = clean_text(raw_text)  # 去除HTML标签、多余空格等
   embedding = model.encode([text])
   ```

3. **不要在CPU上进行大规模编码**
   ```python
   # 数据量大时使用GPU
   model = SentenceTransformer('bge-large', device='cuda')
   ```

4. **不要使用过大的batch size导致OOM**
   ```python
   # 根据显存调整
   batch_size = 32 if has_8gb_gpu else 8
   ```

5. **不要忘记更新索引**
   - 新增文档后及时更新向量索引
   - 可以考虑增量更新或定期全量重建

### 性能调优清单

```
□ 选择合适的模型尺寸（平衡速度和精度）
□ 启用GPU加速（如有）
□ 使用批处理（batch_size=32-64）
□ 实施向量归一化
□ 配置Redis缓存（TTL=24h）
□ 监控关键指标（QPS、延迟、缓存命中率）
□ 定期评估模型效果
□ 建立模型版本管理机制
□ 设置分布漂移告警
□ 文档化embedding pipeline
```

---

## 常见问题

### Q1: 如何选择embedding维度？

**A**: 
- 一般场景：768-1024维足够
- 高精度需求：1536维（OpenAI ada-002）
- 资源受限：512维（bge-small）

维度越高，区分能力越强，但存储和计算成本也越高。

### Q2: Embedding多久需要重新训练？

**A**:
- 有新的大量数据时（> 10万条）
- 业务领域发生变化时
- 评估指标持续下降时
- 一般建议每季度评估一次

### Q3: 如何处理超长文本？

**A**:
1. 切分为多个片段分别编码，然后取平均
2. 使用支持长上下文的模型（如BGE-M3，支持8192）
3. 提取关键句进行编码

```python
def encode_long_text(model, text, max_length=512):
    """处理超长文本"""
    if len(text) <= max_length:
        return model.encode([text])[0]
    
    # 分段编码并平均
    chunks = split_text(text, chunk_size=max_length)
    chunk_embeddings = model.encode(chunks)
    return np.mean(chunk_embeddings, axis=0)
```

### Q4: 向量数据库如何选择？

**A**:
- 小规模（< 100万）：FAISS（本地）
- 中等规模（100万-1000万）：Milvus、Qdrant
- 大规模（> 1000万）：Elasticsearch + 向量插件、Pinecone
- 云原生：AWS OpenSearch、Azure Cognitive Search

---

**文档版本**: v1.0  
**最后更新**: 2024年  
**相关文档**: [RAG系统优化总览.md](./RAG系统优化总览.md)
