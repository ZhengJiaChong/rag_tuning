# Query改写优化

## 目录

1. [概述](#概述)
2. [查询扩展](#查询扩展)
3. [查询分解](#查询分解)
4. [查询规范化](#查询规范化)
5. [指代消解](#指代消解)
6. [查询分类与路由](#查询分类与路由)
7. [评估与优化](#评估与优化)
8. [最佳实践](#最佳实践)

---

## 概述

### 为什么需要Query改写？

用户原始查询往往存在以下问题：
- ❌ **表述模糊**："这个怎么用"
- ❌ **缺少上下文**：多轮对话中的指代
- ❌ **词汇不匹配**：用户用语与文档用语不一致
- ❌ **复杂查询**：包含多个子问题

Query改写的目标：
- ✅ **提升召回率**：通过扩展和规范化，找到更多相关文档
- ✅ **改善精度**：通过分解和澄清，减少噪声
- ✅ **增强理解**：桥接用户语言和系统语言的鸿沟

### Query改写流程

```
原始查询 → 规范化 → 意图识别 → 改写策略选择 → 生成改写查询 → 检索
              ↓            ↓                        ↓
          拼写纠错    分类路由              扩展/分解/重写
```

---

## 查询扩展

### 1. 同义词扩展

#### 基于词典的扩展

```python
import jieba
from typing import List, Dict

class SynonymExpander:
    """
    基于同义词词典的查询扩展
    
    适合领域固定、术语明确的场景
    """
    
    def __init__(self):
        # 领域同义词词典
        self.synonym_dict = {
            # 技术术语
            'AI': ['人工智能', '机器智能', '智能系统'],
            '机器学习': ['ML', 'machine learning'],
            '深度学习': ['DL', 'deep learning', '神经网络'],
            
            # 操作类
            '登录': ['登陆', 'signin', 'login', '登入'],
            '注册': [' signup', 'sign up', '开户'],
            '注销': ['退出', '登出', 'logout', 'signout'],
            
            # 问题类
            'bug': ['错误', '缺陷', '问题', '故障', '异常'],
            '报错': ['错误提示', '异常信息', '告警'],
            
            # 业务术语
            '订单': ['定单', '购买记录'],
            '账单': ['帐单', '费用清单'],
        }
        
        # 反向映射（快速查找）
        self.reverse_dict = {}
        for word, synonyms in self.synonym_dict.items():
            for syn in synonyms:
                if syn not in self.reverse_dict:
                    self.reverse_dict[syn] = []
                self.reverse_dict[syn].append(word)
    
    def expand(self, query: str, max_expansions: int = 5) -> List[str]:
        """
        扩展查询
        
        Args:
            query: 原始查询
            max_expansions: 最大扩展数量
        
        Returns:
            扩展后的查询列表（包含原始查询）
        """
        expanded_queries = {query}  # 使用集合去重
        
        # 分词
        words = jieba.lcut(query)
        
        # 对每个词尝试替换
        for word in words:
            # 查找同义词
            synonyms = self._get_synonyms(word)
            
            for syn in synonyms[:3]:  # 每个词最多取3个同义词
                # 替换生成新查询
                new_query = query.replace(word, syn, 1)
                
                if new_query != query and len(expanded_queries) < max_expansions:
                    expanded_queries.add(new_query)
        
        return list(expanded_queries)
    
    def _get_synonyms(self, word: str) -> List[str]:
        """获取单词的同义词"""
        synonyms = []
        
        # 直接查找
        if word in self.synonym_dict:
            synonyms.extend(self.synonym_dict[word])
        
        # 反向查找
        if word in self.reverse_dict:
            synonyms.extend(self.reverse_dict[word])
        
        return synonyms
    
    def add_synonyms(self, word: str, synonyms: List[str]):
        """动态添加同义词"""
        self.synonym_dict[word] = synonyms
        
        for syn in synonyms:
            if syn not in self.reverse_dict:
                self.reverse_dict[syn] = []
            self.reverse_dict[syn].append(word)


# 使用示例
expander = SynonymExpander()

query = "如何登录系统"
expanded = expander.expand(query)

print("原始查询:", query)
print("扩展查询:")
for q in expanded:
    print(f"  - {q}")

# 输出:
# 原始查询: 如何登录系统
# 扩展查询:
#   - 如何登录系统
#   - 如何登陆系统
#   - 如何signin系统
```

#### 基于Embedding的扩展

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingBasedExpander:
    """
    基于向量相似度的查询扩展
    
    自动发现语义相似的查询，无需维护词典
    """
    
    def __init__(self, model_name='BAAI/bge-large-zh-v1.5'):
        self.model = SentenceTransformer(model_name)
        
        # 历史查询库（从日志中积累）
        self.historical_queries = []
        self.query_embeddings = None
    
    def build_index(self, queries: List[str]):
        """构建历史查询索引"""
        self.historical_queries = queries
        self.query_embeddings = self.model.encode(
            queries, 
            normalize_embeddings=True
        )
    
    def expand(self, query: str, top_k: int = 5,
              threshold: float = 0.8) -> List[str]:
        """
        基于相似度扩展
        
        Args:
            query: 原始查询
            top_k: 返回的相似查询数量
            threshold: 最低相似度阈值
        
        Returns:
            相似查询列表
        """
        if self.query_embeddings is None:
            raise Exception("请先调用build_index构建索引")
        
        # 编码查询
        query_emb = self.model.encode([query], normalize_embeddings=True)
        
        # 计算相似度
        similarities = cosine_similarity(
            query_emb, 
            self.query_embeddings
        )[0]
        
        # 筛选高相似度的历史查询
        similar_indices = np.where(similarities >= threshold)[0]
        
        if len(similar_indices) == 0:
            return [query]
        
        # 按相似度排序
        sorted_indices = similar_indices[np.argsort(similarities[similar_indices])[::-1]]
        
        # 返回Top-K
        expanded = [query]  # 包含原始查询
        for idx in sorted_indices[:top_k]:
            similar_query = self.historical_queries[idx]
            if similar_query != query:
                expanded.append(similar_query)
        
        return expanded
    
    def add_query(self, query: str):
        """动态添加新查询到索引"""
        self.historical_queries.append(query)
        
        # 重新编码（简化实现，实际应增量更新）
        self.query_embeddings = self.model.encode(
            self.historical_queries,
            normalize_embeddings=True
        )
```

---

### 2. HyDE（Hypothetical Document Embeddings）

先生成假设性答案，再用答案进行检索。

```python
from openai import OpenAI
from typing import List

class HyDERewriter:
    """
    HyDE查询改写器
    
    Hypothetical Document Embeddings
    
    原理：
    1. 让LLM生成假设性答案
    2. 用假设答案的embedding进行检索
    3. 或者将原查询和假设答案结合
    """
    
    def __init__(self, llm_client, model='gpt-3.5-turbo'):
        self.llm = llm_client
        self.model = model
    
    def rewrite(self, query: str) -> Dict:
        """
        HyDE改写
        
        Args:
            query: 原始查询
        
        Returns:
            {
                'original_query': 原始查询,
                'hypothetical_doc': 假设性文档,
                'rewritten_query': 改写后的查询
            }
        """
        # 生成假设性文档
        hypothetical_doc = self._generate_hypothetical_answer(query)
        
        # 组合查询
        rewritten_query = f"{query}\n\n相关概念：{hypothetical_doc[:200]}"
        
        return {
            'original_query': query,
            'hypothetical_doc': hypothetical_doc,
            'rewritten_query': rewritten_query
        }
    
    def _generate_hypothetical_answer(self, query: str) -> str:
        """生成假设性答案"""
        prompt = f"""请根据以下问题，生成一个详细的假设性回答。
这个回答应该包含可能出现在相关文档中的信息和术语。
不需要完全准确，目的是帮助检索系统找到相关文档。

问题：{query}

假设性回答（200-300字）："""
        
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()


# 使用示例
llm_client = OpenAI(api_key="your-api-key")
hyde = HyDERewriter(llm_client)

query = "Transformer架构的核心组件有哪些"
result = hyde.rewrite(query)

print("原始查询:", result['original_query'])
print("\n假设性文档:")
print(result['hypothetical_doc'][:200] + "...")
print("\n改写后查询:")
print(result['rewritten_query'][:150] + "...")
```

**优势**：
- ✅ 桥接查询和文档的词汇鸿沟
- ✅ 特别适合知识密集型任务
- ✅ 能召回使用不同术语但内容相关的文档

**劣势**：
- ❌ 需要调用LLM，成本高
- ❌ 延迟增加（~1-2秒）
- ❌ 可能引入幻觉信息

**适用场景**：
- 复杂的技术问答
- 学术研究检索
- 专业领域查询

---

### 3. Query Expansion with LLM

使用LLM直接生成扩展查询。

```python
class LLMQueryExpander:
    """
    基于LLM的查询扩展
    
    让LLM生成多个角度的改写查询
    """
    
    def __init__(self, llm_client, model='gpt-3.5-turbo'):
        self.llm = llm_client
        self.model = model
    
    def expand(self, query: str, num_variations: int = 5) -> List[str]:
        """
        生成查询变体
        
        Args:
            query: 原始查询
            num_variations: 生成的变体数量
        
        Returns:
            查询变体列表
        """
        prompt = f"""你是一个查询优化专家。请为以下查询生成{num_variations}个不同角度的改写版本。

要求：
1. 保持原意不变
2. 使用不同的表述方式
3. 可以补充隐含的背景信息
4. 每个变体一行

原始查询：{query}

改写变体："""
        
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        
        variations = response.choices[0].message.content.strip().split('\n')
        
        # 清理
        variations = [v.strip().lstrip('- ').strip() for v in variations if v.strip()]
        
        # 确保包含原始查询
        if query not in variations:
            variations.insert(0, query)
        
        return variations[:num_variations + 1]


# 使用示例
expander = LLMQueryExpander(llm_client)

query = "如何优化数据库性能"
variations = expander.expand(query, num_variations=5)

print("原始查询:", query)
print("\n改写变体:")
for i, var in enumerate(variations, 1):
    print(f"{i}. {var}")

# 可能的输出:
# 1. 如何优化数据库性能
# 2. 数据库性能优化的方法和技巧
# 3. 提升数据库查询速度的最佳实践
# 4. 数据库索引优化策略
# 5. 如何减少数据库响应时间
# 6. 数据库调优指南
```

---

## 查询分解

将复杂查询拆分为多个简单的子查询。

### 1. 基于LLM的查询分解

```python
import json
from typing import List, Dict

class QueryDecomposer:
    """
    查询分解器
    
    将复杂查询拆分为可独立检索的子查询
    """
    
    def __init__(self, llm_client, model='gpt-4'):
        self.llm = llm_client
        self.model = model
    
    def decompose(self, query: str) -> Dict:
        """
        分解复杂查询
        
        Args:
            query: 原始查询
        
        Returns:
            {
                'original_query': 原始查询,
                'sub_queries': 子查询列表,
                'decomposition_strategy': 分解策略,
                'execution_order': 执行顺序 ('parallel' | 'sequential')
            }
        """
        prompt = f"""请将以下复杂问题分解为2-4个简单的子问题。
每个子问题应该：
1. 独立且易于检索
2. 覆盖原问题的不同方面
3. 避免重复

原始问题：{query}

请按以下JSON格式输出：
{{
    "sub_queries": ["子问题1", "子问题2", ...],
    "strategy": "parallel" 或 "sequential",
    "reasoning": "分解的理由"
}}
"""
        
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            
            return {
                'original_query': query,
                'sub_queries': result['sub_queries'],
                'decomposition_strategy': result.get('strategy', 'parallel'),
                'reasoning': result.get('reasoning', '')
            }
        
        except json.JSONDecodeError:
            # 解析失败，返回原始查询
            return {
                'original_query': query,
                'sub_queries': [query],
                'decomposition_strategy': 'none',
                'reasoning': '解析失败，未分解'
            }
    
    def execute_and_merge(self, query: str, retriever) -> List[Dict]:
        """
        执行分解后的查询并合并结果
        
        Args:
            query: 原始查询
            retriever: 检索器
        
        Returns:
            合并后的检索结果
        """
        # 分解
        decomposition = self.decompose(query)
        
        sub_queries = decomposition['sub_queries']
        strategy = decomposition['decomposition_strategy']
        
        if strategy == 'none' or len(sub_queries) == 1:
            # 无需分解
            return retriever.search(query, top_k=10)
        
        # 并行执行所有子查询
        all_results = []
        
        for sub_query in sub_queries:
            results = retriever.search(sub_query, top_k=10)
            all_results.extend(results)
        
        # 去重（基于文档ID）
        seen_ids = set()
        unique_results = []
        
        for result in all_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        # 重新排序（按相关性）
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return unique_results[:10]


# 使用示例
decomposer = QueryDecomposer(llm_client)

query = "对比GPT-4和Claude在代码生成和安全方面的差异"
result = decomposer.decompose(query)

print("原始查询:", result['original_query'])
print("分解策略:", result['decomposition_strategy'])
print("子查询:")
for i, sq in enumerate(result['sub_queries'], 1):
    print(f"  {i}. {sq}")

# 输出:
# 原始查询: 对比GPT-4和Claude在代码生成和安全方面的差异
# 分解策略: parallel
# 子查询:
#   1. GPT-4的代码生成能力特点
#   2. Claude的代码生成能力特点
#   3. GPT-4的安全性措施
#   4. Claude的安全性措施
```

### 2. 基于规则的查询分解

```python
import re

class RuleBasedDecomposer:
    """
    基于规则的查询分解
    
    适用于模式固定的场景，速度快
    """
    
    def __init__(self):
        # 连接词模式
        self.connectors = [
            r'\s+和\s+',
            r'\s+以及\s+',
            r'\s+还有\s+',
            r'\s+另外\s+',
            r'\s+并且\s+',
        ]
        
        # 对比模式
        self.comparison_patterns = [
            r'(.+)\s+和\s+(.+)\s+的区别',
            r'(.+)\s+vs\s+(.+)',
            r'比较\s+(.+)\s+和\s+(.+)',
        ]
    
    def decompose(self, query: str) -> List[str]:
        """分解查询"""
        
        # 1. 检测对比查询
        comparison_result = self._decompose_comparison(query)
        if comparison_result:
            return comparison_result
        
        # 2. 按连接词分割
        for pattern in self.connectors:
            parts = re.split(pattern, query)
            if len(parts) > 1:
                # 过滤空字符串
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) >= 2:
                    return parts
        
        # 3. 无法分解
        return [query]
    
    def _decompose_comparison(self, query: str) -> List[str]:
        """分解对比查询"""
        for pattern in self.comparison_patterns:
            match = re.search(pattern, query)
            if match:
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip()
                
                # 提取对比维度
                dimension = self._extract_comparison_dimension(query)
                
                if dimension:
                    return [
                        f"{entity1}的{dimension}",
                        f"{entity2}的{dimension}"
                    ]
                else:
                    return [
                        f"{entity1}的特点",
                        f"{entity2}的特点"
                    ]
        
        return None
    
    def _extract_comparison_dimension(self, query: str) -> str:
        """提取对比维度"""
        # 简化的维度提取
        dimensions = [
            '区别', '差异', '不同', '优缺点', '性能',
            '功能', '特点', '优势', '劣势'
        ]
        
        for dim in dimensions:
            if dim in query:
                return dim
        
        return ''


# 使用示例
decomposer = RuleBasedDecomposer()

query = "Python和Java的性能对比"
sub_queries = decomposer.decompose(query)

print("原始查询:", query)
print("子查询:")
for sq in sub_queries:
    print(f"  - {sq}")

# 输出:
# 原始查询: Python和Java的性能对比
# 子查询:
#   - Python的性能
#   - Java的性能
```

---

## 查询规范化

### 1. 拼写纠错

```python
import jieba
from pypinyin import pinyin

class SpellingCorrector:
    """
    拼写纠错器
    
    支持：
    - 常见错别字纠正
    - 拼音模糊匹配
    - 编辑距离匹配
    """
    
    def __init__(self):
        # 常见错别字映射
        self.common_typos = {
            '登路': '登录',
            '注消': '注销',
            '帐单': '账单',
            '登陆': '登录',
            '登人': '登录',
            '密玛': '密码',
            '帐好': '账号',
            '注册 ': '注册',
        }
        
        # 领域术语词典（用于纠错参考）
        self.vocabulary = set()
    
    def build_vocabulary(self, documents: List[str]):
        """从文档构建词汇表"""
        for doc in documents:
            words = jieba.lcut(doc)
            self.vocabulary.update(words)
    
    def correct(self, query: str) -> str:
        """
        纠正拼写错误
        
        Args:
            query: 原始查询
        
        Returns:
            纠正后的查询
        """
        corrected = query
        
        # 1. 基于词典的纠错
        for wrong, correct in self.common_typos.items():
            corrected = corrected.replace(wrong, correct)
        
        # 2. 基于编辑距离的纠错（可选，较慢）
        # corrected = self._edit_distance_correction(corrected)
        
        return corrected
    
    def _edit_distance_correction(self, query: str, max_distance: int = 2) -> str:
        """
        基于编辑距离的纠错
        
        对每个词，在词汇表中查找编辑距离最小的词
        """
        words = jieba.lcut(query)
        corrected_words = []
        
        for word in words:
            if len(word) <= 1:
                corrected_words.append(word)
                continue
            
            # 如果词在词汇表中，无需纠正
            if word in self.vocabulary:
                corrected_words.append(word)
                continue
            
            # 查找最相似的词
            best_match = None
            min_distance = max_distance + 1
            
            for vocab_word in self.vocabulary:
                if abs(len(vocab_word) - len(word)) > max_distance:
                    continue
                
                distance = self._levenshtein_distance(word, vocab_word)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = vocab_word
            
            if best_match and min_distance <= max_distance:
                corrected_words.append(best_match)
            else:
                corrected_words.append(word)
        
        return ''.join(corrected_words)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                
                current_row.append(min(insertions, deletions, substitutions))
            
            previous_row = current_row
        
        return previous_row[-1]


# 使用示例
corrector = SpellingCorrector()

queries = [
    "如何登路系统",
    "忘记密玛怎么办",
    "怎么注消账号"
]

for query in queries:
    corrected = corrector.correct(query)
    print(f"原始: {query}")
    print(f"纠正: {corrected}")
    print()
```

### 2. 查询清洗

```python
import re

class QueryNormalizer:
    """
    查询规范化器
    
    统一查询格式，去除噪声
    """
    
    def __init__(self):
        # 停用词
        self.stop_words = {
            '嗯', '啊', '哦', '呃', '那个', '就是',
            '请问', '我想问', '帮我', '麻烦', '谢谢'
        }
    
    def normalize(self, query: str) -> str:
        """
        完整的查询规范化流程
        
        Args:
            query: 原始查询
        
        Returns:
            规范化后的查询
        """
        # 1. 去除首尾空格
        query = query.strip()
        
        # 2. 规范化空白字符
        query = self._normalize_whitespace(query)
        
        # 3. 去除标点符号（保留问号）
        query = self._remove_punctuation(query)
        
        # 4. 去除停用词
        query = self._remove_stop_words(query)
        
        # 5. 统一大小写（英文）
        query = query.lower()
        
        return query
    
    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        # 多个空格合并为一个
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _remove_punctuation(self, text: str) -> str:
        """去除标点符号（保留问号）"""
        # 保留中文问号和英文问号
        text = re.sub(r'[^\w\s？?]', ' ', text)
        return text
    
    def _remove_stop_words(self, text: str) -> str:
        """去除停用词"""
        words = jieba.lcut(text)
        filtered = [w for w in words if w not in self.stop_words and w.strip()]
        return ''.join(filtered)


# 使用示例
normalizer = QueryNormalizer()

queries = [
    "  请问  如何  登录  系统  ",
    "嗯...我想问一下忘记密码怎么办",
    "帮我查一下订单状态！！！"
]

for query in queries:
    normalized = normalizer.normalize(query)
    print(f"原始: '{query}'")
    print(f"规范化: '{normalized}'")
    print()
```

---

## 指代消解

处理多轮对话中的指代问题。

```python
from typing import List, Dict

class CoreferenceResolver:
    """
    指代消解器
    
    在多轮对话中，将当前查询中的指代词替换为具体内容
    """
    
    def __init__(self, llm_client, model='gpt-3.5-turbo'):
        self.llm = llm_client
        self.model = model
    
    def resolve(self, current_query: str,
               conversation_history: List[Dict]) -> str:
        """
        消解指代
        
        Args:
            current_query: 当前轮次的查询
            conversation_history: 对话历史
                                 [{'role': 'user'|'assistant', 'content': ...}, ...]
        
        Returns:
            消解后的查询
        """
        if not conversation_history:
            return current_query
        
        # 检查是否包含指代词
        pronouns = ['它', '他', '她', '这个', '那个', '前者', '后者', '这', '那']
        has_pronoun = any(p in current_query for p in pronouns)
        
        if not has_pronoun:
            return current_query
        
        # 使用LLM进行指代消解
        history_text = self._format_history(conversation_history[-5:])  # 最近5轮
        
        prompt = f"""根据以下对话历史，重写当前问题，将其中的指代词（如"它"、"这个"、"那个"等）替换为具体的内容。

对话历史：
{history_text}

当前问题：{current_query}

重写后的问题（只输出重写后的问题，不要其他解释）："""
        
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        
        rewritten = response.choices[0].message.content.strip()
        
        # 如果LLM返回空或无效结果，返回原始查询
        if not rewritten or len(rewritten) > 200:
            return current_query
        
        return rewritten
    
    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        lines = []
        for msg in history:
            role = '用户' if msg['role'] == 'user' else '助手'
            lines.append(f"{role}: {msg['content']}")
        
        return '\n'.join(lines)


# 使用示例
resolver = CoreferenceResolver(llm_client)

# 模拟对话历史
history = [
    {'role': 'user', 'content': 'GPT-4是什么'},
    {'role': 'assistant', 'content': 'GPT-4是OpenAI开发的大型语言模型...'},
    {'role': 'user', 'content': '它的训练数据有哪些'},
]

current_query = "它的参数量是多少"

resolved = resolver.resolve(current_query, history)

print("当前查询:", current_query)
print("消解后:", resolved)

# 输出:
# 当前查询: 它的参数量是多少
# 消解后: GPT-4的参数量是多少
```

---

## 查询分类与路由

根据查询类型选择不同的处理策略。

### 查询分类器

```python
import re
from typing import Dict

class QueryClassifier:
    """
    查询分类器
    
    将查询分类到不同的意图类别
    """
    
    def __init__(self):
        self.categories = {
            'factual': '事实查询',
            'howto': '操作指南',
            'comparison': '对比查询',
            'troubleshooting': '故障排查',
            'conceptual': '概念解释',
            'opinion': '观点咨询',
            'chit_chat': '闲聊',
        }
    
    def classify(self, query: str) -> Dict:
        """
        分类查询
        
        Args:
            query: 用户查询
        
        Returns:
            {
                'category': 类别,
                'confidence': 置信度,
                'method': 分类方法
            }
        """
        # 方法1：基于规则（快速）
        rule_result = self._rule_based_classify(query)
        
        if rule_result['confidence'] > 0.8:
            return rule_result
        
        # 方法2：基于模型（更准确，但较慢）
        # model_result = self._model_based_classify(query)
        # return model_result
        
        return rule_result
    
    def _rule_based_classify(self, query: str) -> Dict:
        """基于规则的快速分类"""
        
        # 闲聊检测
        chit_chat_patterns = ['你好', '您好', '再见', '谢谢', '在吗']
        if any(pattern in query for pattern in chit_chat_patterns):
            return {
                'category': 'chit_chat',
                'confidence': 0.9,
                'method': 'rule_based'
            }
        
        # 操作指南
        if any(word in query for word in ['如何', '怎么', '步骤', '教程', '指南']):
            return {
                'category': 'howto',
                'confidence': 0.85,
                'method': 'rule_based'
            }
        
        # 对比查询
        if any(word in query for word in ['区别', '对比', 'vs', '比较', '差异']):
            return {
                'category': 'comparison',
                'confidence': 0.85,
                'method': 'rule_based'
            }
        
        # 故障排查
        if any(word in query for word in ['错误', '失败', '问题', 'bug', '报错', '异常']):
            return {
                'category': 'troubleshooting',
                'confidence': 0.8,
                'method': 'rule_based'
            }
        
        # 概念解释
        if any(word in query for word in ['什么', '什么是', '定义', '含义']):
            return {
                'category': 'conceptual',
                'confidence': 0.8,
                'method': 'rule_based'
            }
        
        # 默认为事实查询
        return {
            'category': 'factual',
            'confidence': 0.6,
            'method': 'rule_based'
        }
```

### 查询路由器

```python
class QueryRouter:
    """
    查询路由器
    
    根据查询类型路由到不同的处理管道
    """
    
    def __init__(self, classifier, retrievers: Dict):
        self.classifier = classifier
        self.retrievers = retrievers  # 不同类别使用不同的检索器
    
    def route(self, query: str) -> Dict:
        """
        路由查询
        
        Args:
            query: 用户查询
        
        Returns:
            {
                'category': 查询类别,
                'retrieval_config': 检索配置,
                'rewriting_strategy': 改写策略
            }
        """
        # 分类
        classification = self.classifier.classify(query)
        category = classification['category']
        
        # 根据类别选择配置
        routing_config = self._get_routing_config(category)
        
        return {
            'category': category,
            'confidence': classification['confidence'],
            'retrieval_config': routing_config['retrieval'],
            'rewriting_strategy': routing_config['rewriting']
        }
    
    def _get_routing_config(self, category: str) -> Dict:
        """获取路由配置"""
        configs = {
            'factual': {
                'retrieval': {
                    'strategy': 'hybrid',
                    'top_k': 10,
                    'need_rerank': True,
                },
                'rewriting': {
                    'expand': True,
                    'decompose': False,
                }
            },
            'howto': {
                'retrieval': {
                    'strategy': 'keyword_boost',
                    'top_k': 15,
                    'need_rerank': True,
                    'step_by_step': True,
                },
                'rewriting': {
                    'expand': False,
                    'decompose': True,
                }
            },
            'comparison': {
                'retrieval': {
                    'strategy': 'decompose_and_merge',
                    'top_k': 20,
                    'need_rerank': True,
                },
                'rewriting': {
                    'expand': False,
                    'decompose': True,
                }
            },
            'troubleshooting': {
                'retrieval': {
                    'strategy': 'error_code_priority',
                    'top_k': 10,
                    'need_rerank': True,
                },
                'rewriting': {
                    'expand': True,
                    'decompose': False,
                    'correct_spelling': True,
                }
            },
            'conceptual': {
                'retrieval': {
                    'strategy': 'vector_dominant',
                    'top_k': 10,
                    'need_rerank': True,
                },
                'rewriting': {
                    'expand': True,
                    'decompose': False,
                    'use_hyde': True,
                }
            },
            'chit_chat': {
                'retrieval': {
                    'strategy': 'none',  # 闲聊无需检索
                    'top_k': 0,
                    'need_rerank': False,
                },
                'rewriting': {
                    'expand': False,
                    'decompose': False,
                }
            },
        }
        
        return configs.get(category, configs['factual'])


# 使用示例
classifier = QueryClassifier()
router = QueryRouter(classifier, retrievers={})

query = "如何重置密码"
routing = router.route(query)

print("查询:", query)
print("类别:", routing['category'])
print("检索策略:", routing['retrieval_config']['strategy'])
print("改写策略:", routing['rewriting_strategy'])
```

---

## 评估与优化

### Query改写效果评估

```python
class QueryRewritingEvaluator:
    """Query改写效果评估器"""
    
    def __init__(self):
        pass
    
    def evaluate_expansion(self, expander, test_cases: List[Dict],
                          retriever) -> Dict:
        """
        评估查询扩展效果
        
        Args:
            test_cases: [{'query': ..., 'relevant_docs': [...]}, ...]
        """
        metrics = {
            'recall_improvement': [],
            'precision_change': [],
            'latency_overhead': []
        }
        
        for case in test_cases:
            query = case['query']
            relevant_docs = set(case['relevant_docs'])
            
            # 原始查询检索
            start = time.time()
            original_results = retriever.search(query, top_k=10)
            original_latency = time.time() - start
            
            original_recall = self._compute_recall(original_results, relevant_docs)
            
            # 扩展后检索
            start = time.time()
            expanded_queries = expander.expand(query)
            
            all_results = []
            for exp_query in expanded_queries:
                results = retriever.search(exp_query, top_k=10)
                all_results.extend(results)
            
            # 去重
            seen = set()
            unique_results = []
            for r in all_results:
                if r['id'] not in seen:
                    seen.add(r['id'])
                    unique_results.append(r)
            
            expanded_latency = time.time() - start
            
            expanded_recall = self._compute_recall(unique_results[:10], relevant_docs)
            
            # 记录指标
            metrics['recall_improvement'].append(expanded_recall - original_recall)
            metrics['latency_overhead'].append(expanded_latency - original_latency)
        
        return {
            'avg_recall_improvement': np.mean(metrics['recall_improvement']),
            'avg_latency_overhead_ms': np.mean(metrics['latency_overhead']) * 1000,
        }
    
    def _compute_recall(self, results: List[Dict], relevant_docs: set) -> float:
        """计算召回率"""
        retrieved = set(r['id'] for r in results)
        if not relevant_docs:
            return 0.0
        return len(retrieved & relevant_docs) / len(relevant_docs)
```

---

## 最佳实践

### ✅ Do's

1. **始终进行查询规范化**
   ```python
   normalized_query = normalizer.normalize(raw_query)
   ```

2. **根据查询类型选择改写策略**
   - 简单查询：同义词扩展
   - 复杂查询：分解
   - 知识密集：HyDE

3. **缓存改写结果**
   ```python
   # 相同查询无需重复改写
   cache_key = hash(query)
   if cache_key in cache:
       return cache[cache_key]
   ```

4. **设置超时保护**
   ```python
   # LLM改写设置超时
   result = llm_rewrite(query, timeout_ms=2000)
   if result is None:
       return query  # fallback到原始查询
   ```

5. **监控改写效果**
   - 记录改写前后的Recall变化
   - A/B测试不同改写策略

6. **渐进式启用**
   - 先上线规范化（风险低）
   - 再上线扩展（中等风险）
   - 最后上线LLM改写（高风险）

### ❌ Don'ts

1. **不要过度扩展**
   ```python
   # 错误！扩展太多会导致噪声
   expanded = expand(query, max_expansions=50)
   
   # 正确：限制扩展数量
   expanded = expand(query, max_expansions=5)
   ```

2. **不要对所有查询使用LLM改写**
   - 成本高
   - 延迟大
   - 只对复杂查询使用

3. **不要忘记Fallback**
   ```python
   try:
       rewritten = llm_rewrite(query)
   except:
       rewritten = query  # 失败时使用原始查询
   ```

4. **不要忽略用户反馈**
   - 收集用户对改写结果的反馈
   - 持续优化改写策略

### 性能优化清单

```
□ 实施查询规范化（低开销，高收益）
□ 建立同义词词典（针对领域定制）
□ 缓存热门查询的改写结果
□ 限制LLM改写的使用频率
□ 设置合理的超时时间
□ 监控改写延迟和效果
□ A/B测试不同改写策略
□ 定期更新同义词词典
□ 收集bad cases并优化
□ 建立降级机制（LLM不可用时）
```

---

## 常见问题

### Q1: Query改写会增加多少延迟？

**A**:
- 规范化：< 1ms
- 同义词扩展：< 10ms
- LLM扩展：500-2000ms
- HyDE：1000-3000ms

建议：对延迟敏感的场景，优先使用规则-based方法。

### Q2: 如何判断是否需要改写？

**A**:
- 查询长度 < 5个字：可能需要扩展
- 包含指代词：需要指代消解
- 包含多个问题：需要分解
- 专有名词多：需要拼写纠错

### Q3: 改写会不会引入噪声？

**A**: 
有可能。缓解方法：
- 限制扩展数量（≤5个）
- 设置相似度阈值
- 对扩展结果去重
- 人工审核高频扩展

### Q4: 何时使用HyDE？

**A**:
- 知识密集型问答
- 专业领域查询
- 召回率低于预期时
- 有充足的时间和预算

---

**文档版本**: v1.0  
**相关文档**: [RAG系统优化总览.md](./RAG系统优化总览.md)
