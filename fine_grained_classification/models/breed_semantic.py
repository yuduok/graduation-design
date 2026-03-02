"""
品种语义增强模块
Breed Semantic Enhancer for Fine-Grained Classification
"""
import torch
from torch import nn
from collections import defaultdict
import json


class BreedAttributeDatabase:
    """
    品种属性数据库
    存储和管理猫狗品种的细粒度特征描述
    """
    def __init__(self):
        # 初始化品种属性数据库
        self.attributes = {
            # 猫
            "Abyssinian": {
                "type": "cat",
                "coat": "short, ticked, agouti",
                "face": "tapered",
                "body": "athletic, muscular",
                "trait": "active, intelligent, curious"
            },
            "Bengal": {
                "type": "cat",
                "coat": "short, spotted or marbled, wild pattern",
                "face": "wedge-shaped",
                "body": "athletic, long",
                "trait": "energetic, vocal, playful"
            },
            "Birman": {
                "type": "cat",
                "coat": "semi-long, colorpoint, pure white paws",
                "face": "round, blue eyes",
                "body": "stocky",
                "trait": "gentle, affectionate, quiet"
            },
            "Bombay": {
                "type": "cat",
                "coat": "short, sleek, black",
                "face": "round, copper eyes",
                "body": "muscular, medium",
                "trait": "friendly, playful, intelligent"
            },
            "British Shorthair": {
                "type": "cat",
                "coat": "short, dense, plush",
                "face": "round, chubby cheeks",
                "body": "stocky, cobby",
                "trait": "calm, easygoing, independent"
            },
            "Egyptian Mau": {
                "type": "cat",
                "coat": "short, spotted",
                "face": "worried expression",
                "body": "slender, muscular",
                "trait": "fast, intelligent, loyal"
            },
            "Maine Coon": {
                "type": "cat",
                "coat": "long, shaggy, water-resistant",
                "face": "square, tufted ears",
                "body": "large, muscular, rectangular",
                "trait": "gentle, playful, sociable"
            },
            "Persian": {
                "type": "cat",
                "coat": "long, fluffy, dense",
                "face": "flat, brachycephalic",
                "body": "cobby, low to ground",
                "trait": "gentle, quiet, calm"
            },
            "Ragdoll": {
                "type": "cat",
                "coat": "semi-long, silky, colorpoint",
                "face": "round, blue eyes",
                "body": "large, muscular",
                "trait": "gentle, affectionate, relaxed"
            },
            "Russian Blue": {
                "type": "cat",
                "coat": "short, plush, blue-gray",
                "face": "wedge-shaped, green eyes",
                "body": "slender, muscular",
                "trait": "shy, intelligent, quiet"
            },
            "Siamese": {
                "type": "cat",
                "coat": "short, colorpoint",
                "face": "triangular, blue eyes",
                "body": "slender, elegant",
                "trait": "vocal, intelligent, affectionate"
            },
            "Sphynx": {
                "type": "cat",
                "coat": "hairless",
                "face": "wedge-shaped, large ears",
                "body": "muscular, pot-bellied",
                "trait": "extroverted, energetic, affectionate"
            },
            
            # 狗
            "American Bulldog": {
                "type": "dog",
                "coat": "short, smooth",
                "face": "boxy, muscular",
                "body": "stocky, muscular",
                "trait": "confident, social, active"
            },
            "American Pit Bull Terrier": {
                "type": "dog",
                "coat": "short, smooth",
                "face": "broad, strong",
                "body": "muscular, athletic",
                "trait": "loyal, energetic, courageous"
            },
            "Basset Hound": {
                "type": "dog",
                "coat": "short, dense",
                "face": "long, droopy ears",
                "body": "low, long",
                "trait": "gentle, patient, friendly"
            },
            "Beagle": {
                "type": "dog",
                "coat": "short, dense, tricolor",
                "face": "long, droopy ears",
                "body": "sturdy, compact",
                "trait": "curious, friendly, merry"
            },
            "Boxer": {
                "type": "dog",
                "coat": "short, smooth, fawn or brindle",
                "face": "brachycephalic",
                "body": "muscular, square",
                "trait": "playful, energetic, loyal"
            },
            "Chihuahua": {
                "type": "dog",
                "coat": "short or long",
                "face": "apple dome head",
                "body": "tiny, compact",
                "trait": "loyal, sassy, alert"
            },
            "English Cocker Spaniel": {
                "type": "dog",
                "coat": "medium, silky, feathered",
                "face": "floppy ears, sweet expression",
                "body": "compact, sturdy",
                "trait": "affectionate, friendly, intelligent"
            },
            "English Setter": {
                "type": "dog",
                "coat": "medium-long, spotted",
                "face": "friendly, alert",
                "body": "athletic, elegant",
                "trait": "gentle, friendly, energetic"
            },
            "French Bulldog": {
                "type": "dog",
                "coat": "short, smooth",
                "face": "bat ears, flat",
                "body": "compact, muscular",
                "trait": "affectionate, adaptable, playful"
            },
            "German Shorthaired Pointer": {
                "type": "dog",
                "coat": "short, dense",
                "face": "alert, noble",
                "body": "athletic, well-balanced",
                "trait": "energetic, intelligent, trainable"
            },
            "Golden Retriever": {
                "type": "dog",
                "coat": "long, dense, golden",
                "face": "friendly, warm",
                "body": "sturdy, muscular",
                "trait": "friendly, intelligent, devoted"
            },
            "Great Dane": {
                "type": "dog",
                "coat": "short, smooth",
                "face": "noble, rectangular",
                "body": "giant, muscular",
                "trait": "friendly, patient, gentle"
            },
            "Havanese": {
                "type": "dog",
                "coat": "long, silky, double",
                "face": "expressive, gentle",
                "body": "small, sturdy",
                "trait": "affectionate, playful, intelligent"
            },
            "Japanese Chin": {
                "type": "dog",
                "coat": "long, silky, flowing",
                "face": "flat, large eyes, oriental",
                "body": "compact, elegant",
                "trait": "cat-like, intelligent, affectionate"
            },
            "Keeshond": {
                "type": "dog",
                "coat": "thick, double, plush",
                "face": "fox-like, expressive",
                "body": "compact, sturdy",
                "trait": "friendly, alert, trainable"
            },
            "Leonberger": {
                "type": "dog",
                "coat": "long, water-resistant, double",
                "face": "gentle, kind",
                "body": "large, muscular",
                "trait": "gentle, affectionate, calm"
            },
            "Miniature Pinscher": {
                "type": "dog",
                "coat": "short, smooth, glossy",
                "face": "high-set ears, alert",
                "body": "small, muscular",
                "trait": "energetic, fearless, alert"
            },
            "Newfoundland": {
                "type": "dog",
                "coat": "thick, double, water-resistant",
                "face": "gentle, sweet",
                "body": "large, strong",
                "trait": "gentle, sweet-natured, intelligent"
            },
            "Pomeranian": {
                "type": "dog",
                "coat": "long, double, fluffy",
                "face": "fox-like, alert",
                "body": "small, compact",
                "trait": "energetic, friendly, curious"
            },
            "Pug": {
                "type": "dog",
                "coat": "short, smooth, fawn or black",
                "face": "flat, wrinkled, large eyes",
                "body": "compact, muscular",
                "trait": "charming, mischievous, loving"
            },
            "Rottweiler": {
                "type": "dog",
                "coat": "short, dense, black with tan",
                "face": "broad, confident",
                "body": "muscular, powerful",
                "trait": "loyal, confident, protective"
            },
            "Saint Bernard": {
                "type": "dog",
                "coat": "thick, double",
                "face": "gentle, kind",
                "body": "large, powerful",
                "trait": "gentle, friendly, patient"
            },
            "Samoyed": {
                "type": "dog",
                "coat": "thick, double, white",
                "face": "smiling, alert",
                "body": "medium, muscular",
                "trait": "friendly, gentle, adaptable"
            },
            "Scottish Terrier": {
                "type": "dog",
                "coat": "wiry, hard",
                "face": "long, pointed, beard",
                "body": "short-legged, sturdy",
                "trait": "independent, confident, spirited"
            },
            "Shiba Inu": {
                "type": "dog",
                "coat": "double, plush",
                "face": "fox-like, curly tail",
                "body": "compact, muscular",
                "trait": "alert, active, independent"
            },
            "Staffordshire Bull Terrier": {
                "type": "dog",
                "coat": "short, smooth",
                "face": "broad, strong",
                "body": "muscular, stocky",
                "trait": "courageous, reliable, affectionate"
            },
            "Wheaten Terrier": {
                "type": "dog",
                "coat": "soft, silky, wheaten",
                "face": "friendly, alert",
                "body": "medium, athletic",
                "trait": "friendly, energetic, spirited"
            },
            "Yorkshire Terrier": {
                "type": "dog",
                "coat": "long, silky, tan and blue",
                "face": "small, alert, perky ears",
                "body": "tiny, compact",
                "trait": "sprightly, affectionate, confident"
            }
        }
        
        # 提示词模板
        self.templates = {
            "simple": [
                "a photo of a {breed}",
                "a {breed}",
            ],
            "detailed": [
                "a {breed} with {coat} coat",
                "a {breed} known for {trait}",
                "a photo of a {breed}, {body} body",
                "a {breed} with {face} face",
                "a {breed} with {coat} coat and {face} face",
            ],
            "comprehensive": [
                "a {breed} with {coat} coat, {face} face, and {body} body",
                "a {breed} with {coat} coat and {trait} temperament",
                "a photo showing a {breed} with distinct {face} facial features",
                "a {breed} characterized by {trait} personality and {body} build",
                "a {breed} with {coat} coat, known for being {trait}",
            ]
        }
        
        # 特征权重（用于增强embeddings）
        self.feature_weights = {
            "coat": 1.0,
            "face": 1.2,  # 面部特征对细粒度分类更重要
            "body": 0.8,
            "trait": 0.6
        }
    
    def get_attributes(self, breed_name):
        """获取品种属性"""
        return self.attributes.get(breed_name, {})
    
    def get_prompts(self, breed_name, template_type="detailed"):
        """生成品种提示词"""
        attributes = self.get_attributes(breed_name)
        if not attributes:
            # 如果没有属性信息，使用简单模板
            return [t.format(breed=breed_name) for t in self.templates["simple"]]
        
        templates = self.templates.get(template_type, self.templates["detailed"])
        prompts = []
        
        for template in templates:
            try:
                # 使用属性填充模板
                prompt = template.format(breed=breed_name, **attributes)
                prompts.append(prompt)
            except KeyError:
                # 如果缺少某个属性，跳过该模板
                continue
        
        # 至少包含一个基本提示词
        if not prompts:
            prompts = [f"a photo of a {breed_name}"]
        
        return prompts
    
    def get_all_breed_prompts(self, classnames, template_type="detailed"):
        """为所有类别生成提示词"""
        all_prompts = {}
        for classname in classnames:
            # 处理带下划线的类名
            clean_name = classname.replace("_", " ")
            all_prompts[classname] = self.get_prompts(clean_name, template_type)
        return all_prompts
    
    def compute_attribute_similarity(self, breed1, breed2):
        """计算两个品种的属性相似度"""
        attr1 = self.get_attributes(breed1)
        attr2 = self.get_attributes(breed2)
        
        if not attr1 or not attr2:
            return 0.0
        
        # Jaccard相似度
        all_keys = set(attr1.keys()) | set(attr2.keys())
        if not all_keys:
            return 0.0
        
        intersection = sum(1 for k in all_keys if attr1.get(k) == attr2.get(k))
        similarity = intersection / len(all_keys)
        
        return similarity
    
    def compute_pairwise_similarities(self, classnames):
        """计算所有品种对的相似度矩阵"""
        n = len(classnames)
        similarity_matrix = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    clean_name_i = classnames[i].replace("_", " ")
                    clean_name_j = classnames[j].replace("_", " ")
                    sim = self.compute_attribute_similarity(clean_name_i, clean_name_j)
                    similarity_matrix[i, j] = sim
        
        return similarity_matrix


class SemanticEnhancer(nn.Module):
    """
    语义增强模块
    - 使用品种属性信息增强文本embeddings
    - 提高细粒度分类的判别能力
    """
    def __init__(self, clip_model, use_attribute_database=True):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.device = next(clip_model.parameters()).device
        
        # 初始化品种属性数据库
        if use_attribute_database:
            self.attribute_db = BreedAttributeDatabase()
        else:
            self.attribute_db = None
        
        # 特征融合网络
        self.enhance_mlp = nn.Sequential(
            nn.Linear(clip_model.text_projection.shape[1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, clip_model.text_projection.shape[1])
        )
    
    def compute_enhanced_embeddings(self, classnames, template_type="detailed"):
        """
        计算增强的文本embeddings
        Args:
            classnames: 类别名称列表
            template_type: 模板类型
        Returns:
            enhanced_features: 增强的文本特征 [n_cls, feat_dim]
        """
        if self.attribute_db:
            all_prompts = self.attribute_db.get_all_breed_prompts(classnames, template_type)
        else:
            # 使用简单模板
            all_prompts = {name: [f"a photo of a {name.replace('_', ' ')}"] 
                          for name in classnames}
        
        enhanced_features = []
        
        for classname in classnames:
            prompts = all_prompts[classname]
            
            # 对每个提示词进行编码
            prompt_features = []
            for prompt in prompts:
                from clip import clip
                tokens = clip.tokenize(prompt).to(self.device)
                with torch.no_grad():
                    text_feature = self.clip_model.encode_text(tokens)
                prompt_features.append(text_feature)
            
            # 平均所有提示词的特征
            avg_feature = torch.mean(torch.stack(prompt_features), dim=0)
            
            # 通过MLP增强
            enhanced = self.enhance_mlp(avg_feature)
            
            # 归一化
            enhanced = enhanced / enhanced.norm(dim=-1, keepdim=True)
            
            enhanced_features.append(enhanced)
        
        enhanced_features = torch.cat(enhanced_features, dim=0)
        
        return enhanced_features
    
    def compute_hard_negative_weights(self, similarity_matrix, temperature=0.1):
        """
        根据品种相似度计算困难负样本权重
        Args:
            similarity_matrix: [n_cls, n_cls] 品种相似度矩阵
            temperature: 温度参数
        Returns:
            weights: [n_cls, n_cls] 困难负样本权重
        """
        # 相似度越高，权重越大（更难区分）
        weights = torch.exp(similarity_matrix / temperature)
        
        # 排除自身
        mask = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
        weights = weights * (1 - mask)
        
        return weights
