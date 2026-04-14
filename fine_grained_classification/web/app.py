"""
Web API服务 - 细粒度猫狗分类
Flask API for Fine-Grained Pet Classification
支持加载训练好的模型
"""
import os
import sys
import io
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 添加CoOp路径（必须！）
COOP_PATH = "/Users/yudu/Documents/毕业设计/CoOp"
sys.path.insert(0, COOP_PATH)

import clip
from clip.simple_tokenizer import SimpleTokenizer

app = Flask(__name__)
CORS(app)


class PetClassifierAPI:
    """宠物分类API服务"""
    
    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.prompt_learner = None
        self.text_encoder = None
        self.classnames = None
        self.use_trained_model = model_path is not None
        self.model_path = model_path
        self.load_model(model_path)
    
    def load_model(self, model_path=None):
        """加载CLIP模型"""
        print("Loading CLIP model...")
        self.clip_model, _ = clip.load("RN50", device=self.device)
        self.clip_model.eval()
        self.clip_model = self.clip_model.float()
        
        # Oxford-IIIT Pets 37类（与数据集严格一致）
        self.classnames = [
            'abyssinian', 'american_bulldog', 'american_pit_bull_terrier',
            'basset_hound', 'beagle', 'bengal',
            'birman', 'bombay', 'boxer', 'british_shorthair', 'chihuahua',
            'egyptian_mau', 'english_cocker_spaniel', 'english_setter',
            'german_shorthaired', 'great_pyrenees', 'havanese',
            'japanese_chin', 'keeshond', 'leonberger',
            'maine_coon', 'miniature_pinscher', 'newfoundland',
            'persian', 'pomeranian', 'pug', 'ragdoll', 'russian_blue',
            'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu',
            'siamese', 'sphynx', 'staffordshire_bull_terrier',
            'wheaten_terrier', 'yorkshire_terrier'
        ]
        
        # 如果提供了模型路径，加载训练好的模型
        if model_path and os.path.exists(model_path):
            print(f"Loading trained model from {model_path}...")
            self.load_trained_model(model_path)
        
        if self.use_trained_model:
            print(f"Using trained model: {model_path}")
        else:
            print("Using zero-shot CLIP model")
        print("Model loaded successfully!")
    
    def load_trained_model(self, model_path):
        """加载训练好的模型（包括动态提示词学习器和文本编码器）"""
        try:
            from dassl.utils import load_checkpoint

            # 加载检查点
            checkpoint = load_checkpoint(model_path)

            # 动态导入自定义模块
            from models.custom_clip import TextEncoder
            from models.dynamic_prompt import AdaptivePromptLearner

            # 创建 prompt_learner
            self.prompt_learner = AdaptivePromptLearner(
                cfg=self._create_cfg(),
                classnames=self.classnames,
                clip_model=self.clip_model
            )

            # 加载权重（忽略固定的 token 向量）
            state_dict = checkpoint['state_dict']
            state_dict.pop("token_prefix", None)
            state_dict.pop("token_suffix", None)
            self.prompt_learner.load_state_dict(state_dict, strict=False)
            self.prompt_learner = self.prompt_learner.to(self.device)
            self.prompt_learner.eval()

            # 创建文本编码器（用于编码动态生成的提示词）
            self.text_encoder = TextEncoder(self.clip_model)
            self.text_encoder = self.text_encoder.to(self.device)
            self.text_encoder.eval()

            print("Trained model loaded successfully (with dynamic prompt pipeline)!")
        except Exception as e:
            print(f"Failed to load trained model: {e}")
            self.use_trained_model = False
    
    def _create_cfg(self):
        """创建临时配置对象"""
        from yacs.config import CfgNode as CN
        cfg = CN()
        cfg.TRAINER = CN()
        cfg.TRAINER.DYNAMIC = CN()
        cfg.TRAINER.DYNAMIC.CTX_INIT = "a photo of a"
        cfg.TRAINER.DYNAMIC.N_CTX = 16
        cfg.TRAINER.DYNAMIC.PREC = "fp32"
        cfg.TRAINER.DYNAMIC.MODEL_TYPE = "dynamic"
        cfg.TRAINER.DYNAMIC.USE_DYNAMIC = True
        cfg.TRAINER.DYNAMIC.USE_ADAPTIVE = True
        return cfg
    
    def preprocess(self, image_bytes):
        """预处理图像"""
        import torchvision.transforms as transforms
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        return image, transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, image_tensor, top_k=5, prompt_template=None):
        """
        预测
        Args:
            image_tensor: 预处理后的图像tensor
            top_k: 返回Top-K结果
            prompt_template: 自定义提示词模板（仅 zero-shot 模式生效）
        """
        with torch.no_grad():
            # 编码图像
            image_features = self.clip_model.encode_image(image_tensor.float())
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if self.use_trained_model and self.prompt_learner is not None:
                # ====== 动态提示词推理路径 ======
                # 1. 通过 prompt_learner 生成图像条件化的动态提示词
                #    prompts: [1, n_cls, n_tkn, ctx_dim]
                prompts = self.prompt_learner(image_features)

                n_cls, n_tkn, ctx_dim = prompts.shape[1], prompts.shape[2], prompts.shape[3]

                # 2. 展平为 [n_cls, n_tkn, ctx_dim] 送入文本编码器
                prompts_flat = prompts.squeeze(0)  # [n_cls, n_tkn, ctx_dim]

                # 3. 用文本编码器编码动态提示词
                tokenized_prompts = self.prompt_learner.tokenized_prompts.to(self.device)
                text_features = self.text_encoder(prompts_flat, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # 4. 计算相似度
                logit_scale = self.clip_model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T).squeeze()
                probs = logits.softmax(dim=-1)

                # 构建提示词描述（标注为动态生成）
                prompt_labels = [f"[dynamic] adaptive prompt for {name.replace('_', ' ')}"
                                 for name in self.classnames]
            else:
                # ====== Zero-shot 推理路径 ======
                if prompt_template and "{cls}" in prompt_template:
                    prompt_labels = [prompt_template.format(cls=name.replace('_', ' '))
                                     for name in self.classnames]
                else:
                    prompt_labels = [f"a photo of a {name.replace('_', ' ')}"
                                     for name in self.classnames]

                tokens = clip.tokenize(prompt_labels).to(self.device)
                text_features = self.clip_model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logit_scale = self.clip_model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T).squeeze()
                probs = logits.softmax(dim=-1)

            # Top-K
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

            results = []
            for prob, idx in zip(top_probs, top_indices):
                results.append({
                    "breed": self.classnames[idx.item()].replace('_', ' '),
                    "prompt": prompt_labels[idx.item()],
                    "probability": round(prob.item(), 4)
                })

            return results


classifier = PetClassifierAPI()


@app.route('/api/classify', methods=['POST'])
def classify():
    """分类API"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        image_bytes = request.files['image'].read()
        top_k = int(request.form.get('top_k', 5))
        prompt_template = request.form.get('prompt_template', None)
        
        image, image_tensor = classifier.preprocess(image_bytes)
        results = classifier.predict(image_tensor, top_k, prompt_template)
        
        return jsonify({
            "success": True,
            "predictions": results,
            "prompt_template": prompt_template,
            "mode": "dynamic_prompt" if classifier.use_trained_model else "zero_shot"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({"status": "ok", "device": classifier.device})


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """获取类别列表"""
    return jsonify({
        "classes": [c.replace('_', ' ') for c in classifier.classnames],
        "count": len(classifier.classnames)
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pet Classification API')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port to run the API server')
    args = parser.parse_args()
    
    # 默认模型路径
    default_model = os.path.join(
        PROJECT_ROOT, 
        "output_fgd/oxford_pets/prompt_learner/model.pth.tar-2"
    )
    
    # 如果没有指定模型，但默认模型存在，使用默认模型
    if args.model is None and os.path.exists(default_model):
        args.model = default_model
    
    print("="*50)
    if args.model:
        print(f"Using trained model: {args.model}")
    else:
        print("Using zero-shot CLIP model")
    print("="*50)
    
    classifier = PetClassifierAPI(model_path=args.model)
    
    print(f"Starting API server on http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)
