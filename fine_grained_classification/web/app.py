"""
Web API服务 - 细粒度猫狗分类（研究增强版）
Flask API for Fine-Grained Pet Classification Research Demo
支持：动态提示词推理 / 多模型对比 / 品种语义信息 / 实验结果展示
"""
import os
import sys
import io
import json
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 添加CoOp路径（必须！）
COOP_PATH = os.path.join(PROJECT_ROOT, "CoOp")
sys.path.insert(0, COOP_PATH)

# 添加fine_grained_classification路径（用于导入models模块）
FGDC_PATH = os.path.join(PROJECT_ROOT, "fine_grained_classification")
sys.path.insert(0, FGDC_PATH)

import clip
from clip.simple_tokenizer import SimpleTokenizer

app = Flask(__name__, static_folder='static')
CORS(app)


class PetClassifierAPI:
    """宠物分类API服务 - 研究增强版"""

    def __init__(self, model_path=None, shot="16"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.prompt_learner = None
        self.text_encoder = None
        self.classnames = None
        self.use_trained_model = model_path is not None
        self.model_path = model_path
        self.shot = shot

        # 品种属性数据库
        from models.breed_semantic import BreedAttributeDatabase
        self.breed_db = BreedAttributeDatabase()

        # 实验结果缓存
        self.experiment_summary = self._load_experiment_summary()

        self.load_model(model_path)

    def _load_experiment_summary(self):
        """加载实验结果摘要"""
        summary_path = os.path.join(FGDC_PATH, "output_fgd", "oxford_pets", "experiment_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                return json.load(f)
        return {}

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

            # 记录原始设备，将 clip_model 临时移到 CPU
            # 避免 AdaptivePromptLearner 初始化时内部 buffer 被创建在 CUDA 上
            original_device = next(self.clip_model.parameters()).device
            self.clip_model = self.clip_model.cpu()

            # 创建 prompt_learner（在 CPU 上）
            self.prompt_learner = AdaptivePromptLearner(
                cfg=self._create_cfg(),
                classnames=self.classnames,
                clip_model=self.clip_model
            )

            # 加载权重（忽略固定的 token 向量）
            state_dict = checkpoint['state_dict']
            state_dict.pop("token_prefix", None)
            state_dict.pop("token_suffix", None)

            # 将 state_dict 中的所有张量移到 CPU
            for key in list(state_dict.keys()):
                if isinstance(state_dict[key], torch.Tensor):
                    state_dict[key] = state_dict[key].cpu()

            self.prompt_learner.load_state_dict(state_dict, strict=False)

            # 将 prompt_learner 和 clip_model 移回目标设备
            self.prompt_learner = self.prompt_learner.to(self.device)
            self.clip_model = self.clip_model.to(original_device)
            self.prompt_learner.eval()

            # 创建文本编码器（用于编码动态生成的提示词）
            self.text_encoder = TextEncoder(self.clip_model)
            self.text_encoder = self.text_encoder.to(self.device)
            self.text_encoder.eval()

            print("Trained model loaded successfully (with dynamic prompt pipeline)!")
        except Exception as e:
            print(f"Failed to load trained model: {e}")
            import traceback
            traceback.print_exc()
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
        预测 - 动态提示词推理路径
        Args:
            image_tensor: 预处理后的图像tensor
            top_k: 返回Top-K结果
            prompt_template: 自定义提示词模板（仅 zero-shot 模式生效）
        Returns:
            results: 预测结果列表
            debug_info: 调试信息（动态提示词相关）
        """
        debug_info = {}

        with torch.no_grad():
            # 编码图像
            image_features = self.clip_model.encode_image(image_tensor.float())
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if self.use_trained_model and self.prompt_learner is not None:
                # ====== 动态提示词推理路径 ======
                prompts = self.prompt_learner(image_features)
                n_cls, n_tkn, ctx_dim = prompts.shape[1], prompts.shape[2], prompts.shape[3]
                prompts_flat = prompts.squeeze(0)

                tokenized_prompts = self.prompt_learner.tokenized_prompts.to(self.device)
                text_features = self.text_encoder(prompts_flat, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logit_scale = self.clip_model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T).squeeze()
                probs = logits.softmax(dim=-1)

                # 构建提示词描述：优先使用用户自定义模板
                if prompt_template and "{cls}" in prompt_template:
                    prompt_labels = [prompt_template.format(cls=name.replace('_', ' '))
                                     for name in self.classnames]
                else:
                    prompt_labels = [f"[dynamic] adaptive prompt for {name.replace('_', ' ')}"
                                     for name in self.classnames]

                # 收集调试信息：动态提示词统计
                debug_info["mode"] = "dynamic_prompt"
                debug_info["prompt_shape"] = list(prompts.shape)
                debug_info["ctx_norm"] = self.prompt_learner.ctx.norm().item()

                # 获取 SoftPromptAdapter 的偏移信息（如果有）
                if self.prompt_learner.soft_prompt_adapter is not None:
                    base_ctx = self.prompt_learner.ctx
                    bias = self.prompt_learner.soft_prompt_adapter.meta_net(image_features)
                    debug_info["bias_norm"] = bias.norm().item()
                    debug_info["bias_mean"] = bias.mean().item()

                # 获取类别自适应因子信息
                if (self.prompt_learner.dynamic_optimizer is not None and
                    hasattr(self.prompt_learner.dynamic_optimizer, 'class_adaptive_factors')):
                    caf = self.prompt_learner.dynamic_optimizer.class_adaptive_factors
                    debug_info["class_adaptive_factors_mean"] = caf.mean().item()
                    debug_info["class_adaptive_factors_std"] = caf.std().item()

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

                debug_info["mode"] = "zero_shot"

            # Top-K
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

            results = []
            for prob, idx in zip(top_probs, top_indices):
                breed_name = self.classnames[idx.item()]
                results.append({
                    "breed": breed_name.replace('_', ' '),
                    "prompt": prompt_labels[idx.item()],
                    "probability": round(prob.item(), 4),
                    "attributes": self.breed_db.get_attributes(breed_name.replace('_', ' ').title())
                })

            # 添加所有类别的概率分布（用于可视化）
            all_probs = probs.cpu().tolist()
            debug_info["all_probabilities"] = {
                self.classnames[i].replace('_', ' '): round(p, 4)
                for i, p in enumerate(all_probs)
            }
            debug_info["top1_confidence"] = round(top_probs[0].item(), 4)
            debug_info["entropy"] = round((-probs * torch.log(probs + 1e-10)).sum().item(), 4)

            return results, debug_info

    def predict_zero_shot(self, image_tensor, top_k=5, prompt_template=None):
        """Zero-shot CLIP 预测（用于对比）"""
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor.float())
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

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

            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

            results = []
            for prob, idx in zip(top_probs, top_indices):
                breed_name = self.classnames[idx.item()]
                results.append({
                    "breed": breed_name.replace('_', ' '),
                    "probability": round(prob.item(), 4),
                    "attributes": self.breed_db.get_attributes(breed_name.replace('_', ' ').title())
                })

            return results


# ============ 全局分类器实例 ============
classifier = PetClassifierAPI()


# ============ API 路由 ============

@app.route('/')
def index():
    """主页 - 返回前端页面"""
    return send_from_directory('static', 'index.html')


@app.route('/api/classify', methods=['POST'])
def classify():
    """分类API - 使用当前加载的模型（动态提示词或zero-shot）"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_bytes = request.files['image'].read()
        top_k = int(request.form.get('top_k', 5))
        prompt_template = request.form.get('prompt_template', None)
        compare = request.form.get('compare', 'false').lower() == 'true'

        image, image_tensor = classifier.preprocess(image_bytes)
        results, debug_info = classifier.predict(image_tensor, top_k, prompt_template)

        response = {
            "success": True,
            "predictions": results,
            "mode": debug_info.get("mode", "unknown"),
            "model_type": "DynamicPromptTrainer" if classifier.use_trained_model else "Zero-shot CLIP",
            "shot": classifier.shot if classifier.use_trained_model else None,
            "debug_info": debug_info
        }

        # 如果请求对比模式，同时返回 zero-shot 结果
        if compare and classifier.use_trained_model:
            zero_shot_results = classifier.predict_zero_shot(image_tensor, top_k, prompt_template)
            response["zero_shot_predictions"] = zero_shot_results
            response["comparison"] = {
                "dynamic_top1": results[0]["breed"],
                "dynamic_confidence": results[0]["probability"],
                "zero_shot_top1": zero_shot_results[0]["breed"],
                "zero_shot_confidence": zero_shot_results[0]["probability"],
                "agreement": results[0]["breed"] == zero_shot_results[0]["breed"]
            }

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/compare', methods=['POST'])
def compare_models():
    """
    多模型对比API
    同时用 Zero-shot CLIP 和 DynamicPrompt 推理同一张图片
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_bytes = request.files['image'].read()
        top_k = int(request.form.get('top_k', 5))

        image, image_tensor = classifier.preprocess(image_bytes)

        # DynamicPrompt 推理
        dynamic_results, dynamic_debug = classifier.predict(image_tensor, top_k)

        # Zero-shot CLIP 推理
        zero_shot_results = classifier.predict_zero_shot(image_tensor, top_k)

        # 计算差异
        dynamic_top5 = [r["breed"] for r in dynamic_results]
        zero_shot_top5 = [r["breed"] for r in zero_shot_results]

        # 找出两种方法预测不同的品种
        dynamic_probs = {r["breed"]: r["probability"] for r in dynamic_results}
        zero_shot_probs = {r["breed"]: r["probability"] for r in zero_shot_results}

        differences = []
        for breed in set(list(dynamic_probs.keys()) + list(zero_shot_probs.keys())):
            dp = dynamic_probs.get(breed, 0)
            zp = zero_shot_probs.get(breed, 0)
            if abs(dp - zp) > 0.01:
                differences.append({
                    "breed": breed,
                    "dynamic_prob": dp,
                    "zero_shot_prob": zp,
                    "difference": round(dp - zp, 4)
                })
        differences.sort(key=lambda x: abs(x["difference"]), reverse=True)

        return jsonify({
            "success": True,
            "dynamic_prompt": {
                "predictions": dynamic_results,
                "debug_info": dynamic_debug
            },
            "zero_shot": {
                "predictions": zero_shot_results
            },
            "comparison": {
                "dynamic_top1": dynamic_results[0]["breed"],
                "dynamic_confidence": dynamic_results[0]["probability"],
                "zero_shot_top1": zero_shot_results[0]["breed"],
                "zero_shot_confidence": zero_shot_results[0]["probability"],
                "agreement": dynamic_results[0]["breed"] == zero_shot_results[0]["breed"],
                "top5_overlap": len(set(dynamic_top5) & set(zero_shot_top5)),
                "probability_differences": differences[:10]
            }
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/breed/<breed_name>', methods=['GET'])
def get_breed_info(breed_name):
    """获取品种详细信息"""
    try:
        # 将 URL 中的名称转换为标准格式
        breed_name = breed_name.replace('_', ' ').title()
        attributes = classifier.breed_db.get_attributes(breed_name)

        if not attributes:
            return jsonify({"error": f"Breed '{breed_name}' not found"}), 404

        # 生成不同模板类型的提示词
        prompts = classifier.breed_db.get_prompts(breed_name, template_type="detailed")

        return jsonify({
            "success": True,
            "breed": breed_name,
            "attributes": attributes,
            "prompts": prompts,
            "type": attributes.get("type", "unknown")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/breeds', methods=['GET'])
def get_all_breeds():
    """获取所有品种列表及属性"""
    try:
        breeds = []
        for classname in classifier.classnames:
            breed_name = classname.replace('_', ' ').title()
            attr = classifier.breed_db.get_attributes(breed_name)
            breeds.append({
                "name": breed_name,
                "type": attr.get("type", "unknown") if attr else "unknown",
                "has_attributes": attr is not None
            })

        return jsonify({
            "success": True,
            "breeds": breeds,
            "count": len(breeds),
            "cats": sum(1 for b in breeds if b["type"] == "cat"),
            "dogs": sum(1 for b in breeds if b["type"] == "dog")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/experiments', methods=['GET'])
def get_experiment_results():
    """获取实验结果摘要"""
    try:
        return jsonify({
            "success": True,
            "experiments": classifier.experiment_summary,
            "comparison_results": _load_comparison_results()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _load_comparison_results():
    """加载模型对比结果"""
    comparison_path = os.path.join(FGDC_PATH, "comparison_results", "comparison_summary.json")
    if os.path.exists(comparison_path):
        with open(comparison_path, 'r') as f:
            return json.load(f)
    return {}


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "device": classifier.device,
        "model_loaded": classifier.use_trained_model,
        "model_type": "DynamicPromptTrainer" if classifier.use_trained_model else "Zero-shot CLIP",
        "shot": classifier.shot if classifier.use_trained_model else None
    })


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """获取当前模型信息"""
    info = {
        "model_type": "DynamicPromptTrainer" if classifier.use_trained_model else "Zero-shot CLIP",
        "device": classifier.device,
        "backbone": "RN50",
        "num_classes": len(classifier.classnames),
        "classnames": [c.replace('_', ' ') for c in classifier.classnames],
        "shot": classifier.shot if classifier.use_trained_model else None
    }

    if classifier.use_trained_model and classifier.prompt_learner is not None:
        pl = classifier.prompt_learner
        info["prompt_learner"] = {
            "n_ctx": pl.n_ctx,
            "ctx_shape": list(pl.ctx.shape),
            "has_dynamic_optimizer": pl.dynamic_optimizer is not None,
            "has_soft_prompt_adapter": pl.soft_prompt_adapter is not None,
            "ctx_norm": pl.ctx.norm().item()
        }

        if pl.dynamic_optimizer is not None:
            info["prompt_learner"]["class_adaptive_factors_mean"] = \
                pl.dynamic_optimizer.class_adaptive_factors.mean().item()

    return jsonify({"success": True, "info": info})


# ============ 静态文件服务 ============

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件"""
    return send_from_directory('static', filename)


# ============ 主程序 ============

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pet Classification API (Research Edition)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port to run the API server')
    parser.add_argument('--shot', type=str, default="16",
                        choices=["1", "2", "4", "8", "16"],
                        help='Few-shot number for model selection')
    args = parser.parse_args()

    # 自动模型路径
    if args.model is None:
        epoch_map = {'1': '100', '2': '80', '4': '60', '8': '40', '16': '20'}
        epoch_str = epoch_map[args.shot]
        auto_path = os.path.join(
            FGDC_PATH,
            f"output_fgd/oxford_pets/DynamicPromptTrainer/shots_{args.shot}/seed_1/prompt_learner/model.pth.tar-{epoch_str}"
        )
        if os.path.exists(auto_path):
            args.model = auto_path
            print(f"Auto-selected model: {auto_path}")

    print("="*60)
    print("Fine-Grained Pet Classification API - Research Edition")
    print("="*60)
    if args.model:
        print(f"Model path: {args.model}")
        print(f"Shot: {args.shot}")
    else:
        print("Mode: Zero-shot CLIP")
    print("="*60)

    classifier = PetClassifierAPI(model_path=args.model, shot=args.shot)

    print(f"\nStarting API server on http://localhost:{args.port}")
    print(f"API Documentation:")
    print(f"  POST /api/classify      - 单图分类")
    print(f"  POST /api/compare       - 多模型对比")
    print(f"  GET  /api/breeds        - 品种列表")
    print(f"  GET  /api/breed/<name>  - 品种详情")
    print(f"  GET  /api/experiments   - 实验结果")
    print(f"  GET  /api/model-info    - 模型信息")
    print(f"  GET  /api/health        - 健康检查")
    print("="*60)

    app.run(host='0.0.0.0', port=args.port, debug=False)
