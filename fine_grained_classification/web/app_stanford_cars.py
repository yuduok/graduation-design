"""
Web API服务 - 细粒度汽车分类（研究增强版）
Flask API for Fine-Grained Car Classification Research Demo
支持：动态提示词推理 / 多模型对比 / 车型语义信息 / 实验结果展示
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

# 导入CoOp数据集模块（注册数据集到DATASET_REGISTRY）
import datasets.stanford_cars  # noqa: F401

import clip
from clip.simple_tokenizer import SimpleTokenizer

app = Flask(__name__, static_folder='static')
CORS(app)


class CarClassifierAPI:
    """汽车分类API服务 - 研究增强版"""

    def __init__(self, model_path=None, shot="16"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.prompt_learner = None
        self.text_encoder = None
        self.classnames = None
        self.use_trained_model = model_path is not None
        self.model_path = model_path
        self.shot = shot

        # 实验结果缓存
        self.experiment_summary = self._load_experiment_summary()

        self.load_model(model_path)

    def _load_experiment_summary(self):
        """加载实验结果摘要"""
        summary_path = os.path.join(FGDC_PATH, "output_fgd", "stanford_cars", "experiment_summary.json")
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

        # Stanford Cars 196类（与数据集严格一致）
        # 实际使用时应该从训练好的模型或数据集中加载类别名称
        self.classnames = self._load_car_classes()

        # 如果提供了模型路径，加载训练好的模型
        if model_path and os.path.exists(model_path):
            print(f"Loading trained model from {model_path}...")
            self.load_trained_model(model_path)

        if self.use_trained_model:
            print(f"Using trained model: {model_path}")
        else:
            print("Using zero-shot CLIP model")
        print("Model loaded successfully!")

    def _load_car_classes(self):
        """加载Stanford Cars类别列表"""
        # 尝试从CoOp数据集加载
        try:
            from dassl.data.datasets import DATASET_REGISTRY
            from dassl.config import get_cfg_default
            cfg = get_cfg_default()
            cfg.DATASET.NAME = "StanfordCars"
            cfg.DATASET.ROOT = os.path.join(PROJECT_ROOT, "data")
            # 添加缺失的 SUBSAMPLE_CLASSES 配置
            cfg.DATASET.SUBSAMPLE_CLASSES = "all"
            dataset_class = DATASET_REGISTRY.get("StanfordCars")
            if dataset_class:
                cfg.defrost()
                cfg.DATASET.NUM_SHOTS = 16
                cfg.SEED = 1
                cfg.freeze()
                dataset = dataset_class(cfg)
                return dataset.classnames
        except Exception as e:
            print(f"无法从数据集加载类别名称: {e}")

        # 如果无法加载，返回简化列表
        return [
            "AM General Hummer SUV 2000", "Acura RL Sedan 2012", "Acura TL Sedan 2012",
            "Acura TL Type-S 2008", "Acura TSX Sedan 2012", "Acura Integra Type R 2001",
            "Acura ZDX Hatchback 2012", "Aston Martin V8 Vantage Convertible 2012",
            "Aston Martin V8 Vantage Coupe 2012", "Aston Martin Virage Convertible 2012",
            "Aston Martin Virage Coupe 2012", "Audi RS 4 Convertible 2008",
            "Audi A5 Coupe 2012", "Audi TTS Coupe 2012", "Audi R8 Coupe 2012",
            "Audi V8 Sedan 1994", "Audi 100 Sedan 1994", "Audi 100 Wagon 1994",
            "Audi TT Hatchback 2011", "Audi S6 Sedan 2011", "Audi S5 Convertible 2012",
            "Audi S5 Coupe 2012", "Audi S4 Sedan 2012", "Audi S4 Sedan 2007",
            "Audi TT RS Coupe 2012", "BMW ActiveHybrid 5 Sedan 2012",
            "BMW 1 Series Convertible 2012", "BMW 1 Series Coupe 2012",
            "BMW 3 Series Sedan 2012", "BMW 3 Series Wagon 2012",
            "BMW 6 Series Convertible 2007", "BMW X5 SUV 2007", "BMW X6 SUV 2012",
            "BMW M3 Coupe 2012", "BMW M5 Sedan 2010", "BMW M6 Convertible 2010",
            "BMW Z4 Convertible 2012", "Bentley Arctic GT Convertible 2012",
            "Bentley Arctic GT Coupe 2012", "Bentley Mulsanne Sedan 2011",
            "Bentley Continental GT Coupe 2007", "Bentley Continental Supersports Conv. Convertible 2012",
            "Bugatti Veyron 16.4 Convertible 2009", "Bugatti Veyron 16.4 Coupe 2009",
            "Buick Regal GS 2012", "Buick Rainier SUV 2007", "Buick Verano Sedan 2012",
            "Buick Enclave SUV 2012", "Cadillac CTS-V Sedan 2012", "Cadillac SRX SUV 2012",
            "Cadillac Escalade EXT Crew Cab 2007", "Chevrolet Silverado 1500 Hybrid Crew Cab 2012",
            "Chevrolet Corvette Convertible 2012", "Chevrolet Corvette ZR1 2012",
            "Chevrolet Corvette Ron Fellows Edition Z06 2007", "Chevrolet Traverse SUV 2012",
            "Chevrolet Camaro Convertible 2012", "Chevrolet HHR SS 2010",
            "Chevrolet Impala Sedan 2007", "Chevrolet Tahoe Hybrid SUV 2012",
            "Chevrolet Sonic Sedan 2012", "Chevrolet Express Cargo Van 2007",
            "Chevrolet Avalanche Crew Cab 2012", "Chevrolet Cobalt SS 2010",
            "Chevrolet Malibu Hybrid Sedan 2010", "Chevrolet TrailBlazer SS 2009",
            "Chevrolet Silverado 1500 Regular Cab 2012", "Chevrolet Silverado 1500 Crew Cab 2007",
            "Chevrolet Silverado 2500HD Regular Cab 2012", "Chevrolet Spark Hatchback 2012",
            "Chevrolet Equinox SUV 2010", "Chevrolet Camaro Coupe 2012",
            "Chevrolet Cruze Sedan 2012", "Chrysler PT Cruiser Convertible 2008",
            "Chrysler 300 SRT-8 2010", "Chrysler Crossfire Convertible 2008",
            "Chrysler Sebring Convertible 2010", "Chrysler Town and Country Minivan 2012",
            "Chrysler 300 Sedan 2012", "Chrysler Aspen SUV 2009",
            "Daewoo Nubira Wagon 2002", "Dodge Caliber Wagon 2012",
            "Dodge Caliber Wagon 2007", "Dodge Caravan Minivan 1997",
            "Dodge Ram Pickup 3500 Crew Cab 2010", "Dodge Ram Pickup 3500 Quad Cab 2009",
            "Dodge Sprinter Cargo Van 2009", "Dodge Journey SUV 2012",
            "Dodge Dakota Crew Cab 2010", "Dodge Dakota Club Cab 2007",
            "Dodge Magnum Wagon 2008", "Dodge Challenger SRT8 2011",
            "Dodge Durango SUV 2012", "Dodge Durango SUV 2007",
            "Dodge Charger Sedan 2012", "Dodge Charger SRT-8 2009",
            "Eagle Talon Hatchback 1998", "FIAT 500 Abarth 2012",
            "FIAT 500 Convertible 2012", "Ferrari FF Coupe 2012",
            "Ferrari California Convertible 2012", "Ferrari 458 Italia Convertible 2012",
            "Ferrari 458 Italia Coupe 2012", "Fisker Karma Sedan 2012",
            "Ford F-450 Super Duty Crew Cab 2012", "Ford Mustang Convertible 2007",
            "Ford Focus Sedan 2007", "Ford Focus Hatchback 2012",
            "Ford E-Series Wagon Van 2012", "Ford Fiesta Sedan 2012",
            "Ford Ranger SuperCab 2011", "Ford GT Coupe 2006",
            "Ford F-150 Regular Cab 2012", "Ford F-150 Regular Cab 2007",
            "Ford Fusion Sedan 2012", "Ford Escape SUV 2009",
            "Ford Edge SUV 2007", "Ford Expedition EL SUV 2009",
            "Ford Explorer SUV 2012", "GMC Yukon Hybrid SUV 2012",
            "GMC Acadia SUV 2012", "GMC Terrain SUV 2012",
            "GMC Savana Van 2012", "GMC Canyon Extended Cab 2012",
            "Geo Metro Convertible 1993", "HUMMER H3T Crew Cab 2010",
            "HUMMER H2 SUT Crew Cab 2009", "Honda Odyssey Minivan 2012",
            "Honda Odyssey Minivan 2007", "Honda Accord Coupe 2012",
            "Honda Accord Sedan 2012", "Honda Pilot SUV 2012",
            "Hyundai Veloster Hatchback 2012", "Hyundai Santa Fe SUV 2012",
            "Hyundai Tucson SUV 2012", "Hyundai Veracruz SUV 2012",
            "Hyundai Sonata Hybrid Sedan 2012", "Hyundai Elantra Sedan 2007",
            "Hyundai Accent Sedan 2012", "Hyundai Genesis Sedan 2012",
            "Hyundai Sonata Sedan 2012", "Hyundai Elantra Touring Hatchback 2012",
            "Hyundai Azera Sedan 2012", "Infiniti G Coupe IPL 2012",
            "Infiniti QX56 SUV 2011", "Isuzu Ascender SUV 2008",
            "Jaguar XK XKR 2012", "Jeep Patriot SUV 2012",
            "Jeep Wrangler SUV 2012", "Jeep Liberty SUV 2012",
            "Jeep Grand Cherokee SUV 2012", "Jeep Compass SUV 2012",
            "Lamborghini Reventon Coupe 2008", "Lamborghini Aventador Coupe 2012",
            "Lamborghini Gallardo LP 570-4 Superleggera 2012", "Lamborghini Diablo Coupe 2001",
            "Land Rover Range Rover SUV 2012", "Land Rover LR2 SUV 2012",
            "Lincoln Town Car Sedan 2011", "MINI Cooper Roadster Convertible 2012",
            "Maybach Landaulet Convertible 2012", "Mazda Tribute SUV 2011",
            "McLaren MP4-12C Coupe 2012", "Mercedes-Benz 300-Class Convertible 1993",
            "Mercedes-Benz C-Class Sedan 2012", "Mercedes-Benz SL-Class Coupe 2009",
            "Mercedes-Benz E-Class Sedan 2012", "Mercedes-Benz S-Class Sedan 2009",
            "Mercedes-Benz Sprinter Van 2012", "Mitsubishi Lancer Sedan 2012",
            "Nissan Leaf Hatchback 2012", "Nissan NV Passenger Van 2012",
            "Nissan Juke Hatchback 2012", "Nissan 240SX Coupe 1998",
            "Plymouth Neon Coupe 1999", "Porsche Panamera Sedan 2012",
            "Ram C/V Cargo Van Minivan 2012", "Rolls-Royce Phantom Drophead Coupe Convertible 2012",
            "Rolls-Royce Ghost Sedan 2012", "Rolls-Royce Phantom Sedan 2012",
            "Scion xD Hatchback 2012", "Spyker C8 Convertible 2009",
            "Spyker C8 Coupe 2009", "Suzuki Aerio Sedan 2007",
            "Suzuki Kizashi Sedan 2012", "Suzuki SX4 Hatchback 2012",
            "Suzuki SX4 Sedan 2012", "Tesla Model S Sedan 2012",
            "Toyota Sequoia SUV 2012", "Toyota Camry Sedan 2012",
            "Toyota Corolla Sedan 2012", "Toyota 4Runner SUV 2012",
            "Volkswagen Beetle Hatchback 2012", "Volkswagen Golf Hatchback 2012",
            "Volkswagen Golf Hatchback 1991", "Volkswagen CC Sedan 2012",
            "Volkswagen Rabbit Hatchback 2006", "Volvo C30 Hatchback 2012",
            "Volvo 240 Sedan 1993", "Volvo XC90 SUV 2007",
            "smart fortwo Convertible 2012",
        ]

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
        支持分批编码文本特征以避免显存溢出
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

                # 分批编码文本特征（避免196类显存溢出）
                chunk_size = 50
                all_text_features = []
                for i in range(0, n_cls, chunk_size):
                    end_i = min(i + chunk_size, n_cls)
                    chunk_prompts = prompts_flat[i:end_i]
                    chunk_tokenized = tokenized_prompts[i:end_i]
                    chunk_features = self.text_encoder(chunk_prompts, chunk_tokenized)
                    chunk_features = chunk_features / chunk_features.norm(dim=-1, keepdim=True)
                    all_text_features.append(chunk_features)

                    # 及时释放显存
                    del chunk_prompts, chunk_tokenized, chunk_features
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                text_features = torch.cat(all_text_features, dim=0)

                logit_scale = self.clip_model.logit_scale.exp()
                logits = logit_scale * (image_features @ text_features.T).squeeze()
                probs = logits.softmax(dim=-1)

                # 构建提示词描述
                if prompt_template and "{cls}" in prompt_template:
                    prompt_labels = [prompt_template.format(cls=name)
                                     for name in self.classnames]
                else:
                    prompt_labels = [f"[dynamic] adaptive prompt for {name}"
                                     for name in self.classnames]

                # 收集调试信息
                debug_info["mode"] = "dynamic_prompt"
                debug_info["prompt_shape"] = list(prompts.shape)
                debug_info["ctx_norm"] = self.prompt_learner.ctx.norm().item()

                # 获取 SoftPromptAdapter 的偏移信息
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
                    prompt_labels = [prompt_template.format(cls=name)
                                     for name in self.classnames]
                else:
                    prompt_labels = [f"a photo of a {name}"
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
                car_name = self.classnames[idx.item()]
                # 解析年份和品牌
                parts = car_name.split()
                year = parts[0] if parts[0].isdigit() else "Unknown"
                brand = parts[1] if len(parts) > 1 else "Unknown"
                results.append({
                    "car_model": car_name,
                    "brand": brand,
                    "year": year,
                    "prompt": prompt_labels[idx.item()],
                    "probability": round(prob.item(), 4)
                })

            # 添加所有类别的概率分布
            all_probs = probs.cpu().tolist()
            debug_info["all_probabilities"] = {
                self.classnames[i]: round(p, 4)
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
                prompt_labels = [prompt_template.format(cls=name)
                                 for name in self.classnames]
            else:
                prompt_labels = [f"a photo of a {name}"
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
                car_name = self.classnames[idx.item()]
                parts = car_name.split()
                year = parts[0] if parts[0].isdigit() else "Unknown"
                brand = parts[1] if len(parts) > 1 else "Unknown"
                results.append({
                    "car_model": car_name,
                    "brand": brand,
                    "year": year,
                    "probability": round(prob.item(), 4)
                })

            return results


# ============ 全局分类器实例 ============
classifier = CarClassifierAPI()


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
                "dynamic_top1": results[0]["car_model"],
                "dynamic_confidence": results[0]["probability"],
                "zero_shot_top1": zero_shot_results[0]["car_model"],
                "zero_shot_confidence": zero_shot_results[0]["probability"],
                "agreement": results[0]["car_model"] == zero_shot_results[0]["car_model"]
            }

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/compare', methods=['POST'])
def compare_models():
    """多模型对比API"""
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
        dynamic_probs = {r["car_model"]: r["probability"] for r in dynamic_results}
        zero_shot_probs = {r["car_model"]: r["probability"] for r in zero_shot_results}

        differences = []
        for car_model in set(list(dynamic_probs.keys()) + list(zero_shot_probs.keys())):
            dp = dynamic_probs.get(car_model, 0)
            zp = zero_shot_probs.get(car_model, 0)
            if abs(dp - zp) > 0.01:
                differences.append({
                    "car_model": car_model,
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
                "dynamic_top1": dynamic_results[0]["car_model"],
                "dynamic_confidence": dynamic_results[0]["probability"],
                "zero_shot_top1": zero_shot_results[0]["car_model"],
                "zero_shot_confidence": zero_shot_results[0]["probability"],
                "agreement": dynamic_results[0]["car_model"] == zero_shot_results[0]["car_model"],
                "probability_differences": differences[:10]
            }
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/model/<car_model>', methods=['GET'])
def get_car_info(car_model):
    """获取车型详细信息"""
    try:
        # 将 URL 中的名称转换
        car_model = car_model.replace('_', ' ')

        # 解析年份和品牌
        parts = car_model.split()
        year = parts[0] if parts and parts[0].isdigit() else "Unknown"
        brand = parts[1] if len(parts) > 1 else "Unknown"

        return jsonify({
            "success": True,
            "car_model": car_model,
            "year": year,
            "brand": brand,
            "full_name": car_model
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_all_models():
    """获取所有车型列表"""
    try:
        models = []
        for classname in classifier.classnames:
            parts = classname.split()
            year = parts[0] if parts and parts[0].isdigit() else "Unknown"
            brand = parts[1] if len(parts) > 1 else "Unknown"
            models.append({
                "name": classname,
                "year": year,
                "brand": brand
            })

        return jsonify({
            "success": True,
            "models": models,
            "count": len(models)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/experiments', methods=['GET'])
def get_experiment_results():
    """获取实验结果摘要"""
    try:
        return jsonify({
            "success": True,
            "experiments": classifier.experiment_summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "device": classifier.device,
        "model_loaded": classifier.use_trained_model,
        "model_type": "DynamicPromptTrainer" if classifier.use_trained_model else "Zero-shot CLIP",
        "shot": classifier.shot if classifier.use_trained_model else None,
        "num_classes": len(classifier.classnames)
    })


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """获取当前模型信息"""
    info = {
        "model_type": "DynamicPromptTrainer" if classifier.use_trained_model else "Zero-shot CLIP",
        "device": classifier.device,
        "backbone": "RN50",
        "num_classes": len(classifier.classnames),
        "classnames": classifier.classnames,
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

    parser = argparse.ArgumentParser(description='Car Classification API (Research Edition)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--port', type=int, default=5002,
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
            f"output_fgd/stanford_cars/DynamicPromptTrainer/shots_{args.shot}/seed_1/prompt_learner/model.pth.tar-{epoch_str}"
        )
        if os.path.exists(auto_path):
            args.model = auto_path
            print(f"Auto-selected model: {auto_path}")

    print("="*60)
    print("Fine-Grained Car Classification API - Research Edition")
    print("="*60)
    if args.model:
        print(f"Model path: {args.model}")
        print(f"Shot: {args.shot}")
    else:
        print("Mode: Zero-shot CLIP")
    print("="*60)

    classifier = CarClassifierAPI(model_path=args.model, shot=args.shot)

    print(f"\nStarting API server on http://localhost:{args.port}")
    print(f"API Documentation:")
    print(f"  POST /api/classify      - 单图分类")
    print(f"  POST /api/compare       - 多模型对比")
    print(f"  GET  /api/models        - 车型列表")
    print(f"  GET  /api/model/<name>  - 车型详情")
    print(f"  GET  /api/experiments   - 实验结果")
    print(f"  GET  /api/model-info    - 模型信息")
    print(f"  GET  /api/health        - 健康检查")
    print("="*60)

    app.run(host='0.0.0.0', port=args.port, debug=False)
