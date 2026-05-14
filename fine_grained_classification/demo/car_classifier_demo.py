"""
细粒度汽车分类可视化演示系统 - 研究增强版
Streamlit Demo for Fine-Grained Car Classification with Dynamic Prompts
"""
import os
import sys
import io

# 禁用 Streamlit 遥测（避免发送到 segment.io）
os.environ['STREAMLIT_TELEMETRY'] = '0'

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 添加CoOp的clip模块路径
COOP_PATH = os.path.join(PROJECT_ROOT, "CoOp")
sys.path.insert(0, COOP_PATH)
import clip

# 添加 fine_grained_classification 路径
FGDC_PATH = os.path.join(PROJECT_ROOT, "fine_grained_classification")
sys.path.insert(0, FGDC_PATH)

# 导入CoOp数据集模块（注册数据集到DATASET_REGISTRY）
import datasets.stanford_cars  # noqa: F401


class CarClassifierDemo:
    """汽车分类演示系统 - 研究增强版"""

    def __init__(self, use_api=False, api_url="http://localhost:5001"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model = None
        self.classnames = None
        self.use_api = use_api
        self.api_url = api_url

        # 类别名称（Stanford Cars 196类，与数据集严格一致）
        # 注意：实际使用时应该从训练好的模型或数据集中加载类别名称
        self.stanford_cars_classes = self._load_car_classes()

        if not use_api:
            self.load_model()

    def _load_car_classes(self):
        """加载Stanford Cars类别列表"""
        # 尝试从CoOp数据集加载
        try:
            from dassl.data.datasets import DATASET_REGISTRY
            from dassl.config import get_cfg_default
            from yacs.config import CfgNode as CN
            cfg = get_cfg_default()
            cfg.DATASET.NAME = "StanfordCars"
            cfg.DATASET.ROOT = os.path.join(PROJECT_ROOT, "data")
            # 添加缺失的 SUBSAMPLE_CLASSES 配置
            cfg.DATASET.SUBSAMPLE_CLASSES = "all"
            # 这里仅获取类别名称，不加载完整数据集
            dataset_class = DATASET_REGISTRY.get("StanfordCars")
            if dataset_class:
                # 创建一个最小配置来实例化数据集
                cfg.defrost()
                cfg.DATASET.NUM_SHOTS = 16
                cfg.SEED = 1
                cfg.freeze()
                dataset = dataset_class(cfg)
                return dataset.classnames
        except Exception as e:
            print(f"无法从数据集加载类别名称: {e}")

        # 如果无法加载，返回一个简化列表（实际使用时应确保完整）
        return [
            "AM General Hummer SUV 2000",
            "Acura RL Sedan 2012",
            "Acura TL Sedan 2012",
            "Acura TL Type-S 2008",
            "Acura TSX Sedan 2012",
            "Acura Integra Type R 2001",
            "Acura ZDX Hatchback 2012",
            "Aston Martin V8 Vantage Convertible 2012",
            "Aston Martin V8 Vantage Coupe 2012",
            "Aston Martin Virage Convertible 2012",
            "Aston Martin Virage Coupe 2012",
            "Audi RS 4 Convertible 2008",
            "Audi A5 Coupe 2012",
            "Audi TTS Coupe 2012",
            "Audi R8 Coupe 2012",
            "Audi V8 Sedan 1994",
            "Audi 100 Sedan 1994",
            "Audi 100 Wagon 1994",
            "Audi TT Hatchback 2011",
            "Audi S6 Sedan 2011",
            "Audi S5 Convertible 2012",
            "Audi S5 Coupe 2012",
            "Audi S4 Sedan 2012",
            "Audi S4 Sedan 2007",
            "Audi TT RS Coupe 2012",
            "BMW ActiveHybrid 5 Sedan 2012",
            "BMW 1 Series Convertible 2012",
            "BMW 1 Series Coupe 2012",
            "BMW 3 Series Sedan 2012",
            "BMW 3 Series Wagon 2012",
            "BMW 6 Series Convertible 2007",
            "BMW X5 SUV 2007",
            "BMW X6 SUV 2012",
            "BMW M3 Coupe 2012",
            "BMW M5 Sedan 2010",
            "BMW M6 Convertible 2010",
            "BMW Z4 Convertible 2012",
            "Bentley Arctic GT Convertible 2012",
            "Bentley Arctic GT Coupe 2012",
            "Bentley Mulsanne Sedan 2011",
            "Bentley Continental GT Coupe 2007",
            "Bentley Continental Supersports Conv. Convertible 2012",
            "Bugatti Veyron 16.4 Convertible 2009",
            "Bugatti Veyron 16.4 Coupe 2009",
            "Buick Regal GS 2012",
            "Buick Rainier SUV 2007",
            "Buick Verano Sedan 2012",
            "Buick Enclave SUV 2012",
            "Cadillac CTS-V Sedan 2012",
            "Cadillac SRX SUV 2012",
            "Cadillac Escalade EXT Crew Cab 2007",
            "Chevrolet Silverado 1500 Hybrid Crew Cab 2012",
            "Chevrolet Corvette Convertible 2012",
            "Chevrolet Corvette ZR1 2012",
            "Chevrolet Corvette Ron Fellows Edition Z06 2007",
            "Chevrolet Traverse SUV 2012",
            "Chevrolet Camaro Convertible 2012",
            "Chevrolet HHR SS 2010",
            "Chevrolet Impala Sedan 2007",
            "Chevrolet Tahoe Hybrid SUV 2012",
            "Chevrolet Sonic Sedan 2012",
            "Chevrolet Express Cargo Van 2007",
            "Chevrolet Avalanche Crew Cab 2012",
            "Chevrolet Cobalt SS 2010",
            "Chevrolet Malibu Hybrid Sedan 2010",
            "Chevrolet TrailBlazer SS 2009",
            "Chevrolet Silverado 1500 Regular Cab 2012",
            "Chevrolet Silverado 1500 Crew Cab 2007",
            "Chevrolet Silverado 2500HD Regular Cab 2012",
            "Chevrolet Spark Hatchback 2012",
            "Chevrolet Equinox SUV 2010",
            "Chevrolet Camaro Coupe 2012",
            "Chevrolet Cruze Sedan 2012",
            "Chrysler PT Cruiser Convertible 2008",
            "Chrysler 300 SRT-8 2010",
            "Chrysler Crossfire Convertible 2008",
            "Chrysler Sebring Convertible 2010",
            "Chrysler Town and Country Minivan 2012",
            "Chrysler 300 Sedan 2012",
            "Chrysler Aspen SUV 2009",
            "Daewoo Nubira Wagon 2002",
            "Dodge Caliber Wagon 2012",
            "Dodge Caliber Wagon 2007",
            "Dodge Caravan Minivan 1997",
            "Dodge Ram Pickup 3500 Crew Cab 2010",
            "Dodge Ram Pickup 3500 Quad Cab 2009",
            "Dodge Sprinter Cargo Van 2009",
            "Dodge Journey SUV 2012",
            "Dodge Dakota Crew Cab 2010",
            "Dodge Dakota Club Cab 2007",
            "Dodge Magnum Wagon 2008",
            "Dodge Challenger SRT8 2011",
            "Dodge Durango SUV 2012",
            "Dodge Durango SUV 2007",
            "Dodge Charger Sedan 2012",
            "Dodge Charger SRT-8 2009",
            "Eagle Talon Hatchback 1998",
            "FIAT 500 Abarth 2012",
            "FIAT 500 Convertible 2012",
            "Ferrari FF Coupe 2012",
            "Ferrari California Convertible 2012",
            "Ferrari 458 Italia Convertible 2012",
            "Ferrari 458 Italia Coupe 2012",
            "Fisker Karma Sedan 2012",
            "Ford F-450 Super Duty Crew Cab 2012",
            "Ford Mustang Convertible 2007",
            "Ford Focus Sedan 2007",
            "Ford Focus Hatchback 2012",
            "Ford E-Series Wagon Van 2012",
            "Ford Fiesta Sedan 2012",
            "Ford Ranger SuperCab 2011",
            "Ford GT Coupe 2006",
            "Ford F-150 Regular Cab 2012",
            "Ford F-150 Regular Cab 2007",
            "Ford Fusion Sedan 2012",
            "Ford Escape SUV 2009",
            "Ford Edge SUV 2007",
            "Ford Expedition EL SUV 2009",
            "Ford Explorer SUV 2012",
            "GMC Yukon Hybrid SUV 2012",
            "GMC Acadia SUV 2012",
            "GMC Terrain SUV 2012",
            "GMC Savana Van 2012",
            "GMC Canyon Extended Cab 2012",
            "Geo Metro Convertible 1993",
            "HUMMER H3T Crew Cab 2010",
            "HUMMER H2 SUT Crew Cab 2009",
            "Honda Odyssey Minivan 2012",
            "Honda Odyssey Minivan 2007",
            "Honda Accord Coupe 2012",
            "Honda Accord Sedan 2012",
            "Honda Pilot SUV 2012",
            "Hyundai Veloster Hatchback 2012",
            "Hyundai Santa Fe SUV 2012",
            "Hyundai Tucson SUV 2012",
            "Hyundai Veracruz SUV 2012",
            "Hyundai Sonata Hybrid Sedan 2012",
            "Hyundai Elantra Sedan 2007",
            "Hyundai Accent Sedan 2012",
            "Hyundai Genesis Sedan 2012",
            "Hyundai Sonata Sedan 2012",
            "Hyundai Elantra Touring Hatchback 2012",
            "Hyundai Azera Sedan 2012",
            "Infiniti G Coupe IPL 2012",
            "Infiniti QX56 SUV 2011",
            "Isuzu Ascender SUV 2008",
            "Jaguar XK XKR 2012",
            "Jeep Patriot SUV 2012",
            "Jeep Wrangler SUV 2012",
            "Jeep Liberty SUV 2012",
            "Jeep Grand Cherokee SUV 2012",
            "Jeep Compass SUV 2012",
            "Lamborghini Reventon Coupe 2008",
            "Lamborghini Aventador Coupe 2012",
            "Lamborghini Gallardo LP 570-4 Superleggera 2012",
            "Lamborghini Diablo Coupe 2001",
            "Land Rover Range Rover SUV 2012",
            "Land Rover LR2 SUV 2012",
            "Lincoln Town Car Sedan 2011",
            "MINI Cooper Roadster Convertible 2012",
            "Maybach Landaulet Convertible 2012",
            "Mazda Tribute SUV 2011",
            "McLaren MP4-12C Coupe 2012",
            "Mercedes-Benz 300-Class Convertible 1993",
            "Mercedes-Benz C-Class Sedan 2012",
            "Mercedes-Benz SL-Class Coupe 2009",
            "Mercedes-Benz E-Class Sedan 2012",
            "Mercedes-Benz S-Class Sedan 2009",
            "Mercedes-Benz Sprinter Van 2012",
            "Mitsubishi Lancer Sedan 2012",
            "Nissan Leaf Hatchback 2012",
            "Nissan NV Passenger Van 2012",
            "Nissan Juke Hatchback 2012",
            "Nissan 240SX Coupe 1998",
            "Plymouth Neon Coupe 1999",
            "Porsche Panamera Sedan 2012",
            "Ram C/V Cargo Van Minivan 2012",
            "Rolls-Royce Phantom Drophead Coupe Convertible 2012",
            "Rolls-Royce Ghost Sedan 2012",
            "Rolls-Royce Phantom Sedan 2012",
            "Scion xD Hatchback 2012",
            "Spyker C8 Convertible 2009",
            "Spyker C8 Coupe 2009",
            "Suzuki Aerio Sedan 2007",
            "Suzuki Kizashi Sedan 2012",
            "Suzuki SX4 Hatchback 2012",
            "Suzuki SX4 Sedan 2012",
            "Tesla Model S Sedan 2012",
            "Toyota Sequoia SUV 2012",
            "Toyota Camry Sedan 2012",
            "Toyota Corolla Sedan 2012",
            "Toyota 4Runner SUV 2012",
            "Volkswagen Beetle Hatchback 2012",
            "Volkswagen Golf Hatchback 2012",
            "Volkswagen Golf Hatchback 1991",
            "Volkswagen CC Sedan 2012",
            "Volkswagen Rabbit Hatchback 2006",
            "Volvo C30 Hatchback 2012",
            "Volvo 240 Sedan 1993",
            "Volvo XC90 SUV 2007",
            "smart fortwo Convertible 2012",
        ]

    def load_model(self):
        """加载CLIP模型"""
        try:
            self.clip_model, self.preprocess = clip.load("RN50", device=self.device)
            self.clip_model.eval()
            self.clip_model = self.clip_model.float()
            self.classnames = self.stanford_cars_classes
            print(f"模型加载完成，使用设备: {self.device}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e

    def classify_via_api(self, image, top_k=5, prompt_template=None, compare=False):
        """
        通过 API 进行分类
        Args:
            image: PIL Image
            top_k: 返回Top-K结果
            prompt_template: 自定义提示词模板
            compare: 是否同时返回 zero-shot 对比结果
        """
        import requests

        # 将图片转为 bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        files = {'image': ('image.jpg', img_byte_arr, 'image/jpeg')}
        data = {
            'top_k': top_k,
            'prompt_template': prompt_template if prompt_template else '',
            'compare': 'true' if compare else 'false'
        }

        response = requests.post(f"{self.api_url}/api/classify", files=files, data=data)
        response.raise_for_status()

        result = response.json()
        return result

    def preprocess_image(self, image):
        """预处理图像"""
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        return image_tensor

    def encode_images(self, image_tensor):
        """编码图像"""
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_texts(self, texts):
        """编码文本"""
        with torch.no_grad():
            tokens = clip.tokenize(texts).to(self.device)
            text_features = self.clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def classify(self, image, use_prompts=True, custom_prompt_template=None):
        """本地分类"""
        image_tensor = self.preprocess_image(image)
        image_features = self.encode_images(image_tensor)

        if custom_prompt_template and "{cls}" in custom_prompt_template:
            prompts = [custom_prompt_template.format(cls=name) for name in self.classnames]
        elif use_prompts:
            prompts = [f"a photo of a {name}" for name in self.classnames]
        else:
            prompts = self.classnames

        text_features = self.encode_texts(prompts)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.T).squeeze()
        probs = logits.softmax(dim=-1)

        return probs, prompts

    def get_top_predictions(self, probs, prompts, top_k=5):
        """获取Top-K预测"""
        if isinstance(probs, list):
            # 来自 API 的结果
            return probs[:top_k]

        top_probs, top_indices = torch.topk(probs, top_k)

        results = []
        for i in range(top_k):
            car_model = self.classnames[top_indices[i].item()]
            prompt = prompts[top_indices[i].item()]
            prob = top_probs[i].item()
            results.append({
                'car_model': car_model,
                'prompt': prompt,
                'probability': prob
            })
        return results

    def visualize_results(self, image, results, title='Top-5 Predictions'):
        """可视化结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')

        car_models = [r['car_model'] for r in results]
        probs = [r['probability'] for r in results]
        y_pos = np.arange(len(results))

        bars = axes[1].barh(y_pos, probs, color='steelblue')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(car_models, fontsize=9)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Probability', fontsize=12)
        axes[1].set_title(title, fontsize=14, fontweight='bold')
        axes[1].set_xlim(0, 1)

        for bar, prob in zip(bars, probs):
            width = bar.get_width()
            axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{prob:.2%}', ha='left', va='center', fontsize=10)

        plt.tight_layout()
        return fig

    def visualize_comparison(self, image, dynamic_results, zero_shot_results):
        """可视化对比结果"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 原始图片
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')

        # DynamicPrompt 结果
        breeds_d = [r['car_model'] for r in dynamic_results]
        probs_d = [r['probability'] for r in dynamic_results]
        y_pos = np.arange(len(dynamic_results))
        bars = axes[1].barh(y_pos, probs_d, color='#667eea')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(breeds_d, fontsize=9)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Probability')
        axes[1].set_title('DynamicPrompt', fontsize=14, fontweight='bold')
        axes[1].set_xlim(0, 1)

        # Zero-shot 结果
        breeds_z = [r['car_model'] for r in zero_shot_results]
        probs_z = [r['probability'] for r in zero_shot_results]
        bars = axes[2].barh(y_pos, probs_z, color='#764ba2')
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(breeds_z, fontsize=9)
        axes[2].invert_yaxis()
        axes[2].set_xlabel('Probability')
        axes[2].set_title('Zero-shot CLIP', fontsize=14, fontweight='bold')
        axes[2].set_xlim(0, 1)

        plt.tight_layout()
        return fig


def load_experiment_summary():
    """加载实验结果摘要"""
    summary_path = os.path.join(FGDC_PATH, "output_fgd", "stanford_cars", "experiment_summary.json")
    if os.path.exists(summary_path):
        import json
        with open(summary_path, 'r') as f:
            return json.load(f)
    return {}


def main():
    """主函数"""
    import streamlit as st

    st.set_page_config(
        page_title="细粒度汽车分类系统 - 研究版",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 加载实验数据
    experiment_data = load_experiment_summary()

    # 侧边栏
    st.sidebar.title("⚙️ 设置")

    # 选择运行模式
    run_mode = st.sidebar.radio(
        "运行模式",
        ["API服务", "本地模型"],
        help="选择使用本地模型还是调用API服务"
    )

    api_url = "http://localhost:5001"
    use_api = (run_mode == "API服务")

    if use_api:
        api_url = st.sidebar.text_input("API地址", value=api_url)
        st.sidebar.info("API服务模式：调用后端API进行分类")

        # 检查API是否可用
        try:
            import requests
            health = requests.get(f"{api_url}/api/health", timeout=2)
            if health.status_code == 200:
                health_data = health.json()
                st.sidebar.success(f"✅ API已连接 | 设备: {health_data.get('device', 'unknown')} | 模型: {health_data.get('model_type', 'unknown')}")
            else:
                st.sidebar.warning("⚠️ API服务响应异常")
        except Exception as e:
            st.sidebar.error(f"❌ API未连接: {e}")
            st.sidebar.info("请先启动API: cd web && python app.py")

    # 加载分类器
    @st.cache_resource
    def load_classifier(use_api=False, api_url="http://localhost:5001"):
        return CarClassifierDemo(use_api=use_api, api_url=api_url)

    classifier = load_classifier(use_api, api_url)

    # 主页面
    st.title("🚗 细粒度汽车型号分类系统")
    st.markdown("基于 **CLIP + 动态提示词优化** 的细粒度汽车分类研究演示")

    # 研究介绍标签页
    tab_classify, tab_research, tab_models, tab_api = st.tabs([
        "🔍 分类演示", "📊 研究结果", "📚 车型知识库", "🔌 API文档"
    ])

    # ======== 分类演示标签页 ========
    with tab_classify:
        st.subheader("📤 上传图片")

        # 动态提示词说明
        with st.container():
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%);
                        border: 2px solid #667eea;
                        border-radius: 12px;
                        padding: 20px;
                        margin-bottom: 20px;">
                <h4 style="color: #667eea; margin-bottom: 10px;">🤖 动态提示词模式</h4>
                <p>本系统使用自动生成图片专属的动态提示词，无需手动输入。</p>
                <p style="color: #666; font-size: 0.9em; margin-top: 10px;">
                提示词会根据图片内容自动调整，每张图片的提示词都是独特的。</p>
            </div>
            """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("上传汽车图片进行分类", type=['jpg', 'png', 'jpeg'])

        col1, col2 = st.columns([1, 2])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')

            with col1:
                st.image(image, caption="上传图片", use_container_width=True)

            with col2:
                top_k = st.slider("显示Top-K预测", 3, 10, 5)
                compare_mode = st.checkbox("对比 Zero-shot CLIP", value=True,
                                          help="同时显示 DynamicPrompt 和 Zero-shot 的结果对比")

                if st.button("🔍 开始识别", type="primary", use_container_width=True):
                    with st.spinner("正在识别..."):
                        try:
                            if use_api:
                                # 调用 API
                                result = classifier.classify_via_api(
                                    image, top_k, compare=compare_mode
                                )

                                if result.get('success'):
                                    predictions = result.get('predictions', [])

                                    # 显示结果
                                    st.subheader("🎯 预测结果")
                                    st.caption(f"模式: {result.get('model_type', 'unknown')} | "
                                              f"Shot: {result.get('shot', 'N/A')}-shot")

                                    for i, pred in enumerate(predictions):
                                        col_model, col_prob = st.columns([3, 1])
                                        with col_model:
                                            emoji = "👑" if i == 0 else f"{i+1}."
                                            st.write(f"{emoji} **{pred['car_model']}**")
                                        with col_prob:
                                            st.progress(pred['probability'],
                                                       text=f"{pred['probability']:.1%}")

                                    # 对比结果
                                    if compare_mode and result.get('comparison'):
                                        comp = result['comparison']
                                        st.subheader("📊 与 Zero-shot CLIP 对比")
                                        c1, c2, c3 = st.columns(3)
                                        with c1:
                                            st.metric("DynamicPrompt Top-1",
                                                     comp['dynamic_top1'],
                                                     f"{comp['dynamic_confidence']:.1%}")
                                        with c2:
                                            st.metric("Zero-shot Top-1",
                                                     comp['zero_shot_top1'],
                                                     f"{comp['zero_shot_confidence']:.1%}")
                                        with c3:
                                            agree = "✅ 一致" if comp['agreement'] else "⚠️ 不同"
                                            st.metric("预测一致性", agree)

                                    # 调试信息
                                    if result.get('debug_info'):
                                        with st.expander("🔧 调试信息"):
                                            st.json(result['debug_info'])
                                else:
                                    st.error(f"识别失败: {result.get('error', '未知错误')}")
                            else:
                                # 本地分类（使用默认提示词）
                                probs, prompts = classifier.classify(
                                    image,
                                    use_prompts=True,
                                    custom_prompt_template=None
                                )
                                results = classifier.get_top_predictions(probs, prompts, top_k)

                                st.subheader("🎯 预测结果")
                                for i, result in enumerate(results):
                                    col_model, col_prob = st.columns([3, 1])
                                    with col_model:
                                        emoji = "👑" if i == 0 else f"{i+1}."
                                        st.write(f"{emoji} **{result['car_model']}**")
                                    with col_prob:
                                        st.progress(result['probability'],
                                                   text=f"{result['probability']:.1%}")

                        except Exception as e:
                            st.error(f"识别失败: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

    # ======== 研究结果标签页 ========
    with tab_research:
        st.subheader("📈 实验结果")

        if experiment_data and experiment_data.get('results'):
            results = experiment_data['results']
            shots = ['1-shot', '2-shot', '4-shot', '8-shot', '16-shot']

            # 准确率对比表
            st.write("**不同 Few-shot 设置下的准确率对比**")
            table_data = []
            for method in ['CoOp', 'CoCoOp', 'DynamicPromptTrainer']:
                row = {"方法": method}
                for shot in shots:
                    acc = results.get(method, {}).get(shot, {}).get('accuracy', 'N/A')
                    row[shot] = f"{acc}%" if acc != 'N/A' else 'N/A'
                table_data.append(row)

            st.dataframe(table_data, use_container_width=True)

            # 与 CoCoOp 的对比
            if experiment_data.get('comparison_vs_cocoop'):
                st.write("**DynamicPrompt vs CoCoOp 差异**")
                comp_data = []
                for shot in shots:
                    diff = experiment_data['comparison_vs_cocoop'].get(shot, 0)
                    our_acc = results.get('DynamicPromptTrainer', {}).get(shot, {}).get('accuracy', 0)
                    comp_data.append({
                        "Shot": shot,
                        "DynamicPrompt": f"{our_acc}%",
                        "vs CoCoOp": f"{'+' if diff > 0 else ''}{diff}%",
                        "优势": "✅" if diff > 0 else "⚠️"
                    })
                st.dataframe(comp_data, use_container_width=True)

            # 训练策略
            if experiment_data.get('epoch_config'):
                st.write("**自适应 Epoch 策略**")
                epoch_data = []
                for shot, epochs in experiment_data['epoch_config'].items():
                    epoch_data.append({"Shot": shot, "Epochs": epochs})
                st.dataframe(epoch_data, use_container_width=True)
        else:
            st.info("暂无实验数据，请先运行训练脚本")

        # 方法对比
        st.subheader("🔬 方法对比")
        comparison_data = {
            "特性": ["提示词类型", "核心参数", "是否感知图像", "是否感知难度", "损失函数", "提示词层数"],
            "CoOp": ["静态可学习 ctx", "ctx", "否", "否", "标准 CE", "单层静态"],
            "CoCoOp": ["图像条件偏移", "ctx + meta_net", "是", "否", "标准 CE", "单层偏移"],
            "DynamicPrompt (Ours)": ["图像条件偏移 + 可学习难度加权",
                                    "ctx + SoftPromptAdapter + DifficultyWeightCalculator + class_adaptive_factors",
                                    "是", "是（可学习）", "加权 CE", "双层（MLP 偏移 + 类别自适应因子）"]
        }
        st.dataframe(comparison_data, use_container_width=True)

    # ======== 车型知识库标签页 ========
    with tab_models:
        st.subheader("📚 车型知识库")

        # 搜索
        search = st.text_input("🔍 搜索车型", placeholder="输入品牌或型号...")

        models_to_show = classifier.stanford_cars_classes
        if search:
            models_to_show = [m for m in models_to_show if search.lower() in m.lower()]

        # 显示车型卡片
        cols = st.columns(2)
        for i, car_model in enumerate(models_to_show[:50]):  # 限制显示数量
            with cols[i % 2]:
                with st.expander(f"🚗 {car_model}"):
                    st.write(f"**完整名称**: {car_model}")
                    # 提取年份
                    year = car_model.split()[0] if car_model.split()[0].isdigit() else "未知"
                    st.write(f"**年份**: {year}")
                    # 提取品牌
                    brand = car_model.split()[1] if len(car_model.split()) > 1 else "未知"
                    st.write(f"**品牌**: {brand}")

    # ======== API文档标签页 ========
    with tab_api:
        st.subheader("🔌 API 接口文档")

        st.write("**分类接口**")
        st.code("""
POST /api/classify
Content-Type: multipart/form-data

参数:
  - image: 图片文件 (required)
  - top_k: 返回Top-K结果 (default: 5)
  - compare: 是否对比 zero-shot (default: false)

响应:
  {
    "success": true,
    "predictions": [
      {"car_model": "...", "probability": 0.95}
    ],
    "mode": "dynamic_prompt",
    "comparison": {...}  // 如果 compare=true
  }
        """, language="bash")

        st.write("**对比接口**")
        st.code("""
POST /api/compare
Content-Type: multipart/form-data

同时返回 DynamicPrompt 和 Zero-shot CLIP 的结果对比
        """, language="bash")

        st.write("**车型信息接口**")
        st.code("""
GET /api/models          # 获取所有车型
GET /api/model/<name>    # 获取车型详情
GET /api/experiments     # 获取实验结果
GET /api/model-info      # 获取模型信息
GET /api/health          # 健康检查
        """, language="bash")

    # 底部说明
    st.markdown("---")
    st.markdown("""
    ### 💡 关于本系统

    本系统展示了 **DynamicPromptTrainer** 方法在 Stanford Cars 数据集上的应用：

    1. **可学习难度加权** — 使用可学习温度参数，让模型自动学习何时关注困难样本
    2. **双层提示词调整** — SoftPromptAdapter MLP 偏移 + 类别自适应因子
    3. **两阶段前向传播** — 训练时先计算基础 logits，再用预测计算难度权重
    4. **汽车细粒度分类** — 196 种汽车型号的精确识别

    **应用场景**: 自动驾驶感知、智能交通监控、车辆型号识别
    """)


if __name__ == "__main__":
    main()
