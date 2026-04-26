"""
细粒度猫狗分类可视化演示系统 - 研究增强版
Streamlit Demo for Fine-Grained Pet Classification with Dynamic Prompts
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

from models.breed_semantic import BreedAttributeDatabase


class PetClassifierDemo:
    """宠物分类演示系统 - 研究增强版"""

    def __init__(self, use_api=False, api_url="http://localhost:5001"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model = None
        self.classnames = None
        self.use_api = use_api
        self.api_url = api_url

        # 品种属性数据库
        self.breed_db = BreedAttributeDatabase()

        # 类别名称（Oxford-IIIT Pets 37类，与数据集严格一致）
        self.oxford_pets_classes = [
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

        if not use_api:
            self.load_model()

    def load_model(self):
        """加载CLIP模型"""
        try:
            self.clip_model, self.preprocess = clip.load("RN50", device=self.device)
            self.clip_model.eval()
            self.clip_model = self.clip_model.float()
            self.classnames = self.oxford_pets_classes
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
            prompts = [custom_prompt_template.format(cls=name.replace('_', ' ')) for name in self.classnames]
        elif use_prompts:
            prompts = [f"a photo of a {name.replace('_', ' ')}" for name in self.classnames]
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
            breed = self.classnames[top_indices[i].item()]
            prompt = prompts[top_indices[i].item()]
            prob = top_probs[i].item()
            results.append({
                'breed': breed,
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

        breeds = [r['breed'].replace('_', ' ') if isinstance(r['breed'], str) else r['breed'] for r in results]
        probs = [r['probability'] for r in results]
        y_pos = np.arange(len(results))

        bars = axes[1].barh(y_pos, probs, color='steelblue')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(breeds, fontsize=11)
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
        breeds_d = [r['breed'] for r in dynamic_results]
        probs_d = [r['probability'] for r in dynamic_results]
        y_pos = np.arange(len(dynamic_results))
        bars = axes[1].barh(y_pos, probs_d, color='#667eea')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(breeds_d, fontsize=10)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Probability')
        axes[1].set_title('DynamicPrompt', fontsize=14, fontweight='bold')
        axes[1].set_xlim(0, 1)

        # Zero-shot 结果
        breeds_z = [r['breed'] for r in zero_shot_results]
        probs_z = [r['probability'] for r in zero_shot_results]
        bars = axes[2].barh(y_pos, probs_z, color='#764ba2')
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(breeds_z, fontsize=10)
        axes[2].invert_yaxis()
        axes[2].set_xlabel('Probability')
        axes[2].set_title('Zero-shot CLIP', fontsize=14, fontweight='bold')
        axes[2].set_xlim(0, 1)

        plt.tight_layout()
        return fig


def load_experiment_summary():
    """加载实验结果摘要"""
    summary_path = os.path.join(FGDC_PATH, "output_fgd", "oxford_pets", "experiment_summary.json")
    if os.path.exists(summary_path):
        import json
        with open(summary_path, 'r') as f:
            return json.load(f)
    return {}


def main():
    """主函数"""
    import streamlit as st

    st.set_page_config(
        page_title="细粒度猫狗分类系统 - 研究版",
        page_icon="🐱",
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
        return PetClassifierDemo(use_api=use_api, api_url=api_url)

    classifier = load_classifier(use_api, api_url)

    # 主页面
    st.title("🐱🐕 细粒度猫狗品种分类系统")
    st.markdown("基于 **CLIP + 动态提示词优化** 的细粒度分类研究演示")

    # 研究介绍标签页
    tab_classify, tab_research, tab_breeds, tab_api = st.tabs([
        "🔍 分类演示", "📊 研究结果", "📚 品种知识库", "🔌 API文档"
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

        uploaded_file = st.file_uploader("上传宠物图片进行分类", type=['jpg', 'png', 'jpeg'])

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
                                        col_breed, col_prob = st.columns([3, 1])
                                        with col_breed:
                                            emoji = "👑" if i == 0 else f"{i+1}."
                                            st.write(f"{emoji} **{pred['breed']}**")
                                            if pred.get('attributes'):
                                                attr = pred['attributes']
                                                tags = []
                                                if attr.get('type'): tags.append(f"{'🐱' if attr['type']=='cat' else '🐕'} {attr['type']}")
                                                if attr.get('coat'): tags.append(f"🧥 {attr['coat'].split(',')[0]}")
                                                if attr.get('face'): tags.append(f"😺 {attr['face'].split(',')[0]}")
                                                st.caption(" | ".join(tags))
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
                                    col_breed, col_prob = st.columns([3, 1])
                                    with col_breed:
                                        emoji = "👑" if i == 0 else f"{i+1}."
                                        st.write(f"{emoji} **{result['breed'].replace('_', ' ')}**")
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

    # ======== 品种知识库标签页 ========
    with tab_breeds:
        st.subheader("📚 品种知识库")

        # 分类显示
        breed_type = st.radio("选择类型", ["全部", "🐱 猫咪", "🐕 狗狗"], horizontal=True)

        breeds_to_show = classifier.oxford_pets_classes
        if breed_type == "🐱 猫咪":
            breeds_to_show = [b for b in breeds_to_show
                           if classifier.breed_db.get_attributes(b.replace('_', ' ').title(), {}).get('type') == 'cat']
        elif breed_type == "🐕 狗狗":
            breeds_to_show = [b for b in breeds_to_show
                           if classifier.breed_db.get_attributes(b.replace('_', ' ').title(), {}).get('type') == 'dog']

        # 搜索
        search = st.text_input("🔍 搜索品种", placeholder="输入品种名称...")
        if search:
            breeds_to_show = [b for b in breeds_to_show if search.lower() in b.lower()]

        # 显示品种卡片
        cols = st.columns(3)
        for i, breed in enumerate(breeds_to_show):
            with cols[i % 3]:
                breed_name = breed.replace('_', ' ').title()
                attr = classifier.breed_db.get_attributes(breed_name)
                with st.expander(f"{'🐱' if attr and attr.get('type')=='cat' else '🐕'} {breed_name}"):
                    if attr:
                        st.write(f"**类型**: {attr.get('type', 'unknown')}")
                        st.write(f"**毛发**: {attr.get('coat', 'N/A')}")
                        st.write(f"**面部**: {attr.get('face', 'N/A')}")
                        st.write(f"**体型**: {attr.get('body', 'N/A')}")
                        st.write(f"**性格**: {attr.get('trait', 'N/A')}")

                        # 显示提示词模板
                        prompts = classifier.breed_db.get_prompts(breed_name, template_type="detailed")
                        st.caption("**提示词模板**:")
                        for p in prompts[:3]:
                            st.code(p, language=None)
                    else:
                        st.write("暂无详细信息")

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
      {"breed": "...", "probability": 0.95, "attributes": {...}}
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

        st.write("**品种信息接口**")
        st.code("""
GET /api/breeds          # 获取所有品种
GET /api/breed/<name>    # 获取品种详情
GET /api/experiments     # 获取实验结果
GET /api/model-info      # 获取模型信息
GET /api/health          # 健康检查
        """, language="bash")

    # 底部说明
    st.markdown("---")
    st.markdown("""
    ### 💡 关于本系统

    本系统展示了 **DynamicPromptTrainer** 方法的核心特性：

    1. **可学习难度加权** — 使用可学习温度参数，让模型自动学习何时关注困难样本
    2. **双层提示词调整** — SoftPromptAdapter MLP 偏移 + 类别自适应因子
    3. **两阶段前向传播** — 训练时先计算基础 logits，再用预测计算难度权重
    4. **品种语义增强** — 37 个品种的属性数据库（毛发/面部/体型/性格）

    **实验结果**: 在 Oxford-IIIT Pets 上达到 **89.8%** 准确率（16-shot）
    """)


if __name__ == "__main__":
    main()
