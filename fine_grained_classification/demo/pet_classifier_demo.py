"""
细粒度猫狗分类可视化演示系统
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
sys.path.insert(0, "/Users/yudu/Documents/毕业设计/CoOp")
import clip


class PetClassifierDemo:
    """宠物分类演示系统"""
    
    def __init__(self, use_api=False, api_url="http://localhost:5001"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model = None
        self.classnames = None
        self.use_api = use_api
        self.api_url = api_url
        
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
    
    def classify_via_api(self, image, top_k=5, prompt_template=None):
        """
        通过 API 进行分类
        Args:
            image: PIL Image
            top_k: 返回Top-K结果
            prompt_template: 自定义提示词模板
        """
        import requests
        
        # 将图片转为 bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        files = {'image': ('image.jpg', img_byte_arr, 'image/jpeg')}
        data = {
            'top_k': top_k,
            'prompt_template': prompt_template if prompt_template else ''
        }
        
        response = requests.post(f"{self.api_url}/api/classify", files=files, data=data)
        response.raise_for_status()
        
        result = response.json()
        return result.get('predictions', [])
    
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
    
    def visualize_results(self, image, results):
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
        axes[1].set_title('Top-5 Predictions', fontsize=14, fontweight='bold')
        axes[1].set_xlim(0, 1)
        
        for bar, prob in zip(bars, probs):
            width = bar.get_width()
            axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{prob:.2%}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig


def main():
    """主函数"""
    import streamlit as st
    
    st.set_page_config(
        page_title="细粒度猫狗分类系统",
        page_icon="🐱",
        layout="wide"
    )
    
    st.title("🐱🐕 细粒度猫狗品种分类系统")
    st.markdown("基于 CLIP + 动态提示词优化的细粒度分类演示")
    
    st.sidebar.title("⚙️ 设置")
    
    # 选择运行模式：本地或API
    run_mode = st.sidebar.radio(
        "运行模式",
        ["本地模型", "API服务"],
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
                st.sidebar.success("✅ API服务已连接")
            else:
                st.sidebar.warning("⚠️ API服务响应异常")
        except:
            st.sidebar.error("❌ API服务未启动，请先运行: cd web && python app.py")
    
    # 加载分类器
    @st.cache_resource
    def load_classifier(use_api=False, api_url="http://localhost:5001"):
        return PetClassifierDemo(use_api=use_api, api_url=api_url)
    
    classifier = load_classifier(use_api, api_url)
    
    # 提示词设置（本地模式和API模式都支持）
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 提示词设置")
    
    prompt_mode = st.sidebar.radio(
        "提示词模式",
        ["标准提示词", "自定义提示词", "无提示词"],
        help="选择不同的提示词模式来体验提示词的作用"
    )
    
    use_prompts = True
    custom_prompt_template = None
    
    if prompt_mode == "标准提示词":
        st.sidebar.info("使用标准提示词: 'a photo of a {breed}'")
    elif prompt_mode == "自定义提示词":
        custom_prompt_template = st.sidebar.text_input(
            "自定义提示词模板",
            value="a photo of a {cls}",
            help="使用 {cls} 占位符表示品种名称，如: 'a cute {cls} cat' 或 'a {cls} dog breed'"
        )
        if custom_prompt_template:
            st.sidebar.success(f"当前模板: {custom_prompt_template}")
    else:
        use_prompts = False
        st.sidebar.warning("不使用提示词，直接使用类别名称")
    
    # 展示支持的提示词示例
    with st.sidebar.expander("💡 提示词示例"):
        st.markdown("""
        **提示词模板示例：**
        - `a photo of a {cls}` - 标准
        - `a cute {cls} cat` - 强调可爱
        - `a {cls} dog breed` - 强调品种
        - `a sleeping {cls}` - 添加动作
        - `{cls}, a lovely pet` - 描述性
        """)
    
    top_k = st.sidebar.slider("显示Top-K预测", 3, 10, 5)
    
    st.subheader("📤 上传图片")
    uploaded_file = st.file_uploader("上传宠物图片进行分类", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="上传图片", use_container_width=True)
        
        if st.button("🔍 开始识别", type="primary"):
            with st.spinner("正在识别..."):
                try:
                    if use_api:
                        # 调用 API，传递提示词模板
                        # 无提示词模式使用空模板
                        api_prompt = "" if prompt_mode == "无提示词" else custom_prompt_template
                        results = classifier.classify_via_api(image, top_k, prompt_template=api_prompt)
                        prompts = [r.get('prompt', '') for r in results]
                    else:
                        # 本地分类
                        probs, prompts = classifier.classify(
                            image, 
                            use_prompts=use_prompts, 
                            custom_prompt_template=custom_prompt_template
                        )
                        results = classifier.get_top_predictions(probs, prompts, top_k)
                    
                    with col2:
                        fig = classifier.visualize_results(image, results)
                        st.pyplot(fig)
                    
                    st.subheader("📊 预测结果")
                    for i, result in enumerate(results):
                        breed = result.get('breed', result.get('breed', ''))
                        prob = result.get('probability', 0)
                        st.write(f"{i+1}. **{breed.replace('_', ' ')}** - {prob:.2%}")
                    
                    # 显示使用的提示词
                    with st.expander("📝 查看使用的提示词"):
                            st.write(prompts[:5], "...")
                            
                except Exception as e:
                    st.error(f"识别失败: {str(e)}")
                    if use_api:
                        st.info("请确保API服务已启动: cd web && python app.py")
    
    # 底部说明
    st.markdown("---")
    st.markdown("""
    ### 💡 提示词的作用
    
    通过修改提示词模板，你可以：
    1. **引导模型关注特定特征** - 如 "a fluffy cat" 引导关注毛发特征
    2. **添加先验知识** - 如 "a purebred {cls}" 强调纯种
    3. **区分相似品种** - 通过不同的描述来区分易混淆的品种
    
    尝试不同的提示词，观察预测概率的变化！
    """)


if __name__ == "__main__":
    main()
