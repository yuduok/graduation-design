"""
细粒度猫狗分类可视化演示系统
Streamlit Demo for Fine-Grained Pet Classification with Dynamic Prompts
"""
import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 添加CoOp的clip路径（避免冲突）
COOP_PATH = "/Users/yudu/Documents/毕业设计/CoOp"
sys.path.insert(0, os.path.join(COOP_PATH, "clip"))
import clip

# 添加dassl路径
sys.path.insert(0, os.path.join(COOP_PATH, "dassl"))


class PetClassifierDemo:
    """宠物分类演示系统"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model = None
        self.classnames = None
        
        # 类别名称（Oxford-IIIT Pets 37类）
        self.oxford_pets_classes = [
            'Abyssinian', 'american_bulldog', 'basset_hound', 'beagle', 'Bengal',
            'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua',
            'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 'German_Shorthaired_Pointer',
            'Great_Dane', 'Havanese', 'japanese_chin', 'Keeshond', 'Leonberger',
            'Maine_Coon', 'Miniature_Pinscher', 'newfoundland', 'Persian', 'Pomeranian',
            'pug', 'Ragdoll', 'Russian_Blue', 'Saint_Bernard', 'Samoyed',
            'Scottish_Terrier', 'Shiba_Inu', 'Siamese', 'Sphynx', 'Staffordshire_Bull_Terrier',
            'wheaten_terrier', 'Yorkshire_Terrier'
        ]
        
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
    
    def classify(self, image, use_prompts=True):
        """分类"""
        image_tensor = self.preprocess_image(image)
        image_features = self.encode_images(image_tensor)
        
        if use_prompts:
            prompts = [f"a photo of a {name.replace('_', ' ')}" for name in self.classnames]
        else:
            prompts = self.classnames
        
        text_features = self.encode_texts(prompts)
        logits = (image_features @ text_features.T).squeeze()
        probs = logits.softmax(dim=-1)
        
        return probs, prompts
    
    def get_top_predictions(self, probs, prompts, top_k=5):
        """获取Top-K预测"""
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
        
        breeds = [r['breed'].replace('_', ' ') for r in results]
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
    
    @st.cache_resource
    def load_classifier():
        return PetClassifierDemo()
    
    classifier = load_classifier()
    
    st.sidebar.title("设置")
    use_prompts = st.sidebar.checkbox("使用提示词", value=True)
    top_k = st.sidebar.slider("显示Top-K预测", 3, 10, 5)
    
    uploaded_file = st.file_uploader("上传宠物图片", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="上传图片", use_column_width=True)
        
        if st.button("开始识别", type="primary"):
            with st.spinner("正在识别..."):
                probs, prompts = classifier.classify(image, use_prompts=use_prompts)
                results = classifier.get_top_predictions(probs, prompts, top_k)
                
                with col2:
                    fig = classifier.visualize_results(image, results)
                    st.pyplot(fig)
                
                st.subheader("预测结果")
                for i, result in enumerate(results):
                    st.write(f"{i+1}. **{result['breed'].replace('_', ' ')}** - {result['probability']:.2%}")


if __name__ == "__main__":
    main()
