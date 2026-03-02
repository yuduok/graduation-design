"""
Web API服务 - 细粒度猫狗分类
Flask API for Fine-Grained Pet Classification
"""
import os
import sys
import io
import torch
import clip
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

app = Flask(__name__)
CORS(app)


class PetClassifierAPI:
    """宠物分类API服务"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.classnames = None
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        print("Loading CLIP model...")
        self.clip_model, _ = clip.load("RN50", device=self.device)
        self.clip_model.eval()
        self.clip_model = self.clip_model.float()
        
        # Oxford-IIIT Pets 37类
        self.classnames = [
            'Abyssinian', 'american_bulldog', 'basset_hound', 'beagle', 'Bengal',
            'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua',
            'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 'German_Shorthaired_Pointer',
            'Great_Dane', 'Havanese', 'japanese_chin', 'Keeshond', 'Leonberger',
            'Maine_Coon', 'Miniature_Pinscher', 'newfoundland', 'Persian', 'Pomeranian',
            'pug', 'Ragdoll', 'Russian_Blue', 'Saint_Bernard', 'Samoyed',
            'Scottish_Terrier', 'Shiba_Inu', 'Siamese', 'Sphynx', 'Staffordshire_Bull_Terrier',
            'wheaten_terrier', 'Yorkshire_Terrier'
        ]
        print("Model loaded successfully!")
    
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
    
    def predict(self, image_tensor, top_k=5):
        """预测"""
        with torch.no_grad():
            # 编码图像
            image_features = self.clip_model.encode_image(image_tensor.float())
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 生成提示词
            prompts = [f"a photo of a {name.replace('_', ' ')}" for name in self.classnames]
            
            # 编码文本
            tokens = clip.tokenize(prompts).to(self.device)
            text_features = self.clip_model.encode_text(tokens.float())
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            logits = (image_features @ text_features.T).squeeze()
            probs = logits.softmax(dim=-1)
            
            # Top-K
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                results.append({
                    "breed": self.classnames[idx.item()].replace('_', ' '),
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
        
        image, image_tensor = classifier.preprocess(image_bytes)
        results = classifier.predict(image_tensor, top_k)
        
        return jsonify({
            "success": True,
            "predictions": results
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
    print("Starting API server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
