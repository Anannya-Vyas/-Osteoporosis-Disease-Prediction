import os
import pickle
import gc

# Global Configuration
DEMO_MODE = False  # Try real mode first

class ModelLoader:
    def __init__(self):
        self.xgb_model = None
        self.image_model = None
        self.is_demo = False
        self.warning_msg = None
        self.classes = ['Normal', 'Osteopenia', 'Osteoporosis']
        self.device = None
        self.transform = None
        
        if not DEMO_MODE:
            self._load_clinical_model()

    def _load_clinical_model(self):
        xgb_path = os.path.join(os.path.dirname(__file__), "../models/clinical/xgboost_baseline.pkl")
        if os.path.exists(xgb_path):
            with open(xgb_path, 'rb') as f:
                self.xgb_model = pickle.load(f)
            print("Loaded XGBoost clinical model.")
        else:
            print(f"Warning: {xgb_path} not found.")

    def _load_image_model(self):
        # Look for the new trained DenseNet model from 'new updates' first
        new_updates_paths = [
            r"D:\Downloads\Osteoporosis-with-XRay-main\new updates\densenet121_clean.pth",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../new updates/densenet121_clean.pth")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/densenet121_clean.pth")),
        ]
        
        pth_path = None
        for p in new_updates_paths:
            if os.path.exists(p):
                pth_path = p
                break

        # Try loading PyTorch directly if available
        try:
            import torch
            import torchvision.models as models
            import torch.nn as nn

            if pth_path:
                device = torch.device('cpu')
                model = models.densenet121()
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_ftrs, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 3)
                )
                model.load_state_dict(torch.load(pth_path, map_location=device))
                model.eval()
                self.image_model = model
                self.model_backend = 'pytorch'
                print(f"SUCCESS: Loaded new trained DenseNet121 model from: {pth_path}")
                return True
        except Exception as e:
            print(f"INFO: PyTorch load attempt info: {e}")

        # Fallback to ONNX if PyTorch load wasn't used or failed
        try:
            import onnxruntime as ort
            img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/densenet121.onnx"))
            if os.path.exists(img_path):
                self.image_model = ort.InferenceSession(img_path, providers=['CPUExecutionProvider'])
                self.model_backend = 'onnx'
                print("Successfully Loaded DenseNet121 ONNX model.")
                return True
            else:
                self.warning_msg = f"Model file not found: {img_path}"
                self.is_demo = True
                return False
        except Exception as e:
            self.warning_msg = f"Model load error: {str(e)}"
            self.is_demo = True
            return False

    def predict_risk(self, features):
        if self.is_demo:
            return 0.82
            
        import pandas as pd
        df = pd.DataFrame([features])
        return float(self.xgb_model.predict_proba(df)[0][1])

    def predict_image(self, image_path):
        if self.is_demo:
            msg = getattr(self, 'warning_msg', 'Demo Mode')
            return f"Fallback: {msg}", [0.1, 0.7, 0.2]
            
        if self.image_model is None:
            success = self._load_image_model()
            if not success:
                msg = getattr(self, 'warning_msg', 'Unknown Error')
                return f"Fallback: {msg}", [0.1, 0.7, 0.2]

        try:
            from PIL import Image
            import numpy as np

            # Standardized DenseNet preprocessing
            img = Image.open(image_path).convert('RGB')
            img = img.resize((256, 256), Image.Resampling.BILINEAR)
            width, height = img.size
            left = (width - 224) / 2
            top = (height - 224) / 2
            img = img.crop((left, top, (width + 224) / 2, (height + 224) / 2))
            
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.transpose(img_array, (2, 0, 1))
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_array = (img_array - mean) / std

            if getattr(self, 'model_backend', 'pytorch') == 'pytorch':
                import torch
                input_tensor = torch.from_numpy(np.expand_dims(img_array, axis=0).astype(np.float32))
                with torch.no_grad():
                    outputs = self.image_model(input_tensor)
                    probs = torch.softmax(outputs, dim=1).squeeze().tolist()
                    pred_idx = int(torch.argmax(outputs, dim=1).item())
                    return (self.classes[pred_idx], probs)
            else:
                input_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)
                input_name = self.image_model.get_inputs()[0].name
                outputs = self.image_model.run(None, {input_name: input_tensor})[0]
                exp_preds = np.exp(outputs[0] - np.max(outputs[0]))
                probs = (exp_preds / exp_preds.sum()).tolist()
                pred_idx = np.argmax(probs)
                return (self.classes[pred_idx], probs)
        finally:
            del self.image_model
            self.image_model = None
            gc.collect()

    def predict_fusion(self, clinical_score, image_prediction):
        if self.is_demo:
            return "Osteoporosis"
            
        img_severity = self.classes.index(image_prediction)
        img_score = img_severity / 2.0  
        final_score = (0.6 * clinical_score) + (0.4 * img_score)
        
        if final_score > 0.65:
            return "Osteoporosis"
        elif final_score > 0.35:
            return "Osteopenia"
        return "Normal"

# Eager loading instance (will only load clinical model now)
ai_system = ModelLoader()
