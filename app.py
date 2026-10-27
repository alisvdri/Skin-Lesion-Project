import os
import numpy as np
import streamlit as st
import torch
from torch import nn
from PIL import Image

MODEL_FILE = "skin_cnn_model.pth"
DEFAULT_NAMES = {
  0:"akiec - Actinic keratoses", 1:"bcc - Basal cell carcinoma",
  2:"bkl - Benign keratosis-like lesions", 3:"df - Dermatofibroma",
  4:"nv - Melanocytic nevi", 5:"mel - Melanoma", 6:"vasc - Vascular lesions"
}
class SmallCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(64,num_classes))
    def forward(self,x):
        return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None, None
    ckpt = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
    model = SmallCNN()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    names = ckpt.get("class_names", DEFAULT_NAMES)
    return model, {int(k):v for k,v in names.items()}

st.set_page_config(page_title="دسته‌بندی پژوهشی ضایعات پوستی", page_icon="🔬")
st.title("🔬 دسته‌بندی پژوهشی تصاویر ضایعات پوستی")
st.write("یک تصویر بارگذاری کنید تا مدل پژوهشی آن را در یکی از ۷ دسته دیتاست دسته‌بندی کند.")
st.warning("این برنامه صرفاً آموزشی و پژوهشی است و ابزار تشخیص پزشکی نیست. نتیجه را مبنای تصمیم درمانی قرار ندهید.")
model, names = load_model()
if model is None:
    st.error("فایل skin_cnn_model.pth پیدا نشد. ابتدا train_model.py را در Google Colab اجرا کنید و فایل مدل را کنار app.py قرار دهید.")
    st.stop()
uploaded = st.file_uploader("تصویر ضایعه پوستی را انتخاب کنید", type=["jpg","jpeg","png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="تصویر بارگذاری‌شده", use_container_width=True)
    if st.button("تحلیل پژوهشی تصویر", type="primary"):
        arr = np.asarray(image.resize((28,28)), dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2,0,1))
        x = torch.from_numpy(arr).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].numpy()
        idx = int(np.argmax(probs))
        st.subheader("نتیجه مدل")
        st.write(f"دسته پیش‌بینی‌شده: **{names.get(idx, str(idx))}**")
        st.write(f"امتیاز مدل برای این دسته: **{probs[idx]*100:.1f}%**")
        st.caption("این درصد، احتمال قطعی بیماری یا میزان اطمینان پزشکی نیست.")
        st.write("امتیاز دسته‌ها:")
        st.bar_chart({names.get(i,str(i)): float(probs[i]) for i in range(len(probs))})
st.caption("دیتاست آموزشی: HMNIST با تصاویر RGB کوچک‌شده ۲۸×۲۸.")
