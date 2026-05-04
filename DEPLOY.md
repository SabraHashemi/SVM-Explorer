# 🚀 راهنمای Deploy

## ① GitHub

```bash
# ۱. ساخت repo جدید در github.com
# ۲. در ترمینال:
git init
git add .
git commit -m "🧠 Add SVM Explorer app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/svm-explorer.git
git push -u origin main
```

## ② Hugging Face Spaces

### روش اول (آسان‌تر) — از GitHub import
1. برو به huggingface.co/new-space
2. انتخاب کن: **Streamlit** SDK
3. گزینه "**Import from GitHub**" رو بزن
4. آدرس repo گیت‌هابت رو بده
5. کلیک **Create Space** ✅

### روش دوم — دستی
```bash
pip install huggingface_hub

git clone https://huggingface.co/spaces/YOUR_USERNAME/svm-explorer
cd svm-explorer
# فایل‌های app.py و requirements.txt رو کپی کن
git add .
git commit -m "🧠 SVM Explorer"
git push
```

## ✅ چک‌لیست

- [ ] `app.py` در root پروژه
- [ ] `requirements.txt` در root پروژه
- [ ] `README.md` با header صحیح (برای HF)
- [ ] Python 3.9+ در محیط
