import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SVM Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.main-header {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid #00d4ff22;
    text-align: center;
}

.main-header h1 {
    color: #00d4ff;
    font-size: 2.5rem;
    margin: 0;
    text-shadow: 0 0 30px #00d4ff66;
}

.main-header p {
    color: #8892b0;
    margin-top: 0.5rem;
    font-size: 1rem;
}

.concept-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #00d4ff33;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: #ccd6f6;
}

.concept-card h3 {
    color: #00d4ff !important;
    margin-top: 0;
}

.metric-box {
    background: #0f0f1a;
    border: 1px solid #00d4ff44;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-box .value {
    font-size: 2rem;
    font-weight: bold;
    color: #00d4ff;
    font-family: 'Space Mono', monospace;
}

.metric-box .label {
    color: #8892b0;
    font-size: 0.85rem;
}

.highlight {
    background: linear-gradient(90deg, #00d4ff22, transparent);
    border-left: 3px solid #00d4ff;
    padding: 0.5rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
    color: #ccd6f6;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: #1a1a2e;
    border-radius: 8px;
    color: #8892b0;
    border: 1px solid #00d4ff22;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00d4ff22, #0099cc22) !important;
    color: #00d4ff !important;
    border-color: #00d4ff66 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 SVM Explorer</h1>
    <p>Support Vector Machine — Interactive Learning & Visualization Tool</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ تنظیمات مدل")
    st.markdown("---")

    kernel = st.selectbox(
        "🔧 Kernel",
        ["linear", "rbf", "poly", "sigmoid"],
        help="نوع تابع kernel برای SVM"
    )

    C = st.slider(
        "📏 C (Regularization)",
        min_value=0.01, max_value=20.0,
        value=1.0, step=0.01,
        help="پارامتر regularization — هرچه بزرگتر، margin سخت‌گیرانه‌تر"
    )

    if kernel == "rbf":
        gamma_mode = st.radio("Gamma", ["scale", "auto", "custom"])
        if gamma_mode == "custom":
            gamma = st.slider("γ مقدار", 0.001, 5.0, 0.5, 0.001)
        else:
            gamma = gamma_mode
    elif kernel == "poly":
        degree = st.slider("🔢 Degree", 2, 6, 3)
        gamma = "scale"
    else:
        gamma = "scale"
        degree = 3

    st.markdown("---")
    st.markdown("### 📊 Dataset")
    dataset_name = st.selectbox(
        "Dataset",
        ["مصنوعی (دو کلاس)", "Iris (دو feature)", "Moons", "Circles", "Blobs"]
    )

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 100, 42)

# ─── Data Generation ────────────────────────────────────────────────────────
@st.cache_data
def get_dataset(name, random_state=42):
    rs = int(random_state)
    if name == "مصنوعی (دو کلاس)":
        X, y = datasets.make_classification(
            n_samples=200, n_features=2, n_redundant=0,
            n_informative=2, random_state=rs, n_clusters_per_class=1
        )
    elif name == "Iris (دو feature)":
        iris = datasets.load_iris()
        X = iris.data[:100, :2]
        y = iris.target[:100]
    elif name == "Moons":
        X, y = datasets.make_moons(n_samples=200, noise=0.2, random_state=rs)
    elif name == "Circles":
        X, y = datasets.make_circles(n_samples=200, noise=0.15, factor=0.4, random_state=rs)
    else:  # Blobs
        X, y = datasets.make_blobs(n_samples=200, centers=2, random_state=rs)
    return X, y

X, y = get_dataset(dataset_name, random_state)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=int(random_state)
)

# ─── Train Model ────────────────────────────────────────────────────────────
try:
    if kernel == "poly":
        clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, probability=True)
    else:
        clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    n_support = clf.n_support_
    model_trained = True
except Exception as e:
    st.error(f"خطا در آموزش مدل: {e}")
    model_trained = False

# ─── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📖 SVM چیست؟",
    "🎯 Decision Boundary",
    "🔬 مقایسه Kernel‌ها",
    "📊 ارزیابی مدل"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — آموزش مفاهیم
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## SVM چیست؟")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
<div class="concept-card">
<h3>🎯 ایده اصلی</h3>
<p>SVM (Support Vector Machine) یک الگوریتم classification است که سعی می‌کند <strong>بهترین خط جداکننده (Hyperplane)</strong> را بین دو کلاس پیدا کند.</p>
<br>
<div class="highlight">هدف: بیشترین فاصله (Margin) بین دو کلاس</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="concept-card">
<h3>📌 Support Vectors</h3>
<p>نقاطی که به hyperplane نزدیک‌ترند و آن را تعریف می‌کنند. حذف سایر نقاط تأثیری روی مدل نمی‌گذارد!</p>
<br>
<div class="highlight">SVM فقط به "سخت‌ترین" نقاط اهمیت می‌دهد</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="concept-card">
<h3>🌀 Kernel Trick</h3>
<p>وقتی داده‌ها خطی جدا نمی‌شوند، kernel داده‌ها را به فضای بالاتر می‌برد تا جداسازی ممکن شود.</p>
<br>
<div class="highlight">مثل تا کردن کاغذ برای جدا کردن نقاط!</div>
</div>
""", unsafe_allow_html=True)

    with col2:
        # ─── Concept Visualization ───
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.patch.set_facecolor('#0f0f1a')

        for ax in axes:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='#8892b0')
            for spine in ax.spines.values():
                spine.set_edgecolor('#00d4ff22')

        # Left: Hard Margin concept
        ax = axes[0]
        np.random.seed(42)
        X_pos = np.random.randn(15, 2) + [2, 2]
        X_neg = np.random.randn(15, 2) + [-2, -2]
        ax.scatter(X_pos[:, 0], X_pos[:, 1], c='#00d4ff', s=60, zorder=5, label='کلاس +1')
        ax.scatter(X_neg[:, 0], X_neg[:, 1], c='#ff6b9d', s=60, zorder=5, label='کلاس -1')
        x_line = np.linspace(-5, 5, 100)
        ax.plot(x_line, x_line * 0 + 0, color='#00ff88', lw=2.5, label='Hyperplane')
        ax.fill_between(x_line, -0.8, 0.8, alpha=0.15, color='#00ff88', label='Margin')
        ax.axhline(y=0.8, color='#00ff88', lw=1, ls='--', alpha=0.7)
        ax.axhline(y=-0.8, color='#00ff88', lw=1, ls='--', alpha=0.7)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_title('Margin & Hyperplane', color='#00d4ff', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, facecolor='#0f0f1a', labelcolor='#ccd6f6')

        # Right: Kernel trick concept
        ax = axes[1]
        theta = np.linspace(0, 2*np.pi, 50)
        r_in = 0.8; r_out = 1.8
        X_in = np.column_stack([r_in*np.cos(theta[:25]), r_in*np.sin(theta[:25])])
        X_out = np.column_stack([r_out*np.cos(theta), r_out*np.sin(theta)])
        ax.scatter(X_in[:, 0], X_in[:, 1], c='#00d4ff', s=60, zorder=5, label='کلاس +1')
        ax.scatter(X_out[:, 0], X_out[:, 1], c='#ff6b9d', s=60, zorder=5, label='کلاس -1')
        circle = plt.Circle((0, 0), 1.3, fill=False, color='#00ff88', lw=2.5, label='Decision Boundary')
        ax.add_patch(circle)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title('Kernel Trick (RBF)', color='#00d4ff', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, facecolor='#0f0f1a', labelcolor='#ccd6f6')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Params explanation
    st.markdown("---")
    st.markdown("## 🔧 پارامترها")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
<div class="concept-card">
<h3>C — Regularization</h3>
<ul>
<li>C کوچک → Margin عریض‌تر، خطاهای بیشتر قابل قبول</li>
<li>C بزرگ → Margin باریک‌تر، سعی می‌کند همه را درست classify کند</li>
<li>C خیلی بزرگ → Overfitting احتمالی</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="concept-card">
<h3>Kernel Types</h3>
<ul>
<li><strong>Linear</strong>: داده‌های خطاً جداپذیر</li>
<li><strong>RBF</strong>: پرکاربردترین، انعطاف‌پذیر</li>
<li><strong>Poly</strong>: مناسب داده‌های چندجمله‌ای</li>
<li><strong>Sigmoid</strong>: شبیه شبکه عصبی</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div class="concept-card">
<h3>Gamma (γ)</h3>
<ul>
<li>فقط در RBF و Poly</li>
<li>γ بزرگ → مرز پیچیده‌تر، overfitting</li>
<li>γ کوچک → مرز ساده‌تر، underfitting</li>
<li>پیشنهاد: از "scale" شروع کنید</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — Decision Boundary
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🎯 Decision Boundary")

    if model_trained:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        metrics = [
            ("🎯 Accuracy", f"{acc:.1%}"),
            ("⚙️ Kernel", kernel.upper()),
            ("📏 C", f"{C:.2f}"),
            ("🔵 Support Vectors", str(sum(n_support)))
        ]
        for col, (label, val) in zip([col_m1, col_m2, col_m3, col_m4], metrics):
            with col:
                st.markdown(f"""
<div class="metric-box">
    <div class="value">{val}</div>
    <div class="label">{label}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Decision boundary plot
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor('#0f0f1a')
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#8892b0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#00d4ff22')

        h = 0.03
        x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
        y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                              np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        cmap_bg = ListedColormap(['#ff6b9d22', '#00d4ff22'])
        ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.6)
        ax.contour(xx, yy, Z, colors=['#00ff88'], linewidths=2, alpha=0.8)

        scatter_colors = ['#ff6b9d', '#00d4ff']
        for i, color in enumerate(scatter_colors):
            mask_train = y_train == i
            mask_test = y_test == i
            ax.scatter(X_train[mask_train, 0], X_train[mask_train, 1],
                       c=color, s=50, alpha=0.8, edgecolors='white', linewidth=0.5, label=f'Train کلاس {i}')
            ax.scatter(X_test[mask_test, 0], X_test[mask_test, 1],
                       c=color, s=80, marker='D', alpha=1.0, edgecolors='white', linewidth=1.0, label=f'Test کلاس {i}')

        # Support vectors
        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none',
                   edgecolors='#00ff88', linewidth=2, zorder=10, label='Support Vectors')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f'Decision Boundary — {kernel.upper()} Kernel | C={C:.2f}',
                     color='#00d4ff', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=9, facecolor='#0f0f1a',
                  labelcolor='#ccd6f6', edgecolor='#00d4ff33')
        ax.grid(True, alpha=0.1, color='#ffffff')
        ax.set_xlabel("Feature 1 (normalized)", color='#8892b0')
        ax.set_ylabel("Feature 2 (normalized)", color='#8892b0')

        st.pyplot(fig)
        plt.close()

        st.info(f"🔵 **{sum(n_support)}** support vector در این مدل وجود دارد  |  "
                f"مربع‌ها (◆) نقاط test هستند")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — Kernel Comparison
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🔬 مقایسه Kernel‌های مختلف")
    st.markdown("مدل روی همان dataset با C یکسان و kernel‌های مختلف آموزش می‌بیند:")

    kernels = ["linear", "rbf", "poly", "sigmoid"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor('#0f0f1a')
    axes = axes.ravel()

    h = 0.05
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    results = {}
    for i, k in enumerate(kernels):
        ax = axes[i]
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#8892b0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#00d4ff22')

        try:
            if k == "poly":
                m = svm.SVC(kernel=k, C=C, degree=3, probability=True)
            else:
                m = svm.SVC(kernel=k, C=C, probability=True)
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            acc_k = accuracy_score(y_test, pred)
            results[k] = acc_k

            Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            cmap_bg = ListedColormap(['#ff6b9d22', '#00d4ff22'])
            ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.6)
            ax.contour(xx, yy, Z, colors=['#00ff88'], linewidths=2, alpha=0.8)

            for cls, color in enumerate(['#ff6b9d', '#00d4ff']):
                ax.scatter(X_scaled[y == cls, 0], X_scaled[y == cls, 1],
                           c=color, s=30, alpha=0.7, edgecolors='none')

            sv = m.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors='none',
                       edgecolors='#00ff88', linewidth=1.5, zorder=10)

            ax.set_title(f'{k.upper()} — Accuracy: {acc_k:.1%}',
                         color='#00d4ff', fontsize=11, fontweight='bold')
        except Exception as e:
            ax.text(0.5, 0.5, f"خطا:\n{str(e)}", transform=ax.transAxes,
                    ha='center', va='center', color='#ff6b9d')
            ax.set_title(k.upper(), color='#8892b0')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.1, color='#ffffff')
        ax.set_xlabel("Feature 1", color='#8892b0', fontsize=8)
        ax.set_ylabel("Feature 2", color='#8892b0', fontsize=8)

    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    if results:
        st.markdown("### 📊 جدول مقایسه")
        df_results = pd.DataFrame({
            "Kernel": list(results.keys()),
            "Accuracy": [f"{v:.1%}" for v in results.values()],
            "Score": list(results.values())
        }).sort_values("Score", ascending=False).drop("Score", axis=1)
        df_results.index = range(1, len(df_results) + 1)
        st.dataframe(df_results, use_container_width=True)

        best_kernel = max(results, key=results.get)
        st.success(f"✅ بهترین kernel روی این dataset: **{best_kernel.upper()}** با accuracy **{results[best_kernel]:.1%}**")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — Model Evaluation
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📊 ارزیابی مدل")

    if model_trained:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#1a1a2e')

            sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                        cmap='Blues', linewidths=0.5,
                        annot_kws={"size": 16, "weight": "bold", "color": "white"},
                        cbar_kws={'shrink': 0.8})
            ax.set_xlabel("Predicted", color='#8892b0')
            ax.set_ylabel("Actual", color='#8892b0')
            ax.set_title("Confusion Matrix", color='#00d4ff', fontsize=12, fontweight='bold')
            ax.tick_params(colors='#8892b0')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df.round(3)
            st.dataframe(report_df, use_container_width=True, height=250)

            st.markdown("### 🔍 جزئیات مدل")
            info_data = {
                "پارامتر": ["Kernel", "C", "Support Vectors (کلاس 0)", "Support Vectors (کلاس 1)", "Train Size", "Test Size"],
                "مقدار": [
                    kernel, C,
                    n_support[0] if len(n_support) > 0 else "-",
                    n_support[1] if len(n_support) > 1 else "-",
                    len(X_train), len(X_test)
                ]
            }
            st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)

        # C Effect plot
        st.markdown("---")
        st.markdown("### 📈 تأثیر پارامتر C روی مدل")

        C_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
        train_accs, test_accs, sv_counts = [], [], []

        for c_val in C_values:
            try:
                m = svm.SVC(kernel=kernel, C=c_val)
                m.fit(X_train, y_train)
                train_accs.append(accuracy_score(y_train, m.predict(X_train)))
                test_accs.append(accuracy_score(y_test, m.predict(X_test)))
                sv_counts.append(sum(m.n_support_))
            except:
                train_accs.append(0); test_accs.append(0); sv_counts.append(0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#0f0f1a')

        for ax in [ax1, ax2]:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='#8892b0')
            for spine in ax.spines.values():
                spine.set_edgecolor('#00d4ff22')
            ax.grid(True, alpha=0.15, color='#ffffff')

        ax1.semilogx(C_values, train_accs, 'o-', color='#00d4ff', lw=2, label='Train Accuracy')
        ax1.semilogx(C_values, test_accs, 's--', color='#ff6b9d', lw=2, label='Test Accuracy')
        ax1.axvline(x=C, color='#00ff88', lw=1.5, ls=':', alpha=0.8, label=f'C={C}')
        ax1.set_xlabel("C (log scale)", color='#8892b0')
        ax1.set_ylabel("Accuracy", color='#8892b0')
        ax1.set_title("Accuracy vs C", color='#00d4ff', fontsize=11, fontweight='bold')
        ax1.legend(facecolor='#0f0f1a', labelcolor='#ccd6f6', fontsize=9)

        ax2.semilogx(C_values, sv_counts, 'o-', color='#00ff88', lw=2)
        ax2.axvline(x=C, color='#ff6b9d', lw=1.5, ls=':', alpha=0.8, label=f'C={C}')
        ax2.set_xlabel("C (log scale)", color='#8892b0')
        ax2.set_ylabel("Support Vectors", color='#8892b0')
        ax2.set_title("تعداد Support Vectors vs C", color='#00d4ff', fontsize=11, fontweight='bold')
        ax2.legend(facecolor='#0f0f1a', labelcolor='#ccd6f6', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8892b0; font-family:'Space Mono',monospace; font-size:0.8rem;">
    SVM Explorer · Built with Streamlit & scikit-learn · 🧠
</div>
""", unsafe_allow_html=True)
