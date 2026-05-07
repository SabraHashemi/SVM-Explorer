import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm, datasets
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, mean_squared_error, r2_score)
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ML Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"], .stApp {
    background-color: #0a0f1e !important;
    font-family: 'DM Sans', sans-serif;
    color: #ccd6f6;
}
h1,h2,h3,h4 { font-family:'JetBrains Mono',monospace !important; color:#4fc3f7 !important; }
[data-testid="stSidebar"] { background-color:#0d1b2a !important; border-right:1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color:#ccd6f6 !important; }
[data-testid="stAppViewContainer"]>.main,
[data-testid="block-container"] { background-color:#0a0f1e !important; }

.main-header {
    background:linear-gradient(135deg,#0d1b2a,#112240,#0a1628);
    padding:1.8rem 2rem; border-radius:14px; margin-bottom:1.5rem;
    border:1px solid #1e3a5f; text-align:center;
    box-shadow:0 4px 30px #00e5ff0d;
}
.main-header h1 { color:#4fc3f7 !important; font-size:2rem; margin:0; text-shadow:0 0 40px #4fc3f766; }
.main-header p  { color:#8892b0; margin-top:0.4rem; font-size:0.9rem; }

.algo-badge {
    display:inline-block; padding:0.25rem 0.75rem; border-radius:20px;
    font-family:'JetBrains Mono',monospace; font-size:0.75rem;
    background:#112240; border:1px solid #4fc3f766; color:#4fc3f7;
    margin-bottom:1rem;
}
.card {
    background:linear-gradient(135deg,#0d1b2a,#112240);
    border:1px solid #1e3a5f; border-radius:12px;
    padding:1.2rem; margin:0.6rem 0; color:#ccd6f6;
}
.card h3 { color:#4fc3f7 !important; margin-top:0; font-size:0.95rem; }
.card ul  { padding-left:1.2rem; margin:0.4rem 0 0; color:#a8b2d8; font-size:0.88rem; line-height:1.7; }
.highlight {
    background:linear-gradient(90deg,#4fc3f711,transparent);
    border-left:3px solid #4fc3f7; padding:0.45rem 1rem;
    border-radius:0 8px 8px 0; margin:0.6rem 0; color:#ccd6f6; font-size:0.88rem;
}
.metric-box {
    background:#0d1b2a; border:1px solid #1e3a5f;
    border-radius:10px; padding:0.9rem; text-align:center; margin-bottom:0.5rem;
}
.metric-box .value { font-size:1.6rem; font-weight:700; color:#4fc3f7; font-family:'JetBrains Mono',monospace; }
.metric-box .label { color:#8892b0; font-size:0.78rem; margin-top:0.15rem; }

.upload-hint {
    background:#0d1b2a; border:2px dashed #1e3a5f; border-radius:12px;
    padding:1.2rem; text-align:center; color:#8892b0; font-size:0.88rem; margin:0.5rem 0;
}

.stTabs [data-baseweb="tab-list"] { gap:5px; background:transparent; }
.stTabs [data-baseweb="tab"] {
    background:#0d1b2a; border-radius:8px; color:#8892b0;
    border:1px solid #1e3a5f; font-family:'JetBrains Mono',monospace;
    font-size:0.8rem; padding:0.45rem 0.9rem;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#112240,#0d1b2a) !important;
    color:#4fc3f7 !important; border-color:#4fc3f766 !important;
}
.stAlert { background:#112240 !important; border:1px solid #1e3a5f !important; color:#ccd6f6 !important; border-radius:8px; }
div[data-baseweb="select"]>div { background:#112240 !important; border-color:#1e3a5f !important; color:#ccd6f6 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def style_ax(ax, fig=None):
    if fig: fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0d1b2a')
    ax.tick_params(colors='#8892b0', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#1e3a5f')
    ax.grid(True, alpha=0.12, color='#4fc3f7')

def metric_box(val, label):
    return f'<div class="metric-box"><div class="value">{val}</div><div class="label">{label}</div></div>'

COLORS = ['#ef5350','#4fc3f7','#66bb6a','#ffa726','#ab47bc']

def plot_boundary(ax, clf, X_scaled, y, X_train, y_train, X_test, y_test, feat1="F1", feat2="F2"):
    h = 0.04
    xmn,xmx = X_scaled[:,0].min()-.6, X_scaled[:,0].max()+.6
    ymn,ymx = X_scaled[:,1].min()-.6, X_scaled[:,1].max()+.6
    xx,yy = np.meshgrid(np.arange(xmn,xmx,h), np.arange(ymn,ymx,h))
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    n_cls = len(np.unique(y))
    bg_colors = ['#ef535014','#4fc3f714','#66bb6a14','#ffa72614'][:n_cls]
    ax.contourf(xx,yy,Z, cmap=ListedColormap(bg_colors), alpha=0.8)
    ax.contour(xx,yy,Z, colors=['#00e5ff'], linewidths=1.5, alpha=0.7)
    for i in range(n_cls):
        c = COLORS[i]
        ax.scatter(X_train[y_train==i,0],X_train[y_train==i,1],
                   c=c,s=45,alpha=0.8,edgecolors='white',lw=0.4,label=f'Train cls {i}')
        ax.scatter(X_test[y_test==i,0],X_test[y_test==i,1],
                   c=c,s=70,marker='D',alpha=1,edgecolors='white',lw=0.8,label=f'Test cls {i}')
    ax.set_xlim(xmn,xmx); ax.set_ylim(ymn,ymx)
    ax.set_xlabel(f"{feat1} (norm.)", color='#8892b0', fontsize=8)
    ax.set_ylabel(f"{feat2} (norm.)", color='#8892b0', fontsize=8)
    ax.legend(fontsize=7, facecolor='#0d1b2a', labelcolor='#ccd6f6', edgecolor='#1e3a5f', loc='upper right')

def plot_confusion(ax, fig, y_test, y_pred):
    style_ax(ax, fig)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.light_palette("#4fc3f7", as_cmap=True),
                linewidths=0.5, annot_kws={"size":14,"weight":"bold","color":"#0a0f1e"},
                cbar_kws={'shrink':0.8})
    ax.set_xlabel("Predicted", color='#8892b0')
    ax.set_ylabel("Actual", color='#8892b0')
    ax.set_title("Confusion Matrix", color='#4fc3f7', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#8892b0')

# ── CSV Upload helper ──────────────────────────────────────────
def csv_uploader(key, task="classification"):
    """Returns (X, y, feat1_name, feat2_name) or None if not uploaded."""
    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"], key=key,
        help="Numeric feature columns + one label column"
    )
    if uploaded is None:
        return None
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}"); return None

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols  = df.columns.tolist()
    if len(num_cols) < 2:
        st.error("Need at least 2 numeric columns."); return None

    c1,c2,c3 = st.columns(3)
    with c1: f1 = st.selectbox("Feature 1", num_cols, index=0, key=key+"_f1")
    with c2: f2 = st.selectbox("Feature 2", num_cols, index=min(1,len(num_cols)-1), key=key+"_f2")
    with c3: lc = st.selectbox("Label column", all_cols, index=len(all_cols)-1, key=key+"_lc")

    try:
        valid = df[[f1,f2,lc]].dropna()
        X = valid[[f1,f2]].values.astype(float)
        y_raw = valid[lc].values
        uniq = np.unique(y_raw)
        if task == "classification" and len(uniq) < 2:
            st.error("Label must have at least 2 unique values."); return None
        lmap = {v:i for i,v in enumerate(uniq)}
        y = np.array([lmap[v] for v in y_raw])
    except Exception as e:
        st.error(f"Column error: {e}"); return None

    st.success(f"✅ {len(X)} samples loaded — {len(uniq)} classes")
    with st.expander("👀 Preview"):
        st.dataframe(valid[[f1,f2,lc]].head(10), use_container_width=True)
    return X, y, f1, f2

@st.cache_data
def builtin_clf(name, rs=42):
    if name == "Moons":       return datasets.make_moons(200,noise=0.25,random_state=rs)
    if name == "Circles":     return datasets.make_circles(200,noise=0.15,factor=0.4,random_state=rs)
    if name == "Blobs":       return datasets.make_blobs(200,centers=3,random_state=rs)
    if name == "Iris":
        d=datasets.load_iris(); return d.data[:,:2], d.target
    return datasets.make_classification(200,n_features=2,n_redundant=0,n_informative=2,
                                         random_state=rs,n_clusters_per_class=1)

@st.cache_data
def builtin_reg(name, rs=42):
    np.random.seed(rs)
    if name == "Linear":
        X=np.linspace(-3,3,200).reshape(-1,1); y=2.5*X.ravel()+np.random.randn(200)*1.2; return X,y
    if name == "Polynomial":
        X=np.linspace(-3,3,200).reshape(-1,1); y=0.8*X.ravel()**3-2*X.ravel()**2+np.random.randn(200)*2; return X,y
    if name == "Noisy":
        X=np.linspace(-3,3,200).reshape(-1,1); y=np.sin(X.ravel()*2)+np.random.randn(200)*1.5; return X,y
    d=datasets.load_diabetes(); return d.data[:,np.newaxis,2], d.target   # Boston-like fallback

# ══════════════════════════════════════════════════════════════
# SIDEBAR — Algorithm Selector
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🤖 ML Explorer")
    st.markdown("---")
    algo = st.radio(
        "Algorithm",
        ["⚡ SVM", "📈 Regression", "🌳 Decision Tree / RF",
         "🔵 K-Means Clustering", "👥 KNN", "🧠 Neural Network (MLP)"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<div style="color:#8892b0;font-size:0.78rem;">Tune parameters below ↓</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ██  SVM
# ══════════════════════════════════════════════════════════════
if algo == "⚡ SVM":
    st.markdown('<div class="main-header"><h1>⚡ Support Vector Machine</h1><p>Find the maximum-margin hyperplane between classes</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### SVM Settings")
        kernel = st.selectbox("Kernel", ["rbf","linear","poly","sigmoid"])
        C      = st.slider("C", 0.01, 20.0, 1.0, 0.01)
        if kernel=="rbf":
            gm = st.radio("Gamma",["scale","auto","custom"])
            gamma = st.slider("γ",0.001,5.0,0.5,0.001) if gm=="custom" else gm
            degree=3
        elif kernel=="poly":
            degree=st.slider("Degree",2,6,3); gamma="scale"
        else:
            gamma="scale"; degree=3
        st.markdown("---")
        st.markdown("### Data")
        src = st.radio("Source",["Built-in","Upload CSV"], key="svm_src")
        if src=="Built-in":
            dsn = st.selectbox("Dataset",["Moons","Circles","Blobs","Iris","Classification"])
        ts  = st.slider("Test size",0.1,0.5,0.2,0.05,key="svm_ts")
        rs  = st.number_input("Seed",0,100,42,key="svm_rs")

    # data
    if src=="Upload CSV":
        result = csv_uploader("svm_csv","classification")
        if result is None: st.stop()
        X,y,f1,f2 = result
    else:
        X,y = builtin_clf(dsn,rs); f1,f2="Feature 1","Feature 2"

    sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=ts,random_state=int(rs))
    clf=svm.SVC(kernel=kernel,C=C,gamma=gamma,degree=degree,probability=True)
    clf.fit(Xtr,ytr); yp=clf.predict(Xte); acc=accuracy_score(yte,yp)

    tab1,tab2,tab3 = st.tabs(["📖 Concept","🎯 Decision Boundary","📊 Evaluation"])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""
<div class="card"><h3>🎯 Core Idea</h3><p>SVM finds the optimal hyperplane that maximizes the <strong>margin</strong> between classes.</p>
<div class="highlight">Goal: maximize the gap between classes</div></div>
<div class="card"><h3>📌 Support Vectors</h3><p>Only the closest points to the boundary matter — the rest are irrelevant.</p>
<div class="highlight">Sparse, robust representation</div></div>
<div class="card"><h3>🌀 Kernel Trick</h3><p>Maps data to higher dimensions so a linear boundary works on non-linear data.</p>
<div class="highlight">linear / rbf / poly / sigmoid</div></div>
""", unsafe_allow_html=True)
        with c2:
            fig,axes=plt.subplots(1,2,figsize=(9,4)); fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)
            np.random.seed(0)
            Xp=np.random.randn(15,2)+[2,2]; Xn=np.random.randn(15,2)+[-2,-2]
            xl=np.linspace(-5,5,100)
            axes[0].scatter(Xp[:,0],Xp[:,1],c='#4fc3f7',s=55,zorder=5,label='Class +1')
            axes[0].scatter(Xn[:,0],Xn[:,1],c='#ef5350',s=55,zorder=5,label='Class -1')
            axes[0].plot(xl,xl*0,color='#00e5ff',lw=2.5,label='Hyperplane')
            axes[0].fill_between(xl,-0.8,0.8,alpha=0.12,color='#00e5ff')
            axes[0].axhline(0.8,color='#00e5ff',lw=1,ls='--',alpha=0.5)
            axes[0].axhline(-0.8,color='#00e5ff',lw=1,ls='--',alpha=0.5)
            axes[0].set_xlim(-5,5); axes[0].set_ylim(-5,5)
            axes[0].set_title('Margin & Hyperplane',color='#4fc3f7',fontsize=10,fontweight='bold')
            axes[0].legend(fontsize=7,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            th=np.linspace(0,2*np.pi,50)
            axes[1].scatter(0.8*np.cos(th[:25]),0.8*np.sin(th[:25]),c='#4fc3f7',s=55,zorder=5)
            axes[1].scatter(1.8*np.cos(th),1.8*np.sin(th),c='#ef5350',s=55,zorder=5)
            axes[1].add_patch(plt.Circle((0,0),1.3,fill=False,color='#00e5ff',lw=2.5))
            axes[1].set_xlim(-2.5,2.5); axes[1].set_ylim(-2.5,2.5); axes[1].set_aspect('equal')
            axes[1].set_title('Kernel Trick (RBF)',color='#4fc3f7',fontsize=10,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        m1,m2,m3,m4=st.columns(4)
        for col,lbl,val in zip([m1,m2,m3,m4],
                                ["🎯 Accuracy","⚙️ Kernel","📏 C","Support Vectors"],
                                [f"{acc:.1%}",kernel.upper(),f"{C:.2f}",str(sum(clf.n_support_))]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,5)); style_ax(ax,fig)
        plot_boundary(ax,clf,Xs,y,Xtr,ytr,Xte,yte,f1,f2)
        sv=clf.support_vectors_
        ax.scatter(sv[:,0],sv[:,1],s=180,facecolors='none',edgecolors='#00e5ff',lw=2,zorder=10,label='Support Vecs')
        ax.set_title(f'SVM — {kernel.upper()} | C={C:.2f} | Acc={acc:.1%}',color='#4fc3f7',fontsize=12,fontweight='bold')
        ax.legend(fontsize=7,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        st.pyplot(fig); plt.close()

    with tab3:
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(5,4)); plot_confusion(ax,fig,yte,yp); st.pyplot(fig); plt.close()
        with c2:
            st.dataframe(pd.DataFrame(classification_report(yte,yp,output_dict=True)).T.round(3),
                         use_container_width=True,height=220)

# ══════════════════════════════════════════════════════════════
# ██  REGRESSION
# ══════════════════════════════════════════════════════════════
elif algo == "📈 Regression":
    st.markdown('<div class="main-header"><h1>📈 Regression</h1><p>Predict continuous values — Linear, Polynomial, Ridge, Lasso</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Regression Settings")
        reg_type = st.selectbox("Model", ["Linear","Polynomial","Ridge","Lasso"])
        if reg_type=="Polynomial":
            deg = st.slider("Degree",2,8,3)
        alpha = st.slider("Alpha (Ridge/Lasso)",0.001,10.0,1.0,0.001) if reg_type in ["Ridge","Lasso"] else 1.0
        st.markdown("---")
        st.markdown("### Data")
        src = st.radio("Source",["Built-in","Upload CSV"],key="reg_src")
        if src=="Built-in":
            dsn = st.selectbox("Dataset",["Linear","Polynomial","Noisy"])
        ts  = st.slider("Test size",0.1,0.5,0.2,0.05,key="reg_ts")
        rs  = st.number_input("Seed",0,100,42,key="reg_rs")

    if src=="Upload CSV":
        st.markdown("### 📂 Upload your CSV")
        uploaded=st.file_uploader("CSV file",type=["csv"],key="reg_csv")
        if uploaded is None:
            st.info("⬆️ Upload a CSV or switch to Built-in."); st.stop()
        df=pd.read_csv(uploaded)
        num_cols=df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols)<2: st.error("Need at least 2 numeric columns."); st.stop()
        c1,c2=st.columns(2)
        with c1: fx=st.selectbox("Feature (X)",num_cols,index=0,key="reg_fx")
        with c2: fy=st.selectbox("Target (Y)",num_cols,index=len(num_cols)-1,key="reg_fy")
        valid=df[[fx,fy]].dropna()
        X=valid[[fx]].values.astype(float); y=valid[fy].values.astype(float)
        st.success(f"✅ {len(X)} rows loaded")
        f_xlabel,f_ylabel=fx,fy
    else:
        X,y=builtin_reg(dsn,rs); f_xlabel,f_ylabel="X","y"

    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=ts,random_state=int(rs))

    if reg_type=="Linear":      model=LinearRegression()
    elif reg_type=="Polynomial": model=Pipeline([("poly",PolynomialFeatures(deg,include_bias=False)),("lr",LinearRegression())])
    elif reg_type=="Ridge":      model=Pipeline([("poly",PolynomialFeatures(3,include_bias=False)),("r",Ridge(alpha=alpha))])
    else:                        model=Pipeline([("poly",PolynomialFeatures(3,include_bias=False)),("l",Lasso(alpha=alpha))])

    model.fit(Xtr,ytr)
    yp_tr=model.predict(Xtr); yp_te=model.predict(Xte)
    r2_tr=r2_score(ytr,yp_tr); r2_te=r2_score(yte,yp_te)
    rmse=np.sqrt(mean_squared_error(yte,yp_te))

    tab1,tab2,tab3=st.tabs(["📖 Concept","📈 Fit","📊 Evaluation"])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""
<div class="card"><h3>📈 Linear Regression</h3><p>Fits a straight line: <strong>y = wx + b</strong> minimizing squared errors.</p>
<div class="highlight">Best for linearly correlated data</div></div>
<div class="card"><h3>🔢 Polynomial Regression</h3><p>Extends linear with higher-degree terms. Degree controls curve complexity.</p>
<div class="highlight">Watch out: high degree → overfitting</div></div>
<div class="card"><h3>🔒 Ridge & Lasso</h3><ul>
<li><strong>Ridge</strong>: shrinks coefficients (L2 penalty)</li>
<li><strong>Lasso</strong>: can zero-out coefficients (L1 penalty)</li>
<li><strong>Alpha</strong>: controls regularization strength</li></ul></div>
""", unsafe_allow_html=True)
        with c2:
            fig,axes=plt.subplots(1,2,figsize=(9,4)); fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)
            xs=np.linspace(-3,3,200).reshape(-1,1)
            np.random.seed(0)
            Xd=np.linspace(-3,3,60).reshape(-1,1)
            yd_lin=2*Xd.ravel()+np.random.randn(60)*0.8
            yd_pol=Xd.ravel()**3-2*Xd.ravel()+np.random.randn(60)*1.5
            axes[0].scatter(Xd,yd_lin,c='#4fc3f7',s=25,alpha=0.7,label='data')
            m=LinearRegression().fit(Xd,yd_lin)
            axes[0].plot(xs,m.predict(xs),color='#00e5ff',lw=2.5,label='Linear fit')
            axes[0].set_title('Linear Regression',color='#4fc3f7',fontsize=10,fontweight='bold')
            axes[0].legend(fontsize=8,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            axes[1].scatter(Xd,yd_pol,c='#4fc3f7',s=25,alpha=0.7,label='data')
            for d2,col,lbl in [(2,'#ffa726','deg 2'),(4,'#00e5ff','deg 4'),(8,'#ef5350','deg 8')]:
                p=Pipeline([("p",PolynomialFeatures(d2)),("r",LinearRegression())]).fit(Xd,yd_pol)
                axes[1].plot(xs,p.predict(xs),color=col,lw=1.8,label=lbl)
            axes[1].set_title('Polynomial Degrees',color='#4fc3f7',fontsize=10,fontweight='bold')
            axes[1].legend(fontsize=8,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        m1,m2,m3,m4=st.columns(4)
        for col,lbl,val in zip([m1,m2,m3,m4],
                                ["📈 Model","R² Train","R² Test","RMSE"],
                                [reg_type,f"{r2_tr:.3f}",f"{r2_te:.3f}",f"{rmse:.3f}"]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(9,5)); style_ax(ax,fig)
        ax.scatter(Xtr,ytr,c='#4fc3f7',s=35,alpha=0.7,label='Train data')
        ax.scatter(Xte,yte,c='#ffa726',s=55,marker='D',alpha=0.9,label='Test data')
        xs=np.linspace(X.min(),X.max(),300).reshape(-1,1)
        ax.plot(xs,model.predict(xs),color='#00e5ff',lw=2.5,label=f'{reg_type} fit')
        ax.set_xlabel(f_xlabel,color='#8892b0'); ax.set_ylabel(f_ylabel,color='#8892b0')
        ax.set_title(f'{reg_type} Regression | R²={r2_te:.3f} | RMSE={rmse:.3f}',
                     color='#4fc3f7',fontsize=12,fontweight='bold')
        ax.legend(fontsize=9,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        st.pyplot(fig); plt.close()

    with tab3:
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(5,4)); style_ax(ax,fig)
            ax.scatter(yte,yp_te,c='#4fc3f7',s=40,alpha=0.7,edgecolors='white',lw=0.3)
            mn,mx=min(yte.min(),yp_te.min()),max(yte.max(),yp_te.max())
            ax.plot([mn,mx],[mn,mx],'--',color='#00e5ff',lw=2,label='Perfect fit')
            ax.set_xlabel("Actual",color='#8892b0'); ax.set_ylabel("Predicted",color='#8892b0')
            ax.set_title("Actual vs Predicted",color='#4fc3f7',fontsize=11,fontweight='bold')
            ax.legend(fontsize=9,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            st.pyplot(fig); plt.close()
        with c2:
            fig,ax=plt.subplots(figsize=(5,4)); style_ax(ax,fig)
            res=yte-yp_te
            ax.scatter(yp_te,res,c='#4fc3f7',s=40,alpha=0.7,edgecolors='white',lw=0.3)
            ax.axhline(0,color='#00e5ff',lw=2,ls='--')
            ax.set_xlabel("Predicted",color='#8892b0'); ax.set_ylabel("Residual",color='#8892b0')
            ax.set_title("Residual Plot",color='#4fc3f7',fontsize=11,fontweight='bold')
            st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# ██  DECISION TREE / RANDOM FOREST
# ══════════════════════════════════════════════════════════════
elif algo == "🌳 Decision Tree / RF":
    st.markdown('<div class="main-header"><h1>🌳 Decision Tree & Random Forest</h1><p>Rule-based splitting meets ensemble power</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Tree Settings")
        model_type=st.radio("Model",["Decision Tree","Random Forest"])
        max_depth =st.slider("Max Depth",1,15,4)
        if model_type=="Random Forest":
            n_est=st.slider("# Estimators",10,200,100,10)
        st.markdown("---")
        st.markdown("### Data")
        src=st.radio("Source",["Built-in","Upload CSV"],key="tree_src")
        if src=="Built-in":
            dsn=st.selectbox("Dataset",["Moons","Circles","Blobs","Iris","Classification"])
        ts =st.slider("Test size",0.1,0.5,0.2,0.05,key="tree_ts")
        rs =st.number_input("Seed",0,100,42,key="tree_rs")

    if src=="Upload CSV":
        result=csv_uploader("tree_csv","classification")
        if result is None: st.stop()
        X,y,f1,f2=result
    else:
        X,y=builtin_clf(dsn,rs); f1,f2="Feature 1","Feature 2"

    sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=ts,random_state=int(rs))

    if model_type=="Decision Tree":
        clf=DecisionTreeClassifier(max_depth=max_depth,random_state=int(rs))
    else:
        clf=RandomForestClassifier(n_estimators=n_est,max_depth=max_depth,random_state=int(rs))
    clf.fit(Xtr,ytr); yp=clf.predict(Xte); acc=accuracy_score(yte,yp)

    tab1,tab2,tab3=st.tabs(["📖 Concept","🌳 Boundary","📊 Evaluation"])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""
<div class="card"><h3>🌳 Decision Tree</h3><p>Recursively splits data using the best feature threshold. Each leaf represents a class.</p>
<div class="highlight">Interpretable — can be read as rules</div></div>
<div class="card"><h3>🌲🌲 Random Forest</h3><p>Trains hundreds of trees on random subsets of data & features, then votes.</p>
<div class="highlight">Reduces overfitting, much more robust</div></div>
<div class="card"><h3>⚙️ Key Parameters</h3><ul>
<li><strong>Max Depth</strong>: limits tree growth → controls overfitting</li>
<li><strong># Estimators</strong>: more trees = more stable (but slower)</li>
<li>Deep trees overfit; shallow trees underfit</li></ul></div>
""", unsafe_allow_html=True)
        with c2:
            fig,axes=plt.subplots(1,2,figsize=(9,4.5)); fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)
            np.random.seed(42)
            Xm,ym=datasets.make_moons(100,noise=0.2,random_state=42)
            sc2=StandardScaler(); Xms=sc2.fit_transform(Xm)
            h=0.05; xmn,xmx=Xms[:,0].min()-.5,Xms[:,0].max()+.5
            ymn2,ymx2=Xms[:,1].min()-.5,Xms[:,1].max()+.5
            xx2,yy2=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn2,ymx2,h))
            for ax,md,title in zip(axes,[2,8],['Depth=2 (simple)','Depth=8 (overfit risk)']):
                dt=DecisionTreeClassifier(max_depth=md).fit(Xms,ym)
                Z=dt.predict(np.c_[xx2.ravel(),yy2.ravel()]).reshape(xx2.shape)
                ax.contourf(xx2,yy2,Z,cmap=ListedColormap(['#ef535018','#4fc3f718']),alpha=0.8)
                ax.contour(xx2,yy2,Z,colors=['#00e5ff'],linewidths=1.5,alpha=0.7)
                ax.scatter(Xms[ym==0,0],Xms[ym==0,1],c='#ef5350',s=35,alpha=0.8,edgecolors='white',lw=0.3)
                ax.scatter(Xms[ym==1,0],Xms[ym==1,1],c='#4fc3f7',s=35,alpha=0.8,edgecolors='white',lw=0.3)
                ax.set_title(title,color='#4fc3f7',fontsize=10,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        m1,m2,m3,m4=st.columns(4)
        n_leaves=clf.get_n_leaves() if hasattr(clf,'get_n_leaves') else "—"
        for col,lbl,val in zip([m1,m2,m3,m4],
                                ["🎯 Accuracy","🌳 Model","📏 Max Depth","🍃 Leaves"],
                                [f"{acc:.1%}",model_type,str(max_depth),str(n_leaves)]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,5)); style_ax(ax,fig)
        plot_boundary(ax,clf,Xs,y,Xtr,ytr,Xte,yte,f1,f2)
        ax.set_title(f'{model_type} | Depth={max_depth} | Acc={acc:.1%}',
                     color='#4fc3f7',fontsize=12,fontweight='bold')
        st.pyplot(fig); plt.close()

        if model_type=="Random Forest":
            st.markdown("### 🏆 Feature Importance")
            fi=clf.feature_importances_
            fig,ax=plt.subplots(figsize=(5,2.5)); style_ax(ax,fig)
            ax.barh([f2,f1],fi[::-1],color=['#4fc3f7','#00e5ff'])
            ax.set_xlabel("Importance",color='#8892b0')
            ax.set_title("Feature Importances",color='#4fc3f7',fontsize=10,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(5,4)); plot_confusion(ax,fig,yte,yp); st.pyplot(fig); plt.close()
        with c2:
            st.dataframe(pd.DataFrame(classification_report(yte,yp,output_dict=True)).T.round(3),
                         use_container_width=True,height=220)

# ══════════════════════════════════════════════════════════════
# ██  K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════
elif algo == "🔵 K-Means Clustering":
    st.markdown('<div class="main-header"><h1>🔵 K-Means Clustering</h1><p>Unsupervised grouping — discover hidden structure in data</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### K-Means Settings")
        k       = st.slider("Number of Clusters (K)",2,8,3)
        max_it  = st.slider("Max Iterations",10,500,300,10)
        n_init  = st.slider("N Init",1,20,10)
        st.markdown("---")
        st.markdown("### Data")
        src=st.radio("Source",["Built-in","Upload CSV"],key="km_src")
        if src=="Built-in":
            dsn=st.selectbox("Dataset",["Blobs","Moons","Circles","Iris"])
        rs=st.number_input("Seed",0,100,42,key="km_rs")

    if src=="Upload CSV":
        result=csv_uploader("km_csv","clustering")
        if result is None: st.stop()
        X,_,f1,f2=result
    else:
        X,y_true=builtin_clf(dsn,rs); f1,f2="Feature 1","Feature 2"

    sc=StandardScaler(); Xs=sc.fit_transform(X)
    km=KMeans(n_clusters=k,max_iter=max_it,n_init=n_init,random_state=int(rs))
    labels=km.fit_predict(Xs)
    inertia=km.inertia_

    tab1,tab2,tab3=st.tabs(["📖 Concept","🔵 Clusters","📊 Elbow Curve"])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""
<div class="card"><h3>🔵 What is K-Means?</h3><p>Partitions data into K clusters by minimizing within-cluster variance. No labels needed!</p>
<div class="highlight">Unsupervised — finds natural groupings</div></div>
<div class="card"><h3>⚙️ The Algorithm</h3><ul>
<li>1. Place K centroids randomly</li>
<li>2. Assign each point to nearest centroid</li>
<li>3. Move centroids to cluster mean</li>
<li>4. Repeat until convergence</li></ul></div>
<div class="card"><h3>📏 Choosing K</h3><p>Use the <strong>Elbow Method</strong>: plot inertia vs K, pick the "elbow" where improvement slows.</p>
<div class="highlight">The elbow = best trade-off</div></div>
""", unsafe_allow_html=True)
        with c2:
            fig,axes=plt.subplots(1,2,figsize=(9,4)); fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)
            Xb,_=datasets.make_blobs(150,centers=3,random_state=0)
            sc3=StandardScaler(); Xbs=sc3.fit_transform(Xb)
            km3=KMeans(3,random_state=0,n_init=10).fit(Xbs)
            for i in range(3):
                axes[0].scatter(Xbs[km3.labels_==i,0],Xbs[km3.labels_==i,1],
                                c=COLORS[i],s=35,alpha=0.8,edgecolors='white',lw=0.3)
            axes[0].scatter(km3.cluster_centers_[:,0],km3.cluster_centers_[:,1],
                            c='#00e5ff',s=200,marker='*',zorder=10,edgecolors='white',lw=1,label='Centroids')
            axes[0].set_title('K=3 Clusters',color='#4fc3f7',fontsize=10,fontweight='bold')
            axes[0].legend(fontsize=8,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
            ks=range(1,9); inertias=[KMeans(ki,n_init=10,random_state=0).fit(Xbs).inertia_ for ki in ks]
            axes[1].plot(ks,inertias,'o-',color='#4fc3f7',lw=2)
            axes[1].set_xlabel("K",color='#8892b0'); axes[1].set_ylabel("Inertia",color='#8892b0')
            axes[1].set_title("Elbow Curve",color='#4fc3f7',fontsize=10,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        m1,m2,m3=st.columns(3)
        for col,lbl,val in zip([m1,m2,m3],["🔵 K","📉 Inertia","📦 Samples"],
                                [str(k),f"{inertia:.1f}",str(len(Xs))]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,5)); style_ax(ax,fig)
        for i in range(k):
            mask=labels==i
            ax.scatter(Xs[mask,0],Xs[mask,1],c=COLORS[i%len(COLORS)],s=45,
                       alpha=0.8,edgecolors='white',lw=0.3,label=f'Cluster {i}')
        ctrs=km.cluster_centers_
        ax.scatter(ctrs[:,0],ctrs[:,1],c='#00e5ff',s=250,marker='*',zorder=10,
                   edgecolors='white',lw=1,label='Centroids')
        ax.set_xlabel(f"{f1} (norm.)",color='#8892b0'); ax.set_ylabel(f"{f2} (norm.)",color='#8892b0')
        ax.set_title(f'K-Means | K={k} | Inertia={inertia:.1f}',color='#4fc3f7',fontsize=12,fontweight='bold')
        ax.legend(fontsize=8,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        st.pyplot(fig); plt.close()

    with tab3:
        st.markdown("### 📊 Elbow Curve — Find the Best K")
        ks=range(1,11)
        inertias=[KMeans(ki,max_iter=max_it,n_init=n_init,random_state=int(rs)).fit(Xs).inertia_ for ki in ks]
        fig,ax=plt.subplots(figsize=(8,4)); style_ax(ax,fig)
        ax.plot(ks,inertias,'o-',color='#4fc3f7',lw=2.5,markersize=8)
        ax.axvline(k,color='#00e5ff',lw=2,ls='--',alpha=0.8,label=f'Current K={k}')
        ax.set_xlabel("Number of Clusters (K)",color='#8892b0')
        ax.set_ylabel("Inertia",color='#8892b0')
        ax.set_title("Elbow Curve",color='#4fc3f7',fontsize=12,fontweight='bold')
        ax.legend(fontsize=9,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        st.pyplot(fig); plt.close()
        st.info("💡 Pick the K where the curve bends sharply — that's your elbow.")

# ══════════════════════════════════════════════════════════════
# ██  KNN
# ══════════════════════════════════════════════════════════════
elif algo == "👥 KNN":
    st.markdown('<div class="main-header"><h1>👥 K-Nearest Neighbors</h1><p>Classify by majority vote of the K closest points</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### KNN Settings")
        k       = st.slider("K (neighbors)",1,25,5)
        weights = st.radio("Weights",["uniform","distance"])
        metric  = st.selectbox("Distance Metric",["euclidean","manhattan","minkowski"])
        st.markdown("---")
        st.markdown("### Data")
        src=st.radio("Source",["Built-in","Upload CSV"],key="knn_src")
        if src=="Built-in":
            dsn=st.selectbox("Dataset",["Moons","Circles","Blobs","Iris","Classification"])
        ts =st.slider("Test size",0.1,0.5,0.2,0.05,key="knn_ts")
        rs =st.number_input("Seed",0,100,42,key="knn_rs")

    if src=="Upload CSV":
        result=csv_uploader("knn_csv","classification")
        if result is None: st.stop()
        X,y,f1,f2=result
    else:
        X,y=builtin_clf(dsn,rs); f1,f2="Feature 1","Feature 2"

    sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=ts,random_state=int(rs))
    clf=KNeighborsClassifier(n_neighbors=k,weights=weights,metric=metric)
    clf.fit(Xtr,ytr); yp=clf.predict(Xte); acc=accuracy_score(yte,yp)

    tab1,tab2,tab3=st.tabs(["📖 Concept","👥 Boundary","📊 Evaluation"])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""
<div class="card"><h3>👥 How KNN Works</h3><p>To classify a point, find its K nearest neighbors and take a majority vote.</p>
<div class="highlight">Non-parametric — no training, just memory</div></div>
<div class="card"><h3>⚙️ Choosing K</h3><ul>
<li>K=1 → very complex, overfits</li>
<li>K too large → underfits, blurry boundary</li>
<li>Odd K avoids ties in binary classification</li></ul></div>
<div class="card"><h3>📏 Distance Metrics</h3><ul>
<li><strong>Euclidean</strong>: straight-line distance</li>
<li><strong>Manhattan</strong>: grid-like (L1)</li>
<li><strong>Minkowski</strong>: generalization of both</li></ul></div>
""", unsafe_allow_html=True)
        with c2:
            fig,axes=plt.subplots(1,3,figsize=(12,4)); fig.patch.set_facecolor('#0a0f1e')
            for ax in axes: style_ax(ax)
            Xm,ym=datasets.make_moons(120,noise=0.2,random_state=0)
            sc4=StandardScaler(); Xms=sc4.fit_transform(Xm)
            h=0.06; xmn,xmx=Xms[:,0].min()-.5,Xms[:,0].max()+.5
            ymn3,ymx3=Xms[:,1].min()-.5,Xms[:,1].max()+.5
            xx3,yy3=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn3,ymx3,h))
            for ax,kk,title in zip(axes,[1,5,20],['K=1 (overfit)','K=5 (balanced)','K=20 (underfit)']):
                kn=KNeighborsClassifier(n_neighbors=kk).fit(Xms,ym)
                Z=kn.predict(np.c_[xx3.ravel(),yy3.ravel()]).reshape(xx3.shape)
                ax.contourf(xx3,yy3,Z,cmap=ListedColormap(['#ef535018','#4fc3f718']),alpha=0.8)
                ax.contour(xx3,yy3,Z,colors=['#00e5ff'],linewidths=1.5,alpha=0.7)
                ax.scatter(Xms[ym==0,0],Xms[ym==0,1],c='#ef5350',s=30,alpha=0.8,edgecolors='white',lw=0.3)
                ax.scatter(Xms[ym==1,0],Xms[ym==1,1],c='#4fc3f7',s=30,alpha=0.8,edgecolors='white',lw=0.3)
                ax.set_title(title,color='#4fc3f7',fontsize=9,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        m1,m2,m3,m4=st.columns(4)
        for col,lbl,val in zip([m1,m2,m3,m4],
                                ["🎯 Accuracy","👥 K","⚖️ Weights","📏 Metric"],
                                [f"{acc:.1%}",str(k),weights,metric]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,5)); style_ax(ax,fig)
        plot_boundary(ax,clf,Xs,y,Xtr,ytr,Xte,yte,f1,f2)
        ax.set_title(f'KNN | K={k} | {weights} | Acc={acc:.1%}',color='#4fc3f7',fontsize=12,fontweight='bold')
        st.pyplot(fig); plt.close()

        st.markdown("### 📈 Accuracy vs K")
        ks=range(1,31); accs=[accuracy_score(yte,KNeighborsClassifier(ki,weights=weights,metric=metric).fit(Xtr,ytr).predict(Xte)) for ki in ks]
        fig,ax=plt.subplots(figsize=(8,3.5)); style_ax(ax,fig)
        ax.plot(ks,accs,'o-',color='#4fc3f7',lw=2,markersize=5)
        ax.axvline(k,color='#00e5ff',lw=2,ls='--',alpha=0.9,label=f'Current K={k}')
        ax.set_xlabel("K",color='#8892b0'); ax.set_ylabel("Test Accuracy",color='#8892b0')
        ax.set_title("Accuracy vs K",color='#4fc3f7',fontsize=11,fontweight='bold')
        ax.legend(fontsize=9,facecolor='#0d1b2a',labelcolor='#ccd6f6',edgecolor='#1e3a5f')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(5,4)); plot_confusion(ax,fig,yte,yp); st.pyplot(fig); plt.close()
        with c2:
            st.dataframe(pd.DataFrame(classification_report(yte,yp,output_dict=True)).T.round(3),
                         use_container_width=True,height=220)

# ══════════════════════════════════════════════════════════════
# ██  MLP / NEURAL NETWORK
# ══════════════════════════════════════════════════════════════
elif algo == "🧠 Neural Network (MLP)":
    st.markdown('<div class="main-header"><h1>🧠 Neural Network (MLP)</h1><p>Multi-layer perceptron — universal function approximator</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### MLP Settings")
        h1     = st.slider("Layer 1 neurons",4,128,64,4)
        h2     = st.slider("Layer 2 neurons",0,128,32,4)
        h3     = st.slider("Layer 3 neurons",0,64,0,4)
        layers = tuple(n for n in [h1,h2,h3] if n>0)
        act    = st.selectbox("Activation",["relu","tanh","logistic"])
        lr     = st.select_slider("Learning Rate",options=[0.0001,0.001,0.01,0.1],value=0.001)
        max_it = st.slider("Max Iterations",50,500,200,50)
        st.markdown("---")
        st.markdown("### Data")
        src=st.radio("Source",["Built-in","Upload CSV"],key="mlp_src")
        if src=="Built-in":
            dsn=st.selectbox("Dataset",["Moons","Circles","Blobs","Iris","Classification"])
        ts =st.slider("Test size",0.1,0.5,0.2,0.05,key="mlp_ts")
        rs =st.number_input("Seed",0,100,42,key="mlp_rs")

    if src=="Upload CSV":
        result=csv_uploader("mlp_csv","classification")
        if result is None: st.stop()
        X,y,f1,f2=result
    else:
        X,y=builtin_clf(dsn,rs); f1,f2="Feature 1","Feature 2"

    sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=ts,random_state=int(rs))
    clf=MLPClassifier(hidden_layer_sizes=layers,activation=act,
                      learning_rate_init=lr,max_iter=max_it,random_state=int(rs))
    clf.fit(Xtr,ytr); yp=clf.predict(Xte); acc=accuracy_score(yte,yp)

    tab1,tab2,tab3=st.tabs(["📖 Concept","🧠 Boundary","📊 Evaluation"])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            st.markdown(f"""
<div class="card"><h3>🧠 Multi-Layer Perceptron</h3><p>Stacked layers of neurons with non-linear activations. Learns complex patterns via backpropagation.</p>
<div class="highlight">Universal approximator — fits any function</div></div>
<div class="card"><h3>⚙️ Your Architecture</h3><ul>
<li>Input: 2 features</li>
{''.join([f'<li>Hidden Layer {i+1}: {n} neurons ({act})</li>' for i,n in enumerate(layers)])}
<li>Output: {len(np.unique(y))} classes (softmax)</li></ul></div>
<div class="card"><h3>🎛️ Key Settings</h3><ul>
<li><strong>Activation</strong>: relu (fast), tanh (smooth), logistic (classic)</li>
<li><strong>Learning Rate</strong>: step size for gradient descent</li>
<li><strong>Iterations</strong>: training epochs</li></ul></div>
""", unsafe_allow_html=True)
        with c2:
            # Network diagram
            fig,ax=plt.subplots(figsize=(8,4.5)); fig.patch.set_facecolor('#0a0f1e'); ax.axis('off')
            ax.set_facecolor('#0a0f1e')
            all_layers=[2]+list(layers)+[len(np.unique(y))]
            max_n=max(all_layers); lx=np.linspace(0.1,0.9,len(all_layers))
            node_pos={}
            for li,n in enumerate(all_layers):
                ys=np.linspace(0.5-(n-1)*0.08,0.5+(n-1)*0.08,n)
                for ni,y_pos in enumerate(ys):
                    node_pos[(li,ni)]=(lx[li],y_pos)
                    c_node='#4fc3f7' if li==0 else ('#00e5ff' if li==len(all_layers)-1 else '#112240')
                    ec='#4fc3f7' if li in [0,len(all_layers)-1] else '#1e3a5f'
                    circle=plt.Circle((lx[li],y_pos),0.025,color=c_node,ec=ec,lw=1.5,zorder=5)
                    ax.add_patch(circle)
            for li in range(len(all_layers)-1):
                n1,n2=all_layers[li],all_layers[li+1]
                for ni in range(n1):
                    for nj in range(n2):
                        x1,y1=node_pos[(li,ni)]; x2,y2=node_pos[(li+1,nj)]
                        ax.plot([x1,x2],[y1,y2],color='#1e3a5f',lw=0.6,alpha=0.5,zorder=1)
            labels=['Input']+[f'Hidden {i+1}' for i in range(len(layers))]+['Output']
            for li,(label,x) in enumerate(zip(labels,lx)):
                ax.text(x,0.08,label,ha='center',va='center',color='#8892b0',fontsize=8,
                        fontfamily='monospace')
                ax.text(x,0.97,str(all_layers[li]),ha='center',va='center',color='#4fc3f7',
                        fontsize=9,fontweight='bold',fontfamily='monospace')
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_title(f'Network: {" → ".join(map(str,all_layers))}',
                         color='#4fc3f7',fontsize=11,fontweight='bold')
            st.pyplot(fig); plt.close()

    with tab2:
        m1,m2,m3,m4=st.columns(4)
        for col,lbl,val in zip([m1,m2,m3,m4],
                                ["🎯 Accuracy","🔢 Layers","⚡ Activation","🔄 Iters"],
                                [f"{acc:.1%}",str(len(layers)),act,str(max_it)]):
            with col: st.markdown(metric_box(val,lbl), unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,5)); style_ax(ax,fig)
        plot_boundary(ax,clf,Xs,y,Xtr,ytr,Xte,yte,f1,f2)
        ax.set_title(f'MLP {list(layers)} | {act} | Acc={acc:.1%}',color='#4fc3f7',fontsize=12,fontweight='bold')
        st.pyplot(fig); plt.close()

        if hasattr(clf,'loss_curve_'):
            st.markdown("### 📉 Training Loss Curve")
            fig,ax=plt.subplots(figsize=(8,3.5)); style_ax(ax,fig)
            ax.plot(clf.loss_curve_,color='#4fc3f7',lw=2)
            ax.set_xlabel("Iteration",color='#8892b0'); ax.set_ylabel("Loss",color='#8892b0')
            ax.set_title("Loss During Training",color='#4fc3f7',fontsize=11,fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(5,4)); plot_confusion(ax,fig,yte,yp); st.pyplot(fig); plt.close()
        with c2:
            st.dataframe(pd.DataFrame(classification_report(yte,yp,output_dict=True)).T.round(3),
                         use_container_width=True,height=220)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#8892b0;font-family:'JetBrains Mono',monospace;font-size:0.75rem;padding:0.4rem 0;">
    ML Explorer &nbsp;·&nbsp; SVM · Regression · Decision Tree · K-Means · KNN · MLP &nbsp;·&nbsp; 🤖
</div>
""", unsafe_allow_html=True)
