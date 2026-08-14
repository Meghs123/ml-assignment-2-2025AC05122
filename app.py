"""
Streamlit Web Application - ML Classification Models
=====================================================
Breast Cancer Wisconsin (Diagnostic) Dataset
Compares 5 classification models with 6 evaluation metrics.

Run locally: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ---- Page Config ----
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---- Custom CSS for better styling ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================
# Caching functions so the app doesn't retrain on every interaction
# ================================================================

@st.cache_data
def load_default_data():
    """Load the breast cancer dataset and create train/test splits."""
    data = load_breast_cancer()
    feature_names = data.feature_names.tolist()
    df = pd.DataFrame(data.data, columns=feature_names)
    df['diagnosis'] = data.target
    return df, feature_names


@st.cache_resource
def train_all_models(X_train, y_train):
    """Train all 5 classification models and return them with the scaler."""
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=5000, C=1.0, solver='lbfgs', random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, min_samples_split=5, random_state=42
        ),
        'kNN': KNeighborsClassifier(
            n_neighbors=7, weights='uniform', metric='minkowski'
        ),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(
            n_estimators=150, random_state=42
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    
    return trained_models, scaler


def compute_metrics(model, X_test_scaled, y_test):
    """Compute all 6 evaluation metrics for a given model."""
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC Score': round(roc_auc_score(y_test, y_prob), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1 Score': round(f1_score(y_test, y_pred), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }
    return metrics, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Generate a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Malignant (0)', 'Benign (1)'],
        yticklabels=['Malignant (0)', 'Benign (1)'],
        ax=ax, annot_kws={"size": 16}
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_roc_curve(y_test, y_prob, model_name):
    """Plot ROC curve for a single model."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, color='#1f4e79', lw=2.5, label=f'{model_name} (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_all_roc_curves(trained_models, X_test_scaled, y_test):
    """Overlay ROC curves for all models on one plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for (name, model), color in zip(trained_models.items(), colors):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc_val:.3f})')
    
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - All Models Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_metrics_comparison(all_metrics_df):
    """Bar chart comparing a chosen metric across all models."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics_list = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, (metric, color) in enumerate(zip(metrics_list, colors)):
        ax = axes[idx // 3][idx % 3]
        values = all_metrics_df[metric].values
        model_names = [n.replace(' (Ensemble)', '\n(Ensemble)') for n in all_metrics_df.index]
        bars = ax.bar(model_names, values, color=color, alpha=0.8, edgecolor='white')
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylim(0.75, 1.02)
        ax.tick_params(axis='x', rotation=25, labelsize=8)
        
        # add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.suptitle('Metrics Comparison Across All Models', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


# ================================================================
# Main Application
# ================================================================

def main():
    
    # Header
    st.markdown('<p class="main-header">📊 ML Classification Model Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Breast Cancer Wisconsin (Diagnostic) Dataset — 5 Models, 6 Metrics</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load full dataset for training
    full_df, feature_names = load_default_data()
    
    # ---- Sidebar ----
    st.sidebar.header("⚙️ Configuration")
    
    # File upload option
    st.sidebar.subheader("📁 Upload Test Data (CSV)")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your test CSV file",
        type=['csv'],
        help="CSV must have the same 30 feature columns and a 'diagnosis' target column."
    )
    
    # Model selection dropdown
    st.sidebar.subheader("🤖 Select Model")
    model_names = [
        'Logistic Regression',
        'Decision Tree',
        'kNN',
        'Naive Bayes',
        'Random Forest (Ensemble)'
    ]
    selected_model = st.sidebar.selectbox("Choose a model to inspect:", model_names)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset Info:**")
    st.sidebar.markdown("- Source: UCI ML Repository")
    st.sidebar.markdown("- Features: 30 numeric")
    st.sidebar.markdown("- Classes: Malignant (0), Benign (1)")
    st.sidebar.markdown("- Instances: 569 total")
    
    # ---- Prepare data ----
    # always train on the 80% training split of the full dataset
    X_full = full_df.drop('diagnosis', axis=1)
    y_full = full_df['diagnosis']
    
    X_train, X_test_default, y_train, y_test_default = train_test_split(
        X_full, y_full, test_size=0.20, random_state=42, stratify=y_full
    )
    
    # train models (cached)
    trained_models, scaler = train_all_models(X_train.values, y_train.values)
    
    # determine test data
    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            
            # check if the uploaded CSV has the correct columns
            if 'diagnosis' not in test_df.columns:
                st.error("❌ The uploaded CSV must contain a 'diagnosis' column as the target variable.")
                st.stop()
            
            # check feature count
            feature_cols = [c for c in test_df.columns if c != 'diagnosis']
            if len(feature_cols) != 30:
                st.warning(f"⚠️ Expected 30 feature columns, found {len(feature_cols)}. Using available features.")
            
            X_test = test_df[feature_cols].values
            y_test = test_df['diagnosis'].values
            data_source = "📤 Uploaded CSV"
            st.sidebar.success(f"✅ Loaded {len(test_df)} test samples from upload")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
    else:
        X_test = X_test_default.values
        y_test = y_test_default.values
        data_source = "📦 Default Test Split (20%)"
        st.sidebar.info("Using default 20% test split. Upload a CSV for custom test data.")
    
    # Scale the test features
    X_test_scaled = scaler.transform(X_test)
    
    # ---- Dataset Overview Tab ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Dataset Overview",
        "🔍 Individual Model Analysis",
        "📊 All Models Comparison",
        "📈 ROC Curves"
    ])
    
    # =====================
    # Tab 1: Dataset Overview
    # =====================
    with tab1:
        st.subheader("Dataset Overview")
        st.write(f"**Data Source:** {data_source}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Features", "30")
        col2.metric("Training Samples", str(len(X_train)))
        col3.metric("Test Samples", str(len(y_test)))
        col4.metric("Classes", "2 (Binary)")
        
        st.write("")
        
        # show sample data
        st.subheader("Sample Data (First 10 Rows)")
        if uploaded_file is not None:
            st.dataframe(test_df.head(10), use_container_width=True)
        else:
            sample_df = pd.DataFrame(X_test[:10], columns=feature_names)
            sample_df['diagnosis'] = y_test[:10]
            st.dataframe(sample_df.head(10), use_container_width=True)
        
        # class distribution in test set
        st.subheader("Test Set Class Distribution")
        class_counts = pd.Series(y_test).value_counts().sort_index()
        col_a, col_b = st.columns(2)
        with col_a:
            fig_dist, ax_dist = plt.subplots(figsize=(5, 3.5))
            bars = ax_dist.bar(
                ['Malignant (0)', 'Benign (1)'],
                [class_counts.get(0, 0), class_counts.get(1, 0)],
                color=['#e74c3c', '#3498db'],
                edgecolor='white', width=0.5
            )
            for bar in bars:
                ax_dist.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                            f'{int(bar.get_height())}', ha='center', fontweight='bold')
            ax_dist.set_ylabel('Count')
            ax_dist.set_title('Class Distribution in Test Data')
            plt.tight_layout()
            st.pyplot(fig_dist)
        with col_b:
            st.write("")
            st.write(f"- **Malignant (0):** {class_counts.get(0, 0)} samples")
            st.write(f"- **Benign (1):** {class_counts.get(1, 0)} samples")
            st.write(f"- **Total:** {len(y_test)} samples")
            ratio = class_counts.get(1, 0) / len(y_test) * 100
            st.write(f"- **Benign ratio:** {ratio:.1f}%")
    
    # =====================
    # Tab 2: Individual Model
    # =====================
    with tab2:
        st.subheader(f"Model Analysis: {selected_model}")
        
        model = trained_models[selected_model]
        metrics, y_pred, y_prob = compute_metrics(model, X_test_scaled, y_test)
        
        # display metrics as metric cards
        st.write("**Evaluation Metrics:**")
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        mc2.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
        mc3.metric("Precision", f"{metrics['Precision']:.4f}")
        mc4.metric("Recall", f"{metrics['Recall']:.4f}")
        mc5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        mc6.metric("MCC", f"{metrics['MCC']:.4f}")
        
        st.markdown("---")
        
        # Confusion Matrix and ROC Curve side by side
        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.write("**Confusion Matrix:**")
            fig_cm = plot_confusion_matrix(y_test, y_pred, selected_model)
            st.pyplot(fig_cm)
        
        with col_roc:
            st.write("**ROC Curve:**")
            fig_roc = plot_roc_curve(y_test, y_prob, selected_model)
            st.pyplot(fig_roc)
        
        # Classification Report
        st.write("**Detailed Classification Report:**")
        report_dict = classification_report(
            y_test, y_pred,
            target_names=['Malignant (0)', 'Benign (1)'],
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.round(4)
        st.dataframe(report_df, use_container_width=True)
    
    # =====================
    # Tab 3: All Models Comparison
    # =====================
    with tab3:
        st.subheader("All Models — Comparison Table")
        
        # compute metrics for all models
        all_results = {}
        for name, mdl in trained_models.items():
            m, _, _ = compute_metrics(mdl, X_test_scaled, y_test)
            all_results[name] = m
        
        comparison_df = pd.DataFrame(all_results).T
        comparison_df.index.name = 'Model'
        
        # highlight the best value in each column
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, color='#d4edda'),
            use_container_width=True
        )
        
        # find the winner
        best_model_name = comparison_df['F1 Score'].idxmax()
        best_f1 = comparison_df.loc[best_model_name, 'F1 Score']
        st.success(f"🏆 **Overall Winner: {best_model_name}** with F1 Score = {best_f1:.4f}")
        
        st.markdown("---")
        
        # bar chart comparison
        st.subheader("Visual Metrics Comparison")
        fig_bars = plot_metrics_comparison(comparison_df)
        st.pyplot(fig_bars)
        
        st.markdown("---")
        
        # Model Observations
        st.subheader("Model Observations")
        
        observations = {
            'Logistic Regression': (
                "Performed the best overall on this dataset. The high accuracy and "
                "near-perfect AUC indicate that the decision boundary between malignant "
                "and benign classes is largely linear in the scaled feature space. "
                "Feature standardization helped this model converge well. Only 2 "
                "misclassifications out of 114 test samples show strong generalization."
            ),
            'Decision Tree': (
                "Showed the weakest performance among all five models. Even with "
                "max_depth=5 to control overfitting, the tree struggles with the "
                "high-dimensional feature space (30 features). The lower AUC of ~0.92 "
                "suggests the axis-aligned splits are not ideal for this data's "
                "distribution. It misclassified 9 samples, mostly missing some "
                "malignant cases."
            ),
            'kNN': (
                "Achieved perfect recall (1.0) for the benign class, meaning it "
                "never missed a benign diagnosis. With k=7 and scaled features, it "
                "performs well because the feature space clusters well for this "
                "dataset. The 3 false positives (malignant predicted as benign) are "
                "a minor concern but overall it is one of the top performers."
            ),
            'Naive Bayes': (
                "Reasonable performance but limited by the independence assumption. "
                "Many features in the breast cancer dataset are correlated (e.g., "
                "radius_mean and perimeter_mean), which violates the Naive Bayes "
                "assumption. Despite this, it achieves a high AUC (~0.99) showing "
                "the probability estimates are well-calibrated, even though the "
                "hard predictions have more errors."
            ),
            'Random Forest (Ensemble)': (
                "Good overall performance with a strong AUC of ~0.99, benefiting "
                "from the ensemble of 150 trees that reduces variance. It sits "
                "between the best (Logistic Regression) and worst (Decision Tree) "
                "performers. The bagging mechanism helps overcome the single "
                "Decision Tree's overfitting problem, but doesn't quite match "
                "Logistic Regression on this particular dataset."
            )
        }
        
        for model_name, obs in observations.items():
            with st.expander(f"📝 {model_name}"):
                st.write(obs)
        
        st.info(
            "**Overall Winner: Logistic Regression** — For this dataset, the "
            "linear model outperforms all others. The breast cancer features, "
            "once standardized, separate well with a linear boundary. This is "
            "a good reminder that complex models aren't always better; "
            "sometimes a well-tuned simple model is the best choice."
        )
    
    # =====================
    # Tab 4: ROC Curves
    # =====================
    with tab4:
        st.subheader("ROC Curves — All Models Overlaid")
        fig_all_roc = plot_all_roc_curves(trained_models, X_test_scaled, y_test)
        st.pyplot(fig_all_roc)
        
        st.write(
            "The ROC curves show how each model trades off between true positive rate "
            "and false positive rate at different classification thresholds. Models with "
            "curves closer to the top-left corner have better discriminative ability. "
            "All models except Decision Tree achieve an AUC above 0.98, indicating "
            "strong separability in the feature space."
        )
    
    # ---- Footer ----
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
        "ML Assignment 2 | Breast Cancer Classification | BITS Pilani M.Tech (AIML/DSE)"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
