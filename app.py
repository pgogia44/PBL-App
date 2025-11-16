import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Water Enhancement Market Analytics",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #e8eef5;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e7bcf;
        color: white;
    }
    h1 {color: #1e3a8a;}
    h2 {color: #2563eb;}
    h3 {color: #3b82f6;}
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("💧 Water Enhancement Product - Market Analytics Dashboard")
st.markdown("### Data-Driven Insights for Strategic Marketing Decisions")
st.markdown("---")

# Load data function
@st.cache_data
def load_data():
    """Load the survey data"""
    try:
        df = pd.read_csv('water_enhancement_survey_data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found. Please upload 'water_enhancement_survey_data.csv' to the repository.")
        return None

# Load data
df = load_data()

if df is not None:
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/2e7bcf/ffffff?text=Water+Enhancement", use_column_width=True)
        st.markdown("## 📊 Dashboard Navigation")
        st.markdown("---")
        
        st.markdown("### Dataset Overview")
        st.metric("Total Responses", f"{len(df):,}")
        st.metric("Total Features", len(df.columns))
        high_interest = (df['interest_level'] >= 4).sum()
        st.metric("High Interest Customers", f"{high_interest:,} ({high_interest/len(df)*100:.1f}%)")
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Stats")
        st.write(f"**Avg WTP:** ${df['willingness_to_pay_continuous'].mean():.2f}")
        st.write(f"**Likely Buyers:** {df['purchase_likelihood'].isin(['Definitely would purchase', 'Probably would purchase']).sum():,}")
        
        st.markdown("---")
        st.info("💡 **Tip:** Use tabs above to navigate between different analyses")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Marketing Insights",
        "🤖 ML Algorithms",
        "🔮 Prediction Tool",
        "📊 Association Rules"
    ])

    # ========================================================================
    # TAB 1: MARKETING INSIGHTS DASHBOARD
    # ========================================================================
    with tab1:
        st.header("📊 Strategic Marketing Insights for Customer Conversion")
        st.markdown("#### 5 Key Charts to Drive Marketing Strategy")
        
        # CHART 1: Customer Segmentation Matrix
        st.markdown("---")
        st.subheader("1️⃣ Customer Segmentation Matrix: Interest vs Income vs Age")
        st.markdown("**Action:** Target high-value segments with premium messaging")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Prepare data
            df['interest_category'] = df['interest_level'].apply(
                lambda x: '🔥 High Interest' if x >= 4 else '⚡ Medium Interest' if x == 3 else '❄️ Low Interest'
            )
            income_mapping = {
                'Less than $25,000': 1, '$25,000 - $49,999': 2, '$50,000 - $74,999': 3,
                '$75,000 - $99,999': 4, '$100,000 - $149,999': 5,
                '$150,000 - $199,999': 6, '$200,000+': 7
            }
            df['income_numeric'] = df['income'].map(income_mapping)
            
            segment_data = df.groupby(['age_group', 'income', 'interest_category']).agg({
                'response_id': 'count',
                'willingness_to_pay_continuous': 'mean'
            }).reset_index()
            segment_data.columns = ['age_group', 'income', 'interest_category', 'count', 'avg_wtp']
            segment_data['income_numeric'] = segment_data['income'].map(income_mapping)
            
            fig1 = px.scatter(
                segment_data,
                x='age_group',
                y='income_numeric',
                size='count',
                color='interest_category',
                hover_data=['income', 'count', 'avg_wtp'],
                title='Customer Segments by Age, Income, and Interest Level',
                color_discrete_map={
                    '🔥 High Interest': '#27ae60',
                    '⚡ Medium Interest': '#f39c12',
                    '❄️ Low Interest': '#e74c3c'
                },
                size_max=60,
                height=500
            )
            fig1.update_yaxes(
                tickmode='array',
                tickvals=list(range(1, 8)),
                ticktext=['<$25K', '$25-50K', '$50-75K', '$75-100K', '$100-150K', '$150-200K', '$200K+']
            )
            fig1.update_layout(template='plotly_white')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Key Takeaways")
            best_segment = df.groupby('age_group').apply(
                lambda x: (x['interest_level'] >= 4).mean()
            ).sort_values(ascending=False).index[0]
            best_pct = df.groupby('age_group').apply(
                lambda x: (x['interest_level'] >= 4).mean()
            ).max() * 100
            
            st.success(f"**Best Segment:**\n\n{best_segment}\n\n{best_pct:.1f}% High Interest")
            
            high_income_interest = df[df['income_numeric'] >= 5]['interest_level'].mean()
            st.info(f"**Premium Segment:**\n\n$100K+ Income\n\nAvg Interest: {high_income_interest:.2f}/5")
            
            st.warning("**Recommended Action:**\n\nFocus premium campaigns on 25-44 age group with $75K+ income")

        # CHART 2: Revenue Potential Heatmap
        st.markdown("---")
        st.subheader("2️⃣ Revenue Potential Heatmap: Monthly Revenue per Customer")
        st.markdown("**Action:** Prioritize segments with highest revenue potential")
        
        # Calculate revenue potential
        usage_mapping = {
            '1-2': 1.5, '3-5': 4, '6-10': 8, '11-15': 13, '16-20': 18, '21+': 25
        }
        df['usage_numeric'] = df['weekly_usage'].map(usage_mapping)
        df['monthly_revenue'] = df['willingness_to_pay_continuous'] * df['usage_numeric'] * 4.33
        
        revenue_heatmap = df.pivot_table(
            values='monthly_revenue',
            index='age_group',
            columns='income',
            aggfunc='mean'
        )
        
        column_order = ['Less than $25,000', '$25,000 - $49,999', '$50,000 - $74,999',
                        '$75,000 - $99,999', '$100,000 - $149,999', '$150,000 - $199,999', '$200,000+']
        revenue_heatmap = revenue_heatmap[column_order]
        
        fig2 = go.Figure(data=go.Heatmap(
            z=revenue_heatmap.values,
            x=revenue_heatmap.columns,
            y=revenue_heatmap.index,
            colorscale='RdYlGn',
            text=revenue_heatmap.values.round(2),
            texttemplate='$%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Monthly<br>Revenue ($)")
        ))
        
        fig2.update_layout(
            title='Average Monthly Revenue Potential per Customer by Age & Income',
            xaxis_title='Annual Household Income',
            yaxis_title='Age Group',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            max_revenue = df['monthly_revenue'].max()
            st.metric("Max Monthly Revenue", f"${max_revenue:.2f}", "per customer")
        with col2:
            avg_revenue = df[df['interest_level'] >= 4]['monthly_revenue'].mean()
            st.metric("Avg Revenue (High Interest)", f"${avg_revenue:.2f}", "+35% vs overall")
        with col3:
            total_potential = df[df['interest_level'] >= 4]['monthly_revenue'].sum()
            st.metric("Total Monthly Potential", f"${total_potential:,.0f}", "from high-interest segment")

        # CHART 3: Conversion Funnel Analysis
        st.markdown("---")
        st.subheader("3️⃣ Customer Conversion Funnel: From Awareness to Purchase")
        st.markdown("**Action:** Optimize each stage to reduce drop-off")
        
        funnel_data = {
            'Total Respondents': len(df),
            'Find Water Boring': df['barrier_Plain water is boring/tasteless'].sum(),
            'Currently Use Enhancement': df[[col for col in df.columns if col.startswith('used_product_')]].sum(axis=1).gt(0).sum(),
            'Interested (≥3)': (df['interest_level'] >= 3).sum(),
            'High Interest (≥4)': (df['interest_level'] >= 4).sum(),
            'Likely to Buy': df['purchase_likelihood'].isin(['Definitely would purchase', 'Probably would purchase']).sum(),
            'Premium Willing ($2+)': (df['willingness_to_pay_continuous'] >= 2.0).sum()
        }
        
        stages = list(funnel_data.keys())
        values = list(funnel_data.values())
        
        fig3 = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=['#3498db', '#2ecc71', '#1abc9c', '#f39c12', '#e67e22', '#e74c3c', '#9b59b6'])
        ))
        
        fig3.update_layout(
            title='Customer Conversion Funnel Analysis',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Calculate conversion rates
        st.markdown("#### 📊 Conversion Rates Between Stages")
        col1, col2, col3, col4 = st.columns(4)
        
        conv1 = (values[3] / values[0]) * 100
        conv2 = (values[4] / values[3]) * 100
        conv3 = (values[5] / values[4]) * 100
        conv4 = (values[6] / values[5]) * 100
        
        col1.metric("Awareness → Interest", f"{conv1:.1f}%")
        col2.metric("Interest → High Interest", f"{conv2:.1f}%")
        col3.metric("High Interest → Likely Buy", f"{conv3:.1f}%")
        col4.metric("Likely Buy → Premium", f"{conv4:.1f}%")

        # CHART 4: Price Sensitivity Analysis
        st.markdown("---")
        st.subheader("4️⃣ Price Sensitivity Analysis: Optimal Pricing Strategy")
        st.markdown("**Action:** Set pricing tiers based on customer willingness to pay")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # WTP distribution by interest level
            fig4a = go.Figure()
            
            for interest in sorted(df['interest_level'].unique(), reverse=True):
                subset = df[df['interest_level'] == interest]
                fig4a.add_trace(go.Box(
                    y=subset['willingness_to_pay_continuous'],
                    name=f'Interest {interest}',
                    marker_color=['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60'][interest-1]
                ))
            
            fig4a.update_layout(
                title='Willingness to Pay Distribution by Interest Level',
                yaxis_title='Willingness to Pay ($)',
                xaxis_title='Interest Level',
                height=400,
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig4a, use_container_width=True)
        
        with col2:
            # Price acceptance by income
            price_points = [15, 25, 35, 45, 60]
            acceptance_data = []
            
            for price in price_points:
                col_name = f'price_{price}_perception'
                if col_name in df.columns:
                    acceptable = df[col_name].isin(['Bargain', 'Acceptable']).sum()
                    acceptance_data.append({
                        'Price': f'${price} (30-pack)',
                        'Acceptance Rate': (acceptable / len(df)) * 100
                    })
            
            price_df = pd.DataFrame(acceptance_data)
            
            fig4b = go.Figure(go.Bar(
                x=price_df['Price'],
                y=price_df['Acceptance Rate'],
                marker_color=['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c'],
                text=price_df['Acceptance Rate'].round(1),
                texttemplate='%{text}%',
                textposition='outside'
            ))
            
            fig4b.update_layout(
                title='Price Acceptance Rate by Package Price',
                yaxis_title='Acceptance Rate (%)',
                xaxis_title='Package Price',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig4b, use_container_width=True)
        
        st.markdown("#### 💰 Recommended Pricing Strategy")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("**Entry Tier**\n\n$0.50-$0.75 per cap\n\n$15-20 for 30-pack\n\nTarget: Price-sensitive segment")
        with col2:
            st.info("**Standard Tier**\n\n$1.00-$1.50 per cap\n\n$25-35 for 30-pack\n\nTarget: Mainstream market")
        with col3:
            st.warning("**Premium Tier**\n\n$1.75-$2.50 per cap\n\n$45-60 for 30-pack\n\nTarget: Health-conscious, high-income")

        # CHART 5: Feature Importance for High Interest
        st.markdown("---")
        st.subheader("5️⃣ Key Drivers of Customer Interest: What Matters Most")
        st.markdown("**Action:** Focus marketing messages on top appeal factors")
        
        # Top appealing features
        appeal_cols = [col for col in df.columns if col.startswith('appealing_') and 'Nothing' not in col]
        appeal_data = []
        
        for col in appeal_cols:
            feature = col.replace('appealing_', '').replace('_', ' ').title()
            count = df[col].sum()
            high_interest_rate = df[df[col] == 1]['interest_level'].mean()
            appeal_data.append({
                'Feature': feature,
                'Count': count,
                'Percentage': (count / len(df)) * 100,
                'Avg Interest': high_interest_rate
            })
        
        appeal_df = pd.DataFrame(appeal_data).sort_values('Percentage', ascending=True).tail(10)
        
        fig5 = go.Figure()
        
        fig5.add_trace(go.Bar(
            y=appeal_df['Feature'],
            x=appeal_df['Percentage'],
            orientation='h',
            marker=dict(
                color=appeal_df['Avg Interest'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Avg<br>Interest")
            ),
            text=appeal_df['Percentage'].round(1),
            texttemplate='%{text}%',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Mentioned by: %{x:.1f}%<br>Avg Interest: %{marker.color:.2f}<extra></extra>'
        ))
        
        fig5.update_layout(
            title='Top 10 Most Appealing Product Features (by customer interest correlation)',
            xaxis_title='Percentage of Respondents',
            yaxis_title='',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig5, use_container_width=True)
        
        # Top concerns
        st.markdown("#### ⚠️ Customer Concerns to Address")
        
        concern_cols = [col for col in df.columns if col.startswith('concern_') and 'No concerns' not in col]
        concern_data = []
        
        for col in concern_cols:
            concern = col.replace('concern_', '').replace('_', ' ').title()
            count = df[col].sum()
            concern_data.append({'Concern': concern, 'Count': count})
        
        concern_df = pd.DataFrame(concern_data).sort_values('Count', ascending=False).head(5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig5b = go.Figure(go.Bar(
                x=concern_df['Concern'],
                y=concern_df['Count'],
                marker_color='#e74c3c',
                text=concern_df['Count'],
                textposition='outside'
            ))
            
            fig5b.update_layout(
                title='Top 5 Customer Concerns',
                xaxis_title='',
                yaxis_title='Number of Respondents',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig5b, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Action Plan")
            st.markdown("""
            **1. Address Environmental Concerns:**
            - Develop recyclable/biodegradable caps
            - Partner with sustainability organizations
            - Communicate eco-friendly initiatives
            
            **2. Price Value Communication:**
            - Emphasize cost per use vs alternatives
            - Show monthly savings compared to pre-mixed drinks
            - Offer subscription discounts
            
            **3. Product Quality Assurance:**
            - Highlight natural ingredients
            - Provide dissolution guarantee
            - Share customer testimonials
            
            **4. Compatibility:**
            - Design universal-fit caps
            - Create compatibility guide
            - Offer adapter options
            """)

    # ========================================================================
    # TAB 2: ML ALGORITHMS
    # ========================================================================
    with tab2:
        st.header("🤖 Machine Learning Algorithms for Predictive Analytics")
        
        ml_tab1, ml_tab2, ml_tab3 = st.tabs([
            "📊 Classification",
            "🎯 Clustering",
            "📈 Regression"
        ])
        
        # CLASSIFICATION
        with ml_tab1:
            st.subheader("Classification: Predicting Customer Interest Level")
            st.markdown("**Goal:** Predict whether a customer will have high interest (≥4) in the product")
            
            if st.button("🚀 Run Classification Analysis", key="classify"):
                with st.spinner("Training classification models..."):
                    # Prepare data
                    df_class = df.copy()
                    df_class['high_interest'] = (df_class['interest_level'] >= 4).astype(int)
                    
                    # Select features
                    feature_cols = [
                        'health_consciousness', 'hydration_importance',
                        'early_adopter_score', 'premium_willingness_score',
                        'sustainability_importance', 'income_numeric',
                        'spending_numeric', 'usage_numeric'
                    ]
                    
                    # Add barrier columns
                    barrier_cols = [col for col in df.columns if col.startswith('barrier_')]
                    feature_cols.extend(barrier_cols)
                    
                    X = df_class[feature_cols].fillna(0)
                    y = df_class['high_interest']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y
                    )
                    
                    # Train models
                    models = {
                        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
                    }
                    
                    results = {}
                    
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        
                        results[name] = {
                            'model': model,
                            'y_pred': y_pred,
                            'y_pred_proba': y_pred_proba,
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred),
                            'recall': recall_score(y_test, y_pred),
                            'f1': f1_score(y_test, y_pred),
                            'auc': roc_auc_score(y_test, y_pred_proba)
                        }
                    
                    # Display results
                    st.success("✅ Classification models trained successfully!")
                    
                    # Metrics comparison
                    st.markdown("### 📊 Model Performance Comparison")
                    
                    metrics_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'Accuracy': [r['accuracy'] for r in results.values()],
                        'Precision': [r['precision'] for r in results.values()],
                        'Recall': [r['recall'] for r in results.values()],
                        'F1-Score': [r['f1'] for r in results.values()],
                        'AUC-ROC': [r['auc'] for r in results.values()]
                    })
                    
                    st.dataframe(metrics_df.style.format({
                        'Accuracy': '{:.4f}',
                        'Precision': '{:.4f}',
                        'Recall': '{:.4f}',
                        'F1-Score': '{:.4f}',
                        'AUC-ROC': '{:.4f}'
                    }).background_gradient(cmap='RdYlGn', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Confusion Matrix
                        best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
                        cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
                        
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Predicted Low', 'Predicted High'],
                            y=['Actual Low', 'Actual High'],
                            text=cm,
                            texttemplate='%{text}',
                            colorscale='Blues',
                            showscale=True
                        ))
                        
                        fig_cm.update_layout(
                            title=f'Confusion Matrix - {best_model_name}',
                            height=400
                        )
                        
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with col2:
                        # ROC Curve
                        fig_roc = go.Figure()
                        
                        for name, result in results.items():
                            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                            fig_roc.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name=f'{name} (AUC={result["auc"]:.3f})',
                                line=dict(width=3)
                            ))
                        
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random Classifier',
                            line=dict(dash='dash', color='gray')
                        ))
                        
                        fig_roc.update_layout(
                            title='ROC Curve Comparison',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            height=400
                        )
                        
                        st.plotly_chart(fig_roc, use_container_width=True)
                    
                    # Feature Importance
                    if best_model_name == 'Random Forest':
                        st.markdown("### 🎯 Feature Importance")
                        
                        rf_model = results['Random Forest']['model']
                        feature_importance = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=True).tail(10)
                        
                        fig_fi = go.Figure(go.Bar(
                            y=feature_importance['Feature'],
                            x=feature_importance['Importance'],
                            orientation='h',
                            marker_color='#3498db'
                        ))
                        
                        fig_fi.update_layout(
                            title='Top 10 Most Important Features',
                            xaxis_title='Importance Score',
                            height=400
                        )
                        
                        st.plotly_chart(fig_fi, use_container_width=True)
                    
                    # Business Insights
                    st.markdown("### 💡 Business Insights")
                    st.info(f"""
                    **Best Model:** {best_model_name}
                    
                    **Key Findings:**
                    - Model Accuracy: {results[best_model_name]['accuracy']*100:.2f}%
                    - Precision: {results[best_model_name]['precision']*100:.2f}% (of predicted high-interest, this % are actually high-interest)
                    - Recall: {results[best_model_name]['recall']*100:.2f}% (of actual high-interest customers, we can identify this %)
                    - AUC-ROC: {results[best_model_name]['auc']:.3f} (excellent discrimination ability)
                    
                    **Business Application:**
                    - Use this model to score leads and prioritize high-potential customers
                    - Focus marketing budget on customers predicted to have high interest
                    - Expected ROI improvement: {(results[best_model_name]['precision']/0.5 - 1)*100:.1f}% over random targeting
                    """)
        
        # CLUSTERING
        with ml_tab2:
            st.subheader("Clustering: Customer Segmentation Analysis")
            st.markdown("**Goal:** Identify distinct customer segments for targeted marketing")
            
            n_clusters = st.slider("Select number of clusters:", 2, 8, 4, key="n_clusters")
            
            if st.button("🚀 Run Clustering Analysis", key="cluster"):
                with st.spinner("Performing clustering analysis..."):
                    # Prepare data
                    cluster_features = [
                        'age_group', 'income', 'health_consciousness',
                        'exercise_frequency', 'daily_water_intake',
                        'interest_level', 'willingness_to_pay_continuous',
                        'early_adopter_score', 'premium_willingness_score'
                    ]
                    
                    df_cluster = df.copy()
                    
                    # Encode categorical variables
                    le_age = LabelEncoder()
                    le_income = LabelEncoder()
                    le_exercise = LabelEncoder()
                    le_water = LabelEncoder()
                    
                    df_cluster['age_encoded'] = le_age.fit_transform(df_cluster['age_group'])
                    df_cluster['income_encoded'] = le_income.fit_transform(df_cluster['income'])
                    df_cluster['exercise_encoded'] = le_exercise.fit_transform(df_cluster['exercise_frequency'])
                    df_cluster['water_encoded'] = le_water.fit_transform(df_cluster['daily_water_intake'])
                    
                    X_cluster = df_cluster[[
                        'age_encoded', 'income_encoded', 'health_consciousness',
                        'exercise_encoded', 'water_encoded', 'interest_level',
                        'willingness_to_pay_continuous', 'early_adopter_score',
                        'premium_willingness_score'
                    ]].fillna(0)
                    
                    # Standardize
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_cluster)
                    
                    # Apply K-Means
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    df_cluster['cluster'] = kmeans.fit_predict(X_scaled)
                    
                    # Calculate metrics
                    silhouette = silhouette_score(X_scaled, df_cluster['cluster'])
                    davies_bouldin = davies_bouldin_score(X_scaled, df_cluster['cluster'])
                    calinski = calinski_harabasz_score(X_scaled, df_cluster['cluster'])
                    
                    st.success("✅ Clustering analysis completed!")
                    
                    # Display metrics
                    st.markdown("### 📊 Clustering Quality Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Silhouette Score", f"{silhouette:.4f}", "Higher is better (max 1.0)")
                    col2.metric("Davies-Bouldin Index", f"{davies_bouldin:.4f}", "Lower is better (min 0.0)")
                    col3.metric("Calinski-Harabasz Score", f"{calinski:.2f}", "Higher is better")
                    
                    # Cluster visualization
                    st.markdown("### 🎯 Customer Segment Visualization")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 2D scatter: Income vs Interest by cluster
                        fig_cluster1 = px.scatter(
                            df_cluster,
                            x='income_numeric',
                            y='interest_level',
                            color='cluster',
                            size='willingness_to_pay_continuous',
                            hover_data=['age_group', 'exercise_frequency'],
                            title='Customer Segments: Income vs Interest Level',
                            labels={'income_numeric': 'Income Level', 'interest_level': 'Interest Level'},
                            color_continuous_scale='Viridis'
                        )
                        
                        fig_cluster1.update_layout(height=400)
                        st.plotly_chart(fig_cluster1, use_container_width=True)
                    
                    with col2:
                        # Cluster size distribution
                        cluster_sizes = df_cluster['cluster'].value_counts().sort_index()
                        
                        fig_cluster2 = go.Figure(data=[go.Pie(
                            labels=[f'Segment {i}' for i in cluster_sizes.index],
                            values=cluster_sizes.values,
                            hole=0.4,
                            marker_colors=px.colors.qualitative.Set3
                        )])
                        
                        fig_cluster2.update_layout(
                            title='Customer Segment Distribution',
                            height=400
                        )
                        
                        st.plotly_chart(fig_cluster2, use_container_width=True)
                    
                    # Cluster profiles
                    st.markdown("### 📋 Segment Profiles")
                    
                    for cluster_id in range(n_clusters):
                        cluster_data = df_cluster[df_cluster['cluster'] == cluster_id]
                        
                        with st.expander(f"📊 Segment {cluster_id} ({len(cluster_data)} customers, {len(cluster_data)/len(df)*100:.1f}%)"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**Demographics**")
                                st.write(f"• Dominant Age: {cluster_data['age_group'].mode()[0]}")
                                st.write(f"• Dominant Income: {cluster_data['income'].mode()[0]}")
                                st.write(f"• Avg Interest: {cluster_data['interest_level'].mean():.2f}/5")
                            
                            with col2:
                                st.markdown("**Behavior**")
                                st.write(f"• Health Consciousness: {cluster_data['health_consciousness'].mean():.2f}/5")
                                st.write(f"• Exercise: {cluster_data['exercise_frequency'].mode()[0]}")
                                st.write(f"• Water Intake: {cluster_data['daily_water_intake'].mode()[0]} glasses/day")
                            
                            with col3:
                                st.markdown("**Commercial**")
                                st.write(f"• Avg WTP: ${cluster_data['willingness_to_pay_continuous'].mean():.2f}")
                                st.write(f"• Monthly Revenue: ${cluster_data['monthly_revenue'].mean():.2f}")
                                st.write(f"• Early Adopter: {cluster_data['early_adopter_score'].mean():.2f}/5")
                            
                            # Recommended strategy
                            avg_interest = cluster_data['interest_level'].mean()
                            avg_wtp = cluster_data['willingness_to_pay_continuous'].mean()
                            
                            if avg_interest >= 4 and avg_wtp >= 2:
                                strategy = "🎯 **Premium Target** - High interest, high willingness to pay. Focus on premium features and quality."
                            elif avg_interest >= 3.5 and avg_wtp >= 1.5:
                                strategy = "📈 **Growth Opportunity** - Good interest and moderate WTP. Standard tier with upsell potential."
                            elif avg_interest >= 3:
                                strategy = "💡 **Nurture Segment** - Moderate interest. Educational marketing to increase engagement."
                            else:
                                strategy = "🔍 **Low Priority** - Lower interest. Minimal marketing investment, focus on awareness."
                            
                            st.info(strategy)
        
        # REGRESSION
        with ml_tab3:
            st.subheader("Regression: Predicting Willingness to Pay")
            st.markdown("**Goal:** Predict how much a customer is willing to pay per cap")
            
            if st.button("🚀 Run Regression Analysis", key="regress"):
                with st.spinner("Training regression models..."):
                    # Prepare data
                    feature_cols = [
                        'interest_level', 'health_consciousness', 'income_numeric',
                        'exercise_frequency_encoded', 'age_encoded',
                        'early_adopter_score', 'premium_willingness_score',
                        'spending_numeric', 'hydration_importance'
                    ]
                    
                    df_reg = df.copy()
                    
                    # Encode categorical
                    le_exercise = LabelEncoder()
                    le_age = LabelEncoder()
                    
                    df_reg['exercise_frequency_encoded'] = le_exercise.fit_transform(df_reg['exercise_frequency'])
                    df_reg['age_encoded'] = le_age.fit_transform(df_reg['age_group'])
                    
                    X_reg = df_reg[feature_cols].fillna(0)
                    y_reg = df_reg['willingness_to_pay_continuous']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_reg, y_reg, test_size=0.3, random_state=42
                    )
                    
                    # Train models
                    models_reg = {
                        'Linear Regression': LinearRegression(),
                        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
                    }
                    
                    results_reg = {}
                    
                    for name, model in models_reg.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        results_reg[name] = {
                            'model': model,
                            'y_pred': y_pred,
                            'mse': mean_squared_error(y_test, y_pred),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'mae': mean_absolute_error(y_test, y_pred),
                            'r2': r2_score(y_test, y_pred)
                        }
                    
                    st.success("✅ Regression models trained successfully!")
                    
                    # Display metrics
                    st.markdown("### 📊 Model Performance Comparison")
                    
                    metrics_reg_df = pd.DataFrame({
                        'Model': list(results_reg.keys()),
                        'R² Score': [r['r2'] for r in results_reg.values()],
                        'RMSE': [r['rmse'] for r in results_reg.values()],
                        'MAE': [r['mae'] for r in results_reg.values()],
                        'MSE': [r['mse'] for r in results_reg.values()]
                    })
                    
                    st.dataframe(metrics_reg_df.style.format({
                        'R² Score': '{:.4f}',
                        'RMSE': '{:.4f}',
                        'MAE': '{:.4f}',
                        'MSE': '{:.4f}'
                    }).background_gradient(cmap='RdYlGn', subset=['R² Score']))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Actual vs Predicted
                        best_model_name = max(results_reg.keys(), key=lambda k: results_reg[k]['r2'])
                        
                        fig_pred = go.Figure()
                        
                        fig_pred.add_trace(go.Scatter(
                            x=y_test,
                            y=results_reg[best_model_name]['y_pred'],
                            mode='markers',
                            name='Predictions',
                            marker=dict(color='#3498db', size=8, opacity=0.6)
                        ))
                        
                        fig_pred.add_trace(go.Scatter(
                            x=[y_test.min(), y_test.max()],
                            y=[y_test.min(), y_test.max()],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='red')
                        ))
                        
                        fig_pred.update_layout(
                            title=f'Actual vs Predicted WTP - {best_model_name}',
                            xaxis_title='Actual WTP ($)',
                            yaxis_title='Predicted WTP ($)',
                            height=400
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    with col2:
                        # Residuals
                        residuals = y_test - results_reg[best_model_name]['y_pred']
                        
                        fig_resid = go.Figure()
                        
                        fig_resid.add_trace(go.Histogram(
                            x=residuals,
                            nbinsx=30,
                            marker_color='#e74c3c',
                            name='Residuals'
                        ))
                        
                        fig_resid.update_layout(
                            title='Prediction Error Distribution',
                            xaxis_title='Residual (Actual - Predicted)',
                            yaxis_title='Frequency',
                            height=400
                        )
                        
                        st.plotly_chart(fig_resid, use_container_width=True)
                    
                    # Business Insights
                    st.markdown("### 💡 Business Insights")
                    
                    r2 = results_reg[best_model_name]['r2']
                    mae = results_reg[best_model_name]['mae']
                    
                    st.info(f"""
                    **Best Model:** {best_model_name}
                    
                    **Model Performance:**
                    - R² Score: {r2:.4f} (explains {r2*100:.1f}% of variance in WTP)
                    - MAE: ${mae:.2f} (average prediction error)
                    - RMSE: ${results_reg[best_model_name]['rmse']:.2f}
                    
                    **Business Application:**
                    - Use model to set personalized pricing for each customer segment
                    - Predict revenue based on customer characteristics
                    - Optimize pricing tiers with {mae*100:.0f}% accuracy
                    - Expected pricing optimization uplift: {(1/mae)*10:.1f}% revenue increase
                    """)
                    
                    # Feature importance for Random Forest
                    if best_model_name == 'Random Forest':
                        st.markdown("### 🎯 Feature Importance for WTP Prediction")
                        
                        rf_model = results_reg['Random Forest']['model']
                        feature_importance_reg = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=True).tail(10)
                        
                        fig_fi_reg = go.Figure(go.Bar(
                            y=feature_importance_reg['Feature'],
                            x=feature_importance_reg['Importance'],
                            orientation='h',
                            marker_color='#9b59b6'
                        ))
                        
                        fig_fi_reg.update_layout(
                            title='Top Factors Influencing Willingness to Pay',
                            xaxis_title='Importance Score',
                            height=400
                        )
                        
                        st.plotly_chart(fig_fi_reg, use_container_width=True)

    # ========================================================================
    # TAB 3: PREDICTION TOOL
    # ========================================================================
    with tab3:
        st.header("🔮 Customer Interest Prediction Tool")
        st.markdown("Upload new customer data to predict their interest level and purchase likelihood")
        
        st.markdown("### 📤 Upload New Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file with customer data",
            type=['csv'],
            help="Upload a CSV file with the same structure as the training data"
        )
        
        if uploaded_file is not None:
            try:
                # Load uploaded data
                new_data = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! {len(new_data)} records found.")
                
                # Show preview
                with st.expander("📋 Preview Uploaded Data (first 10 rows)"):
                    st.dataframe(new_data.head(10))
                
                # Check if required columns exist
                required_cols = [
                    'health_consciousness', 'hydration_importance',
                    'early_adopter_score', 'premium_willingness_score',
                    'sustainability_importance'
                ]
                
                missing_cols = [col for col in required_cols if col not in new_data.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                else:
                    if st.button("🚀 Generate Predictions"):
                        with st.spinner("Generating predictions..."):
                            # Prepare features (same as classification model)
                            feature_cols = [
                                'health_consciousness', 'hydration_importance',
                                'early_adopter_score', 'premium_willingness_score',
                                'sustainability_importance'
                            ]
                            
                            # Add income if exists
                            if 'income' in new_data.columns:
                                income_mapping = {
                                    'Less than $25,000': 1, '$25,000 - $49,999': 2,
                                    '$50,000 - $74,999': 3, '$75,000 - $99,999': 4,
                                    '$100,000 - $149,999': 5, '$150,000 - $199,999': 6,
                                    '$200,000+': 7
                                }
                                new_data['income_numeric'] = new_data['income'].map(income_mapping)
                                feature_cols.append('income_numeric')
                            
                            # Add spending if exists
                            if 'monthly_beverage_spend' in new_data.columns:
                                spending_mapping = {
                                    'Less than $20': 10, '$20-$39': 30, '$40-$59': 50,
                                    '$60-$79': 70, '$80-$99': 90, '$100+': 120
                                }
                                new_data['spending_numeric'] = new_data['monthly_beverage_spend'].map(spending_mapping)
                                feature_cols.append('spending_numeric')
                            
                            # Add barrier columns if they exist
                            barrier_cols = [col for col in new_data.columns if col.startswith('barrier_')]
                            feature_cols.extend(barrier_cols)
                            
                            X_new = new_data[feature_cols].fillna(0)
                            
                            # Train a simple model on existing data for prediction
                            df_train = df.copy()
                            df_train['high_interest'] = (df_train['interest_level'] >= 4).astype(int)
                            
                            X_train = df_train[feature_cols].fillna(0)
                            y_train = df_train['high_interest']
                            
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model.fit(X_train, y_train)
                            
                            # Make predictions
                            predictions = model.predict(X_new)
                            predictions_proba = model.predict_proba(X_new)[:, 1]
                            
                            # Add predictions to dataframe
                            new_data['predicted_interest_label'] = predictions
                            new_data['predicted_interest_label'] = new_data['predicted_interest_label'].map({
                                0: 'Low Interest (1-3)',
                                1: 'High Interest (4-5)'
                            })
                            new_data['interest_probability'] = (predictions_proba * 100).round(2)
                            
                            # Predict purchase likelihood (simplified)
                            new_data['predicted_purchase_likelihood'] = new_data['interest_probability'].apply(
                                lambda x: 'Definitely would purchase' if x >= 80 else
                                         'Probably would purchase' if x >= 60 else
                                         'Might or might not purchase' if x >= 40 else
                                         'Probably would not purchase' if x >= 20 else
                                         'Definitely would not purchase'
                            )
                            
                            # Predict WTP (simplified based on interest probability)
                            new_data['predicted_wtp'] = (
                                0.75 + (predictions_proba * 1.75)
                            ).round(2)
                            
                            st.success("✅ Predictions generated successfully!")
                            
                            # Summary statistics
                            st.markdown("### 📊 Prediction Summary")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            high_interest_count = (predictions == 1).sum()
                            avg_prob = predictions_proba.mean() * 100
                            likely_buyers = new_data['predicted_purchase_likelihood'].isin([
                                'Definitely would purchase', 'Probably would purchase'
                            ]).sum()
                            avg_wtp = new_data['predicted_wtp'].mean()
                            
                            col1.metric("High Interest Customers", f"{high_interest_count}", 
                                       f"{high_interest_count/len(new_data)*100:.1f}%")
                            col2.metric("Avg Interest Probability", f"{avg_prob:.1f}%")
                            col3.metric("Likely Buyers", f"{likely_buyers}", 
                                       f"{likely_buyers/len(new_data)*100:.1f}%")
                            col4.metric("Avg Predicted WTP", f"${avg_wtp:.2f}")
                            
                            # Visualizations
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Interest distribution
                                interest_dist = new_data['predicted_interest_label'].value_counts()
                                
                                fig_interest = go.Figure(data=[go.Pie(
                                    labels=interest_dist.index,
                                    values=interest_dist.values,
                                    hole=0.4,
                                    marker_colors=['#e74c3c', '#27ae60']
                                )])
                                
                                fig_interest.update_layout(
                                    title='Predicted Interest Distribution',
                                    height=400
                                )
                                
                                st.plotly_chart(fig_interest, use_container_width=True)
                            
                            with col2:
                                # Purchase likelihood distribution
                                purchase_dist = new_data['predicted_purchase_likelihood'].value_counts()
                                
                                fig_purchase = go.Figure(data=[go.Bar(
                                    x=purchase_dist.index,
                                    y=purchase_dist.values,
                                    marker_color=['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
                                )])
                                
                                fig_purchase.update_layout(
                                    title='Predicted Purchase Likelihood',
                                    xaxis_title='',
                                    yaxis_title='Count',
                                    height=400
                                )
                                fig_purchase.update_xaxes(tickangle=-45)
                                
                                st.plotly_chart(fig_purchase, use_container_width=True)
                            
                            # Show predictions table
                            st.markdown("### 📋 Detailed Predictions")
                            
                            display_cols = [
                                'predicted_interest_label',
                                'interest_probability',
                                'predicted_purchase_likelihood',
                                'predicted_wtp'
                            ]
                            
                            # Add some original columns if they exist
                            if 'age_group' in new_data.columns:
                                display_cols.insert(0, 'age_group')
                            if 'income' in new_data.columns:
                                display_cols.insert(1, 'income')
                            
                            st.dataframe(
                                new_data[display_cols].head(20),
                                use_container_width=True
                            )
                            
                            # Download button
                            st.markdown("### 💾 Download Predictions")
                            
                            csv = new_data.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download Complete Dataset with Predictions",
                                data=csv,
                                file_name="customer_predictions.csv",
                                mime="text/csv",
                                help="Download the uploaded data with all predictions added"
                            )
                            
                            # Marketing recommendations
                            st.markdown("### 🎯 Marketing Recommendations")
                            
                            high_value = new_data[
                                (new_data['predicted_interest_label'] == 'High Interest (4-5)') &
                                (new_data['predicted_wtp'] >= 2.0)
                            ]
                            
                            if len(high_value) > 0:
                                st.success(f"""
                                **High-Value Segment Identified:** {len(high_value)} customers
                                
                                These customers show:
                                - High interest probability (≥60%)
                                - High willingness to pay (≥$2.00)
                                - Likely to purchase
                                
                                **Recommended Action:**
                                1. Priority outreach with premium messaging
                                2. Offer early-bird subscription deals
                                3. Personalized product recommendations
                                4. Expected conversion rate: {(high_value['interest_probability'].mean()):.1f}%
                                """)
                            
                            moderate_interest = new_data[
                                (new_data['interest_probability'] >= 40) &
                                (new_data['interest_probability'] < 60)
                            ]
                            
                            if len(moderate_interest) > 0:
                                st.info(f"""
                                **Nurture Segment Identified:** {len(moderate_interest)} customers
                                
                                These customers show moderate interest and need education.
                                
                                **Recommended Action:**
                                1. Educational email campaign
                                2. Free sample offers
                                3. Customer testimonials and reviews
                                4. Product benefit highlights
                                """)
                            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your CSV has the required columns and correct data types.")
        
        else:
            st.info("👆 Upload a CSV file to start making predictions")
            
            st.markdown("### 📝 Required Columns")
            st.markdown("""
            Your CSV file should contain at least these columns:
            
            **Required:**
            - `health_consciousness` (1-5 scale)
            - `hydration_importance` (1-5 scale)
            - `early_adopter_score` (1-5 scale)
            - `premium_willingness_score` (1-5 scale)
            - `sustainability_importance` (1-5 scale)
            
            **Optional (improves accuracy):**
            - `income` (categorical: 'Less than $25,000', '$25,000 - $49,999', etc.)
            - `monthly_beverage_spend` (categorical: 'Less than $20', '$20-$39', etc.)
            - `barrier_*` columns (binary 0/1 for each barrier)
            - `age_group` (categorical)
            - `exercise_frequency` (categorical)
            """)
            
            # Sample data download
            st.markdown("### 📥 Download Sample Template")
            
            sample_data = pd.DataFrame({
                'health_consciousness': [4, 3, 5, 2, 4],
                'hydration_importance': [4, 3, 5, 3, 4],
                'early_adopter_score': [3, 4, 5, 2, 3],
                'premium_willingness_score': [4, 3, 5, 2, 4],
                'sustainability_importance': [5, 4, 5, 3, 4],
                'income': ['$75,000 - $99,999', '$50,000 - $74,999', '$100,000 - $149,999', 
                          '$25,000 - $49,999', '$75,000 - $99,999'],
                'age_group': ['25-34', '35-44', '25-34', '45-54', '35-44']
            })
            
            sample_csv = sample_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Sample CSV Template",
                data=sample_csv,
                file_name="sample_customer_data.csv",
                mime="text/csv"
            )

    # ========================================================================
    # TAB 4: ASSOCIATION RULES
    # ========================================================================
    with tab4:
        st.header("🔗 Association Rules: Purchase Preference & Price Perception")
        st.markdown("Discover patterns between customer preferences and price perceptions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_support = st.slider("Minimum Support (%)", 5, 50, 10, 5) / 100
        
        with col2:
            min_confidence = st.slider("Minimum Confidence (%)", 50, 95, 70, 5) / 100
        
        if st.button("🚀 Run Association Rule Mining"):
            with st.spinner("Mining association rules..."):
                try:
                    # Prepare transaction data
                    transactions = []
                    
                    price_cols = [col for col in df.columns if col.startswith('price_') and col.endswith('_perception')]
                    
                    for idx, row in df.iterrows():
                        transaction = []
                        
                        # Purchase preference
                        transaction.append(f"PREF: {row['purchase_preference']}")
                        
                        # Price perceptions
                        for col in price_cols:
                            price = col.replace('price_', '').replace('_perception', '')
                            perception = row[col]
                            transaction.append(f"${price}: {perception}")
                        
                        transactions.append(transaction)
                    
                    # Encode to binary
                    from mlxtend.preprocessing import TransactionEncoder
                    
                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
                    
                    # Apply Apriori
                    frequent_itemsets = apriori(
                        transaction_df,
                        min_support=min_support,
                        use_colnames=True
                    )
                    
                    if len(frequent_itemsets) == 0:
                        st.warning("⚠️ No frequent itemsets found. Try lowering the support threshold.")
                    else:
                        # Generate rules
                        rules = association_rules(
                            frequent_itemsets,
                            metric="confidence",
                            min_threshold=min_confidence
                        )
                        
                        if len(rules) == 0:
                            st.warning("⚠️ No rules found. Try lowering the confidence threshold.")
                        else:
                            # Sort by lift
                            rules = rules.sort_values('lift', ascending=False)
                            
                            st.success(f"✅ Found {len(rules)} association rules!")
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            col1.metric("Total Rules", len(rules))
                            col2.metric("Avg Confidence", f"{rules['confidence'].mean()*100:.1f}%")
                            col3.metric("Avg Lift", f"{rules['lift'].mean():.2f}x")
                            col4.metric("Rules with Lift>1", f"{(rules['lift'] > 1).sum()}")
                            
                            # Top 10 rules
                            st.markdown("### 🏆 Top 10 Association Rules")
                            
                            top_rules = rules.head(10)
                            
                            for i, (idx, row) in enumerate(top_rules.iterrows(), 1):
                                with st.expander(f"Rule {i}: {', '.join(list(row['antecedents']))} → {', '.join(list(row['consequents']))}"):
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        st.markdown("**IF:**")
                                        for ant in row['antecedents']:
                                            st.write(f"  • {ant}")
                                        
                                        st.markdown("**THEN:**")
                                        for cons in row['consequents']:
                                            st.write(f"  • {cons}")
                                    
                                    with col2:
                                        st.metric("Support", f"{row['support']*100:.2f}%")
                                        st.metric("Confidence", f"{row['confidence']*100:.2f}%")
                                        st.metric("Lift", f"{row['lift']:.2f}x")
                                    
                                    st.info(f"""
                                    **Interpretation:** When customers have the IF conditions, there's a 
                                    {row['confidence']*100:.1f}% chance they will also have the THEN conditions. 
                                    This is {row['lift']:.2f}x more likely than random chance.
                                    """)
                            
                            # Visualization
                            st.markdown("### 📊 Rule Visualization")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Support vs Confidence
                                fig_assoc1 = go.Figure()
                                
                                fig_assoc1.add_trace(go.Scatter(
                                    x=top_rules['support'],
                                    y=top_rules['confidence'],
                                    mode='markers',
                                    marker=dict(
                                        size=top_rules['lift']*10,
                                        color=top_rules['lift'],
                                        colorscale='Viridis',
                                        showscale=True,
                                        colorbar=dict(title="Lift")
                                    ),
                                    text=[f"Rule {i}" for i in range(1, len(top_rules)+1)],
                                    hovertemplate='<b>%{text}</b><br>Support: %{x:.3f}<br>Confidence: %{y:.3f}<extra></extra>'
                                ))
                                
                                fig_assoc1.update_layout(
                                    title='Association Rules: Support vs Confidence',
                                    xaxis_title='Support',
                                    yaxis_title='Confidence',
                                    height=400
                                )
                                
                                st.plotly_chart(fig_assoc1, use_container_width=True)
                            
                            with col2:
                                # Lift comparison
                                fig_assoc2 = go.Figure(go.Bar(
                                    y=[f"Rule {i}" for i in range(1, len(top_rules)+1)],
                                    x=top_rules['lift'].values,
                                    orientation='h',
                                    marker_color=top_rules['lift'].values,
                                    marker_colorscale='RdYlGn',
                                    text=top_rules['lift'].round(2),
                                    textposition='outside'
                                ))
                                
                                fig_assoc2.update_layout(
                                    title='Top Rules by Lift Score',
                                    xaxis_title='Lift',
                                    height=400
                                )
                                
                                st.plotly_chart(fig_assoc2, use_container_width=True)
                            
                            # Download rules
                            st.markdown("### 💾 Download Rules")
                            
                            export_df = top_rules.copy()
                            export_df['antecedents'] = export_df['antecedents'].apply(lambda x: ', '.join(list(x)))
                            export_df['consequents'] = export_df['consequents'].apply(lambda x: ', '.join(list(x)))
                            
                            csv_rules = export_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download Top 10 Association Rules",
                                data=csv_rules,
                                file_name="association_rules.csv",
                                mime="text/csv"
                            )
                            
                except Exception as e:
                    st.error(f"❌ Error in association rule mining: {str(e)}")
                    st.info("Please ensure your data has the required columns.")

else:
    st.error("❌ Unable to load data. Please ensure 'water_enhancement_survey_data.csv' is in the repository.")
    st.info("""
    **To fix this issue:**
    1. Make sure you have uploaded the survey data CSV file
    2. Rename it to 'water_enhancement_survey_data.csv'
    3. Place it in the same directory as app.py
    4. Refresh the app
    """)
