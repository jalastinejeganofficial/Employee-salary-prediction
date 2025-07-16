import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score
import time
import json
from streamlit_lottie import st_lottie
import os
from datetime import datetime

# --------------------------
# 🎨 APP CONFIG & ANIMATIONS
# --------------------------
st.set_page_config(
    page_title="Salary Oracle Pro", 
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Lottie animations
def load_lottie(filepath):
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None

# --------------------------
# 🧠 AI ASSISTANT COMPONENT
# --------------------------
class AIAssistant:
    def __init__(self):
        self.personality = {
            "name": "Salara",
            "mood": "enthusiastic",
            "role": "Compensation AI"
        }
        self.memory = []
        
    def respond(self, prompt):
        responses = {
            "hello": f"👋 Hi there! I'm {self.personality['name']}, your {self.personality['role']}. Let's predict some salaries!",
            "prediction": "📊 Based on my analysis, the salary progression seems...",
            "advice": "💡 Pro tip: Consider these factors for salary growth...",
            "default": "🤖 I'm analyzing your request... one moment please!"
        }
        return responses.get(prompt.lower(), responses["default"])

# --------------------------
# 💰 SALARY SIMULATOR
# --------------------------
class CareerSimulator:
    def __init__(self, model):
        self.model = model
        
    def growth_path(self, current_exp, target_salary):
        current_pred = self.model.predict([[current_exp]])[0][0]
        years_needed = (target_salary - current_pred) / (self.model.coef_[0][0] * 1.1)  # 10% premium
        return max(0, years_needed)
    
    def career_trajectory(self, current_exp, years_ahead=5):
        trajectory = []
        for year in range(years_ahead + 1):
            exp = current_exp + year
            salary = self.model.predict([[exp]])[0][0]
            bonus = salary * 0.1 * (1 + year/years_ahead)  # Simulate bonus growth
            trajectory.append({
                "Year": datetime.now().year + year,
                "Experience": exp,
                "Base Salary": salary,
                "Bonus": bonus,
                "Total Comp": salary + bonus
            })
        return pd.DataFrame(trajectory)

# --------------------------
# 🎮 GAMIFICATION ELEMENTS
# --------------------------
class SalaryGame:
    def __init__(self):
        self.level = 1
        self.points = 0
        
    def award_points(self, action):
        points_map = {
            "prediction": 10,
            "comparison": 15,
            "export": 20,
            "daily_login": 50
        }
        self.points += points_map.get(action, 5)
        if self.points > self.level * 100:
            self.level += 1
            return f"🎉 Level Up! Now at Level {self.level}"
        return None

# --------------------------
# 📊 MAIN APP
# --------------------------
def main():
    # Initialize components
    ai = AIAssistant()
    game = SalaryGame()
    
    # Load resources with cool animations
    with st.spinner('🚀 Launching Salary Oracle...'):
        time.sleep(1)
    
    # Header with animation
    col1, col2 = st.columns([3,1])
    with col1:
        st.title("🔮 Salary Oracle Pro")
        st.markdown("### AI-Powered Compensation Insights")
    with col2:
        lottie_ai = load_lottie("ai.json")  # Replace with your Lottie file
        if lottie_ai:
            st_lottie(lottie_ai, height=100)
    
    # Load model and data
    try:
        model = joblib.load('final_model.sav')
        data = pd.read_csv('Dataset.csv')
        simulator = CareerSimulator(model)
    except Exception as e:
        st.error(f"🚨 Error loading resources: {str(e)}")
        st.stop()
    
    # Sidebar with futuristic controls
    with st.sidebar:
        st.header("🔧 Control Panel")
        
        # 3D Experience Selector
        years_exp = st.slider(
            "🧑‍💻 Years of Experience", 
            0.0, 30.0, 5.0, 0.5,
            help="Drag through your career timeline"
        )
        
        # AI Mode selector
        ai_mode = st.selectbox(
            "🤖 AI Analysis Depth",
            ["Basic", "Advanced", "Executive"],
            index=1
        )
        
        # Visual theme
        theme = st.selectbox(
            "🎨 Visualization Theme",
            ["Corporate", "Dark", "Neon", "Solar"],
            index=0
        )
        
        # Gamification
        if st.button("🎮 Check My Points"):
            level_up = game.award_points("daily_login")
            if level_up:
                st.balloons()
                st.success(level_up)
            st.info(f"🏆 Level {game.level} | Points: {game.points}")
    
    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Predict", "🚀 Simulate", "🌍 Compare", "💼 Career Lab"])
    
    with tab1:
        # Interactive prediction with AI
        st.subheader("AI-Powered Salary Prediction")
        
        # Real-time prediction card
        prediction = model.predict([[years_exp]])[0][0]
        
        with st.container():
            cols = st.columns([1,2])
            with cols[0]:
                st.markdown(f"""
                <div style='border-radius:10px; padding:20px; background:linear-gradient(135deg, #6e8efb, #a777e3); color:white;'>
                    <h3 style='color:white;'>Your Prediction</h3>
                    <h1 style='font-size:2.5rem;'>${prediction:,.0f}</h1>
                    <p>for {years_exp} years experience</p>
                </div>
                """, unsafe_allow_html=True)
                
                # AI response
                with st.expander("💬 Ask Salara for insights"):
                    user_q = st.text_input("Ask about your prediction")
                    if user_q:
                        st.write(ai.respond(user_q))
            
            with cols[1]:
                # Interactive 3D plot
                fig = px.scatter_3d(
                    data, 
                    x='YearsExperience', 
                    y='Salary', 
                    z=np.random.rand(len(data)),  # Random z for 3D effect
                    color='Salary',
                    hover_name='Salary',
                    title="Salary Landscape"
                )
                fig.update_traces(
                    marker=dict(size=12, line=dict(width=2)),
                    selector=dict(mode='markers')
                )
                fig.add_trace(go.Scatter3d(
                    x=[years_exp],
                    y=[prediction],
                    z=[0.5],
                    mode='markers',
                    marker=dict(size=15, color='gold'),
                    name='You'
                ))
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Career simulation
        st.subheader("🚀 Career Trajectory Simulator")
        
        col1, col2 = st.columns(2)
        with col1:
            target_salary = st.number_input(
                "💸 Enter your target salary", 
                min_value=30000, 
                max_value=500000, 
                value=100000
            )
            if st.button("Calculate Path"):
                years_needed = simulator.growth_path(years_exp, target_salary)
                if years_needed <= 0:
                    st.success(f"🎯 You've already reached this salary!")
                else:
                    st.info(f"⏳ You'll need {years_needed:.1f} more years to reach ${target_salary:,.0f}")
                    game.award_points("prediction")
        
        with col2:
            trajectory = simulator.career_trajectory(years_exp)
            fig = px.area(
                trajectory, 
                x="Year", 
                y="Total Comp",
                title="Your 5-Year Compensation Forecast"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Global comparison
        st.subheader("🌍 Global Salary Comparison")
        
        # Mock country data
        countries = pd.DataFrame({
            "Country": ["USA", "UK", "Germany", "Japan", "India"],
            "Multiplier": [1.0, 0.8, 0.85, 0.9, 0.3]
        })
        
        # Apply multipliers
        countries["Salary"] = prediction * countries["Multiplier"]
        
        # Interactive map
        fig = px.choropleth(
            countries,
            locations="Country",
            locationmode="country names",
            color="Salary",
            hover_name="Country",
            color_continuous_scale=px.colors.sequential.Plasma,
            title="Equivalent Salaries Worldwide"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Interactive career lab
        st.subheader("💼 Career Experiment Lab")
        
        # Skill investment simulator
        st.markdown("### 🛠 Skill Investment Simulator")
        skills = st.multiselect(
            "Select skills to develop",
            ["AI/ML", "Leadership", "Cloud Computing", "Data Science", "Product Management"],
            default=["AI/ML"]
        )
        
        # Calculate boost
        if skills:
            boost = 1 + (len(skills) * 0.05)
            boosted_salary = prediction * boost
            st.success(f"📈 With these skills, you could earn ${boosted_salary:,.0f} (+{((boost-1)*100):.0f}%)")
            
            # Show skill impact
            fig = px.bar(
                x=["Current", "With Skills"],
                y=[prediction, boosted_salary],
                text=[f"${prediction:,.0f}", f"${boosted_salary:,.0f}"],
                title="Skill Investment Impact"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center'>
        <p>🔮 Salary Oracle Pro v2.0 | Powered by AI | ✨ Gamified Experience</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()