import os
import json
import re
from dotenv import load_dotenv
from typing import Optional, Type
import yfinance as yf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import gradio as gr
import plotly.express as px
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)

@tool
def get_market_data(tickers: str) -> str:
    """
    Fetches real-time market data for given tickers (e.g., 'VOO' for Vanguard S&P 500 ETF).
    Returns current price, 1-day change, and brief trend.
    """
    try:
        tickers_list = [t.strip() for t in tickers.split(',')]
        data = {}
        for ticker in tickers_list:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                change = hist['Close'].iloc[-1] - hist['Open'].iloc[0]
                trend = "upward" if change > 0 else "downward"
                data[ticker] = {
                    "current_price": round(current_price, 2),
                    "change": round(change, 2),
                    "trend": trend
                }
            else:
                data[ticker] = {"error": "No data available"}
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error fetching data: {str(e)}"

# Define tools
tools = [get_market_data]

# Prompt template for agent
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are WealthWise Agent, a personal finance planner. 
    Analyze the user's goal, income, and expenses. Use the 50/30/20 rule (50% needs, 30% wants, 20% savings/investments) as a base.
    Allow users to customize the percentages if provided.
    Support multiple goals with timelines, and calculate how much to save monthly for each goal.
    Fetch real-time market data for investment suggestions (e.g., low-risk ETFs like VOO).
    First, output your step-by-step reasoning as plain text without markdown symbols or JSON.
    Then, provide the final plan in JSON with: budget_breakdown (dict of categories:amounts), investments (list of suggestions with reasons), goal_savings (dict: goal -> monthly amount), total_savings_rate."""), 
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def run_agent(goal: str, salary: float, expenses: float, needs_pct: float, wants_pct: float, savings_pct: float, other_goals: str) -> tuple[str, str, str]:
    """
    Run the agent with user inputs.
    Returns: reasoning, JSON plan, chart HTML
    """
    # Prepare goal input including multiple goals
    goals_info = f"Primary goal: {goal}."
    if other_goals.strip():
        goals_info += f" Additional goals: {other_goals}."
    
    input_text = (
        f"{goals_info} Annual salary: ${salary:,.2f}. Estimated annual expenses: ${expenses:,.2f}. "
        f"Budget split - Needs: {needs_pct}%, Wants: {wants_pct}%, Savings: {savings_pct}%."
    )
    
    # Execute agent
    try:
        result = agent_executor.invoke({"input": input_text})
        raw_output = result.get('output', 'No output generated.')
        
        # Extract reasoning (before JSON)
        plan_start = raw_output.find('{')
        if plan_start == -1:
            reasoning = raw_output
            plan_json = {"error": "No JSON plan found"}
        else:
            reasoning = raw_output[:plan_start].strip()
            reasoning = re.sub(r'[\*\#\-\_\`]', '', reasoning).strip()
            try:
                plan_json = json.loads(raw_output[plan_start:raw_output.rfind('}') + 1])
            except:
                plan_json = {"error": "Could not parse plan", "details": raw_output}
        
        plan_str = json.dumps(plan_json, indent=2)
        
        # Generate pie chart for budget_breakdown
        if "budget_breakdown" in plan_json and isinstance(plan_json["budget_breakdown"], dict):
            df = pd.DataFrame(list(plan_json["budget_breakdown"].items()), columns=["Category", "Amount"])
            df["Amount"] = pd.to_numeric(df["Amount"], errors='coerce')
            fig = px.pie(df, values="Amount", names="Category", title="Budget Allocation")
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
        else:
            chart_html = "<p>No budget data for chart.</p>"
        
        return reasoning, plan_str, chart_html
    except Exception as e:
        return f"Error: {str(e)}", "{}", "<p>Error generating chart.</p>"

# Custom Gradio theme
custom_theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="green",
    neutral_hue="slate",
).set(
    body_background_fill="*black",
    body_background_fill_dark="*black",
    body_text_color="*white",
    body_text_color_dark="*white",
    button_primary_background_fill="*blue.700",
    button_primary_background_fill_hover="*blue.600",
    button_primary_text_color="*white",
    slider_color="*green.500",
    slider_color_dark="*green.500",
)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
body, .gradio-container {
    font-family: 'Orbitron', sans-serif !important;
    background-color: #000000 !important;
    color: #ffffff !important;
}
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    color: #ffffff !important;
}
button.primary {
    background-color: #1e40af !important;
    color: #ffffff !important;
    border: 1px solid #ffffff !important;
}
button.primary:hover {
    background-color: #2563eb !important;
}
input, textarea {
    background-color: #1a1a1a !important;
    border: 1px solid #ffffff !important;
    color: #ffffff !important;
}
.output-text, .output-html, .output-text textarea {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #ffffff !important;
}
label, .label {
    color: #ffffff !important;
}
.slider .svelte-1gfknpe {
    background-color: #1a1a1a !important;
}
.slider .svelte-1gfknpe .thumb {
    background-color: #ffffff !important;
}
"""

# Gradio UI
def create_ui():
    with gr.Blocks(title="WealthWise Agent - Step 1", theme=custom_theme, css=custom_css) as demo:
        gr.Markdown("# WealthWise Agent: Personal Finance Planner")
        gr.Markdown("Enter your financial goal, salary, expenses, and optionally multiple goals or custom budget split.")
        
        with gr.Row():
            goal_input = gr.Textbox(label="Primary Financial Goal", placeholder="e.g., Save for a house in 5 years", lines=2)
            other_goals_input = gr.Textbox(label="Additional Goals (optional)", placeholder="e.g., Retirement, Vacation", lines=2)
        
        with gr.Row():
            salary_slider = gr.Slider(minimum=20000, maximum=500000, value=50000, step=1000, label="Annual Salary ($)")
            expenses_slider = gr.Slider(minimum=10000, maximum=200000, value=30000, step=1000, label="Annual Expenses ($)")
        
        with gr.Row():
            needs_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Needs (%)")
            wants_slider = gr.Slider(minimum=0, maximum=100, value=30, step=1, label="Wants (%)")
            savings_slider = gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Savings (%)")
        
        run_btn = gr.Button("Generate Plan", variant="primary")
        
        with gr.Row():
            with gr.Column(scale=1):
                log_output = gr.Textbox(label="Agent Reasoning Log", lines=10, interactive=False)
            with gr.Column(scale=1):
                plan_output = gr.Textbox(label="Budget Plan (JSON)", lines=10, interactive=False)
        
        chart_output = gr.HTML(label="Budget Visualization")
        
        run_btn.click(
            fn=run_agent,
            inputs=[goal_input, salary_slider, expenses_slider, needs_slider, wants_slider, savings_slider, other_goals_input],
            outputs=[log_output, plan_output, chart_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=False)
