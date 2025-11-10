Interactive Complexity Theory Explorer (Python/Streamlit)

Project Architecture

This project is an interactive web application built entirely in Python using the Streamlit framework. The goal is to create an interactive, server-side tool to help students visually and tangibly understand the concepts of complexity theory, particularly P, NP, and NP-Completeness.

Core Principles

Python-First: The entire application logic, UI, and state management are handled in a single Python file (app.py). This is ideal for students and researchers who are more comfortable in Python than in JavaScript.

Modular Functions: The application is divided into "modules." Each module is a distinct Python function (e.g., show_home(), show_visualizer()) that is responsible for rendering one page of the application.

Interactive Components: The app uses various Streamlit components (st.button, st.text_area, st.tabs, st.popover) to create a rich, interactive learning experience.

State Management: Streamlit's st.session_state is used to store data (like the JSON for the current graph) between button clicks and page re-renders.

Visualization: The app uses Altair and Pandas to create clean, interactive, and zoomable charts for visualizing complexity classes.

File Structure

app.py: The entire application.

Imports: Loads streamlit, pandas, altair, etc.

Helper Functions: factorial(), generate_random_graph(), etc.

Module Functions (show\_...): Each function (e.g., show_home(), show_vertex_cover()) defines one "page" of the app.

main() Function: The app's entry point. It sets up the page layout and sidebar navigation (st.sidebar.radio) and calls the correct module function based on the user's selection.

requirements.txt: A list of the Python dependencies (streamlit, pandas, numpy, altair) needed to run the project.

README.md: This file, explaining the project's Streamlit-based architecture.

CONTRIBUTIONS.md: Explains the purpose of each function within app.py.

How to Run This Project

Save the files:

Save the main application file as app.py.

Save the dependencies list as requirements.txt.

Set up a Python environment:

Open your terminal or command prompt.

Create a virtual environment:

python -m venv venv

Activate the environment:

macOS/Linux: source venv/bin/activate

Windows: venv\Scripts\activate

Install the dependencies:

Run the following command:

pip install -r requirements.txt

Run the app:

Run this command in your terminal:

streamlit run app.py

Your web browser will automatically open with the running application.
