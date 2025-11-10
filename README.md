# Interactive Complexity Theory Explorer (Python/Streamlit) 🧩

### Project Architecture

This project is an interactive web application built entirely in Python using the Streamlit framework. The goal is to create an interactive, server-side tool to help students visually and tangibly understand the concepts of complexity theory, particularly P, NP, and NP-Completeness.

### Core Principles

1. **Python-First:** The entire application logic, UI, and state management are handled in a single Python file (`app.py`).

2. **Modular Functions:** The application is divided into "modules." Each module is a distinct Python function (e.g., `show_home()`, `show_visualizer()`) that is responsible for rendering one page.

3. **Interactive Components:** The app uses various Streamlit components (`st.button`, `st.text_area`, `st.tabs`, `st.popover`) to create a rich, interactive learning experience.

4. **State Management:** Streamlit's `st.session_state` is used to store data (like the JSON for the current graph) between button clicks and page re-renders.

5. **Visualization:** The app uses **Altair** and **Pandas** to create clean, interactive, and zoomable charts for visualizing complexity classes.

6. **Core Lesson: P vs. NP vs. Approximation:** The demo modules (VC and TSP) are built around a three-way comparison:

   - Verify (P-time): Checking a solution is fast.
   - Approximate (P-time): Finding a good enough solution (via heuristic or approximation) is also fast.
   - Optimal (NP-hard): Finding the perfect solution (via brute force) is intractably slow.

7. **Beylond NP**: The `PSPACE` module introduces a new complexity class and demonstrates the crucial difference between **Time Complexity** (exponential) and **Space Complexity** (polynomial).

### File Structure

- `app.py`: The entire application.

- Imports: Loads `streamlit`, `pandas`, `altair`, etc.

- Helper Functions: `factorial()`, `generate_random_graph()`, `solve_vc_approx()`, etc.

- Module Functions (`show_...`): Each function (e.g., `show_home()`, `show_vertex_cover()`, `show_reductions()`) defines one "page" of the app.

- `main()` **Function:** The app's entry point. It sets up the page layout and sidebar navigation (`st.sidebar.radio`) and calls the correct module function based on the user's selection.

- `requirements.txt`: A list of the Python dependencies (`streamlit`, `pandas`, `numpy`, `altair`) needed to run the project.

- `README.md`: This file, explaining the project's Streamlit-based architecture.

- `CONTRIBUTIONS.md`: Explains the purpose of each function within `app.py`.

### Project Structure Visualization

Here is a visual representation of the project's simple, flat file structure:

```bash
complexity-explorer/
├── app.py # The entire Streamlit application
├── requirements.txt # Python dependencies
├── README.md # This documentation file
└── CONTRIBUTIONS.md # Code contribution explanations
```

### How to Run This Project

1. Save the files:

   - Save the main application file as `app.py`.

   - Save the dependencies list as `requirements.txt`.

2. Set up a Python environment:

   - Open your terminal or command prompt and create a virtual environment: `python -m venv venv`
   - Activate it: (macOS/Linux: `source venv/bin/activate`) (Windows: `venv\Scripts\activate`)

3. Install the dependencies:

   - Run: `pip install -r requirements.txt`

4. Run the app:

   - Run: `streamlit run app.py`

   - Your web browser will automatically open with the running application.
