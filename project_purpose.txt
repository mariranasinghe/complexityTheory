This file explains the purpose of each major component in the app.py file.

app.py (The Main Application)

This single file contains all the code for the Streamlit application. It is broken into logical sections:

1. Imports

This section imports all necessary libraries:

streamlit (aliased as st): The core framework for building the web UI.

json, math, time, random, itertools: Standard Python libraries for various helper tasks (data, math, brute-force permutations).

pandas, numpy, altair: Used for data manipulation and creating the interactive charts in Module 1.

2. Utility Functions

factorial(n): A helper function to calculate factorial, with a safety cap at n > 20 to prevent crashing the app during chart generation.

generate_random_graph(n, p): Creates a random graph as an adjacency list (Python dictionary).

generate_complete_graph(n): Creates a fully connected graph.

generate_random_cities(n): Creates a dictionary of city coordinates for the TSP demo.

3. Module Functions (show_...)

These functions are the "pages" of the application. The main() function calls one of them based on the sidebar selection.

show_home():

Purpose: Renders the landing page.

How: Uses st.title, st.markdown, st.columns, and st.info to lay out the page. It introduces P, NP, and NP-Complete, and provides detailed summaries of the foundational research papers.

It uses raw strings (e.g., r"""...""") for st.markdown to ensure that inline LaTeX ($O(n \log n)$) renders correctly.

show_visualizer():

Purpose: To render the "Time Complexity Visualizer" page.

How: This function creates two separate charts using Altair to avoid the visual clutter of one single chart.

Chart 1 shows only Polynomial-time algorithms on a linear scale.

Chart 2 compares P ($O(n^2)$) against NP ($O(2^n)$, $O(n!)$) on a logarithmic scale to show the "explosion."

show_vertex_cover() / show_tsp():

Purpose: To demonstrate the "P vs. NP" gap using classic NP-Complete problems.

How: These modules follow an identical structure:

st.tabs creates a "What is...?" tab and an "Interactive Demo" tab.

The "What is...?" tab provides the formal theory. It uses st.markdown for text and st.latex() for complex, block-level equations to ensure they render cleanly.

The "Interactive Demo" tab provides the hands-on experiment.

st.buttons (like "Generate Random Graph") use st.session_state to update the text in the st.text_area.

"Verify" Button: Triggers a fast, polynomial-time algorithm that just checks the user's solution.

"Solve" Button: Triggers a slow, brute-force, exponential/factorial-time algorithm ($O(2^n)$ or $O(n!)$) that finds the best solution.

show_open_problems():

Purpose: To act as an advanced resource on open research questions.

How: This module is a "deep dive" into the 7 big questions. Its key feature is the use of st.popover. This Streamlit component places a button next to a technical term (like "co-NP" or "BQP"). When a student clicks it, a popover appears with a detailed definition.

show_reductions() (New Feature):

Purpose: To teach the core concept of polynomial-time reductions.

How: Uses st.tabs to separate the theory from the demo. It formally defines a reduction ($L \le_p L'$) and then provides a detailed proof and explanation for the classic Independent Set (IS) $\leftrightarrow$ Vertex Cover (VC) reduction.

4. main() Function

Purpose: This is the entry point that runs the app.

How:

st.set_page_config(layout="wide"): Sets the app to use the full screen width.

st.sidebar.radio(...): Creates the main navigation menu in the left sidebar. The user's selection is stored in the page variable.

A simple if/elif/else block checks the value of page and calls the corresponding show_...() function to render the correct module.