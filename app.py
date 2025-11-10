import streamlit as st
import json
import math
import pandas as pd
import numpy as np
import altair as alt
import time
import random
from itertools import permutations
import copy

# --- Utility Functions ---

def factorial(n):
    """Calculates factorial, with a cap to prevent overflow."""
    if n > 20:
        return np.inf
    return math.factorial(n)

# --- Graph Generator Functions ---
def generate_random_graph(n, p):
    """Generates a random graph as an adjacency list."""
    adj_list = {str(i): [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj_list[str(i)].append(str(j))
                adj_list[str(j)].append(str(i))
    return adj_list

def generate_complete_graph(n):
    """Generates a complete graph (K_n)."""
    return {str(i): [str(j) for j in range(n) if i != j] for i in range(n)}

def generate_random_cities(n):
    """Generates n random cities."""
    return {
        chr(65 + i): {"x": random.randint(0, 100), "y": random.randint(0, 100)}
        for i in range(n)
    }
    
# --- Algorithm for VC Approximation ---
def solve_vc_approx(adj_list):
    """
    Implements the 2-approximation algorithm for Vertex Cover. O(E) time.
    """
    start_time = time.time()
    op_count = 0
    
    # Deep copy to avoid modifying the original
    edges = set()
    for u, neighbors in adj_list.items():
        for v in neighbors:
            if u < v: # Avoid duplicate edges and self-loops
                edges.add(tuple(sorted((u, v))))
                
    approx_cover = set()
    
    while edges:
        op_count += 1
        u, v = edges.pop()
        approx_cover.add(u)
        approx_cover.add(v)
        
        edges_to_remove = set()
        for edge in edges:
            op_count += 1
            if edge[0] in (u, v) or edge[1] in (u, v):
                edges_to_remove.add(edge)
        edges -= edges_to_remove
        
    end_time = time.time()
    return approx_cover, op_count, (end_time - start_time)

# --- Algorithm for TSP Heuristic ---
def get_dist(city_a, city_b):
    """Calculates Euclidean distance between two cities."""
    return math.sqrt((city_a['x'] - city_b['x'])**2 + (city_a['y'] - city_b['y'])**2)

def solve_tsp_heuristic(cities):
    """
    Implements the Nearest Neighbor heuristic for TSP. O(n^2) time.
    """
    start_time = time.time()
    op_count = 0
    
    city_names = list(cities.keys())
    if not city_names:
        return [], 0, 0, 0
        
    start_city = city_names[0]
    tour = [start_city]
    unvisited = set(city_names[1:])
    total_dist = 0
    
    current_city_name = start_city
    
    while unvisited:
        op_count += 1
        nearest_city = None
        min_dist = float('inf')
        
        current_city_coords = cities[current_city_name]
        
        for city_name in unvisited:
            op_count += 1
            city_coords = cities[city_name]
            dist = get_dist(current_city_coords, city_coords)
            
            if dist < min_dist:
                min_dist = dist
                nearest_city = city_name
        
        if nearest_city:
            total_dist += min_dist
            current_city_name = nearest_city
            tour.append(current_city_name)
            unvisited.remove(current_city_name)
        else:
            break
            
    # Add distance back to start
    total_dist += get_dist(cities[current_city_name], cities[start_city])
    tour.append(start_city)
    
    end_time = time.time()
    return tour, total_dist, op_count, (end_time - start_time)


# --- Module: Home ---
def show_home():
    st.title("Welcome to the Complexity Explorer!")
    st.markdown("""
    <p class"text-lg text-gray-300 mb-6">
    This tool is designed to help you, a university student, get an intuitive feel for computational complexity theory.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("What is Complexity Theory?")
        st.markdown("""
        It's the study of how "hard" computational problems are. We classify problems by the *resources* (usually time or memory) required to solve them as the input size 'n' grows.
        """)
        
        st.header("The Big Classes: P vs. NP")
        st.markdown(r"""
        - **<span style='color: #22c55e;'>P (Polynomial Time):</span>**
          These are "efficiently solvable" problems. We can find a solution in a time that is a polynomial function of the input size $n$.
          <br>
        """, unsafe_allow_html=True)
        st.latex(r"Time \le O(n^k) \text{ for some constant } k.")
        st.markdown(r"""
          <br>**Example:** Sorting an array ($O(n \log n)$).
        
        - **<span style='color: #fab005;'>NP (Nondeterministic Polynomial Time):</span>**
          These are "efficiently verifiable" problems. We may not be able to *find* a solution quickly, but if given a potential solution (a "certificate"), we can *check* if it's correct in polynomial time.
        
        - **<span style='color: #f43f5e;'>NP-Complete:</span>**
          These are the "hardest" problems in NP. They have two properties:
          1.  They are in NP.
          2.  Every other problem in NP can be **reduced** to this problem in polynomial time. (Formally: $L \le_p L_{npc}$ for all $L \in \text{NP}$).
        """, unsafe_allow_html=True)
    
    with col2:
        st.header("How To Use This Tool")
        st.markdown(r"""
        Follow these steps to build an intuition for complexity:

        1.  **<span style='color: #06b6d4;'>Explore the Charts (Module 1):</span>**
            See the difference between polynomial ($O(n^2)$) and exponential ($O(2^n)$) growth.
            
        2.  **<span style='color: #06b6d4;'>Feel the P vs. NP Gap (Module 2):</span>**
            - **Verify (P):** Click 'Verify'. It's instant, $O(|V| \cdot |E|)$.
            - **Solve (NP):** Click 'Find Optimal Solution'. For a small graph (n=18), you will *feel* the exponential $O(2^n \cdot |E|)$ delay.
            
        3.  **<span style='color: #06b6d4;'>Compare Brute-Force vs. Heuristic (Module 3):</span>**
            - **Solve (NP):** Click 'Find Optimal Tour' with 9 cities. It takes a few seconds ($O(n!)$).
            - **Approx. (P):** Click 'Find Heuristic Tour' with 9 cities. It's *instant* ($O(n^2)$). The answer will be *good*, but may not be the *best* one. This is the real-world tradeoff.
            
        4.  **<span style='color: #06b6d4;'>Experience Reductions (Module 5):</span>**
            Learn how the **Independent Set** problem is just a "mirror image" of **Vertex Cover**.
            
        5.  **<span style='color: #06b6d4;'>Read the Research (Module 4):</span>**
            See the actual papers that defined these "hard questions" and read Stephen Cook's thoughts on P vs. NP.
        """, unsafe_allow_html=True)

    st.divider()
    
    st.header("Learn More: Foundational Research")
    st.markdown("These are some of the most important papers in the field of computational complexity:")
    
    st.info(
        """
        **Cook, S. A. (1971). "The complexity of theorem-proving procedures."**
        
        **Summary:** This is the foundational paper that birthed the field of NP-Completeness. Cook proved that the **Satisfiability (SAT)** problem has a special property: *any* problem in NP can be "reduced" to it. This means if you could solve SAT efficiently, you could solve *every* problem in NP efficiently. He called this property "NP-Complete."
        """,
        icon="📄"
    )
    
    st.info(
        """
        **Karp, R. M. (1972). "Reducibility among combinatorial problems."**
        
        **Summary:** Karp's paper showed that Cook's discovery wasn't a fluke. He took the idea of "reducibility" and ran with it, identifying 21 other famous, seemingly unrelated problems (including Vertex Cover and TSP) that were also NP-Complete. This established that a vast "web" of problems were all fundamentally the same hard problem, just in different disguises.
        """,
        icon="📄"
    )
    
    st.info(
        """
        **Garey, M. R., & Johnson, D. S. (1979). "Computers and Intractability: A Guide to the Theory of NP-Completeness."**
        
        **Summary:** This is the "bible" of NP-Completeness. It's not a research paper but a comprehensive textbook that standardized the theory, provided a huge catalog of NP-Complete problems, and gave computer scientists a practical 'how-to' manual for identifying and dealing with intractable problems in their own work.
        """,
        icon="📚"
    )

# --- Module: Time Complexity Visualizer ---
def show_visualizer():
    st.title("Module 1: Time Complexity Visualizer")
    st.markdown("Here are **two different charts** to understand the two key scenarios.")

    # --- Chart 1: The Polynomial World (P) ---
    st.header("Chart 1: The 'Fast' Algorithms (Polynomial Time)")
    st.markdown(r"""
    This chart shows *only* the "fast" algorithms on a **linear scale**, for `n` up to 100.
    This lets you see the real, practical difference between $O(n)$, $O(n \log n)$, and $O(n^2)$.
    """)
    
    n_range_p = range(1, 101) # n from 1 to 100
    data_p = []
    for n in n_range_p:
        data_p.append({'n': n, 'Operations': n, 'Complexity': 'O(n)'})
        data_p.append({'n': n, 'Operations': n * math.log2(n), 'Complexity': 'O(n log n)'})
        data_p.append({'n': n, 'Operations': n**2, 'Complexity': 'O(n²)'})
    
    df_p = pd.DataFrame(data_p)
    
    chart_p = alt.Chart(df_p).mark_line().encode(
        x=alt.X('n', title='n (Input Size)'),
        y=alt.Y('Operations', title='Operations (Linear Scale)'), # Linear scale
        color='Complexity',
        tooltip=['n', 'Complexity', 'Operations']
    ).properties(
        title="Polynomial Growth (n=1 to 100)"
    ).interactive() # Enable zoom/pan

    st.altair_chart(chart_p, use_container_width=True)

    # --- Chart 2: The Intractability Wall (P vs. NP) ---
    st.header("Chart 2: The 'Explosion' (P vs. NP)")
    st.markdown(r"""
    This chart compares a "fast" algorithm ($O(n^2)$) to the "slow" ones ($O(2^n)$, $O(n!)$)
    on a **logarithmic scale**.
    1.  At small 'n' (zoom in!), the lines are "jumbled" and cross over.
    2.  After `n=20`, the $O(n!)$ line disappears as it hits infinity (the "factorial wall").
    """)

    n_range_np = range(1, 26) # n from 1 to 25
    data_np = []
    for n in n_range_np:
        data_np.append({'n': n, 'Operations': n**2, 'Complexity': 'O(n²)'})
        data_np.append({'n': n, 'Operations': 2**n, 'Complexity': 'O(2ⁿ)'})
        data_np.append({'n': n, 'Operations': factorial(n), 'Complexity': 'O(n!)'}) # factorial() caps at n=20

    df_np = pd.DataFrame(data_np)

    chart_np = alt.Chart(df_np).mark_line().encode(
        x=alt.X('n', title='n (Input Size)'),
        y=alt.Y('Operations', scale=alt.Scale(type="log"), title='Operations (Log Scale)'), # Log scale
        color='Complexity',
        tooltip=['n', 'Complexity', 'Operations']
    ).properties(
        title="The P vs. NP 'Explosion' (n=1 to 25)"
    ).interactive() # Enable zoom/pan

    st.altair_chart(chart_np, use_container_width=True)

# --- Module: Vertex Cover ---
def show_vertex_cover():
    st.title("Module 2: P vs. NP (Vertex Cover)")

    tab1, tab2 = st.tabs(["What is Vertex Cover?", "Interactive Demo"])

    with tab1:
        st.header("What is Vertex Cover?")
        st.markdown(r"""
        A <span style='color: #06b6d4;'>**Vertex Cover**</span> is a subset of vertices $V' \subseteq V$ in a graph $G=(V, E)$ such that every edge $(u, v) \in E$
        is "covered" (i.e., $\{u, v\} \cap V' \neq \emptyset$).
        """, unsafe_allow_html=True)
        st.latex(r"V' \subseteq V \text{ s.t. } \forall (u, v) \in E, \{u, v\} \cap V' \neq \emptyset")

        st.markdown(r"""
        ### The NP-Complete Problem
        The *optimization* problem is "Find the *minimum* vertex cover."
        The *decision* problem (NP-Complete) is **K-VERTEX-COVER**:
        > "Given $G$ and $k$, does $G$ have a vertex cover of size $k$ or less?"
        """, unsafe_allow_html=True)
        st.latex(r"\text{Does a } V' \subseteq V \text{ exist s.t. } |V'| \le k \text{ and } V' \text{ is a vertex cover for } G?")
        
        st.markdown(r"""
        ### The P vs. NP vs. Approximation Tradeoff
        - **<span style='color: #22c55e;'>Verifying a Solution is FAST (in P):</span>**
          Given $V'$, check if it's a cover. ($O(|V| + |E|)$ time).
          
        - **<span style='color: #3b82f6;'>Finding an Approx. Solution is FAST (in P):</span>**
          The 2-approximation algorithm in the demo runs in $O(|E|)$ time and is *guaranteed* to find a cover no more than 2x the optimal size.
          
        - **<span style='color: #f43f5e;'>Finding an Optimal Solution is SLOW (in NP):</span>**
          Finding the *smallest* cover is hard. The brute-force way is $O(2^|V| \cdot |E|)$ time.
        """, unsafe_allow_html=True)
        
        st.divider()
        st.header("Key Concepts")
        
        st.markdown("**Graph $G=(V, E)$**")
        st.markdown("A set of **Vertices (V)** (nodes) and **Edges (E)** (links) that connect pairs of vertices.")

        st.markdown("**Adjacency List**")
        st.markdown("A representation of a graph where each vertex has a list of its neighbors. This is what the JSON in the demo represents.")

        st.markdown("**Decision Problem vs. Optimization Problem**")
        st.markdown("An **Optimization Problem** asks for the *best* solution (e.g., \"Find the *smallest* vertex cover\"). A **Decision Problem** asks a \"yes/no\" question (e.g., \"Is there a vertex cover of size *k* or less?\").")
        
        st.markdown("**Approximation Algorithm**")
        st.markdown("A fast (P-time) algorithm for an NP-hard optimization problem that guarantees a solution 'close' to the optimal one. For Vertex Cover, we have a **2-approximation**, meaning the cover it finds is at most $2 \times$ the size of the *true* smallest cover.")

    with tab2:
        st.header("Graph Generators")
        
        if 'vc_graph_text' not in st.session_state:
            st.session_state.vc_graph_text = '{"A": ["B", "C"], "B": ["A", "C", "D"], "C": ["A", "B", "D"], "D": ["B",C", "E"], "E": ["D"]}'

        col1, col2, col3 = st.columns(3)
        if col1.button("Generate Random Graph (n=10, p=0.3)", use_container_width=True):
            st.session_state.vc_graph_text = json.dumps(generate_random_graph(10, 0.3))
            st.rerun()
        if col2.button("Generate Complete Graph (K5)", use_container_width=True):
            st.session_state.vc_graph_text = json.dumps(generate_complete_graph(5))
            st.rerun()
        if col3.button("Generate Larger Graph (n=15, p=0.2)", use_container_width=True):
            st.session_state.vc_graph_text = json.dumps(generate_random_graph(15, 0.2))
            st.rerun()

        graph_text = st.text_area("Graph JSON (Adjacency List):", value=st.session_state.vc_graph_text, height=150)
        
        st.header("1. Verify a Solution (Polynomial Time)")
        solution_text = st.text_input("Proposed Solution (e.g., A,D):", value="B,C,E")
        
        if st.button("Verify Solution (Fast)", use_container_width=True, type="secondary"):
            try:
                adj_list = json.loads(graph_text)
                solution_set = set(s.strip() for s in solution_text.split(',') if s.strip())
                
                edges = set()
                for u, neighbors in adj_list.items():
                    for v in neighbors:
                        edge = tuple(sorted((u, v)))
                        edges.add(edge)
                
                uncovered_edges = []
                op_count = 0
                for u, v in edges:
                    op_count += 1
                    if u not in solution_set and v not in solution_set:
                        uncovered_edges.append(f"{u}-{v}")
                
                if not uncovered_edges:
                    st.success(f"VERIFICATION SUCCESSFUL!\nSet {solution_set} is a valid vertex cover.\nOperations: ~{op_count} (polynomial)")
                else:
                    st.error(f"VERIFICATION FAILED!\nThe set does not cover these edges: {', '.join(uncovered_edges)}\nOperations: ~{op_count} (polynomial)")
            except Exception as e:
                st.error(f"Error parsing input: {e}")

        st.header("2. Find a Solution")
        st.markdown("Compare the P-time approximation with the exponential brute-force solver.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Find 2-Approximation (Fast)", use_container_width=True, type="secondary"):
                with st.spinner("Finding approximation..."):
                    try:
                        adj_list = json.loads(graph_text)
                        approx_cover, op_count, time_taken = solve_vc_approx(adj_list)
                        st.info(f"APPROXIMATION FOUND!\nCover: {approx_cover}\nSize: {len(approx_cover)}\nOperations: ~{op_count} (polynomial)\nTime taken: {time_taken:.6f} seconds")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with col2:
            if st.button("Find Optimal Solution (Slow)", use_container_width=True, type="primary"):
                with st.spinner("Solving (Brute Force)... This may take a moment."):
                    try:
                        adj_list = json.loads(graph_text)
                        vertices = list(adj_list.keys())
                        n = len(vertices)
                        
                        if n > 18:
                            st.error("Graph is too large for client-side brute-force (n > 18). This demonstrates exponential complexity!")
                            return

                        edges = set()
                        for u, neighbors in adj_list.items():
                            for v in neighbors:
                                edge = tuple(sorted((u, v)))
                                edges.add(edge)
                        
                        smallest_cover = None
                        op_count = 0
                        
                        start_time = time.time()
                        for i in range(1 << n):
                            op_count += 1
                            subset = set()
                            for j in range(n):
                                if (i >> j) & 1:
                                    subset.add(vertices[j])
                            
                            is_cover = True
                            for u, v in edges:
                                op_count += 1
                                if u not in subset and v not in subset:
                                    is_cover = False
                                    break
                            
                            if is_cover:
                                if smallest_cover is None or len(subset) < len(smallest_cover):
                                    smallest_cover = subset
                        
                        end_time = time.time()
                        st.success(f"OPTIMAL SOLVED!\nCover: {smallest_cover}\nSize: {len(smallest_cover)}\nOperations: ~{op_count} (exponential)\nTime taken: {end_time - start_time:.4f} seconds")
                    
                    except Exception as e:
                        st.error(f"Error parsing input: {e}")

# --- Module: Traveling Salesperson (TSP) ---
def show_tsp():
    st.title("Module 3: NP-Complete (Traveling Salesperson)")
    
    tab1, tab2 = st.tabs(["What is TSP?", "Interactive Demo"])

    with tab1:
        st.header("What is the Traveling Salesperson (TSP)?")
        st.markdown(r"""
        Given a list of cities and the distances between them, what is the <span style='color: #06b6d4;'>**shortest possible route**</span> 
        that visits each city exactly once and returns to the origin city?
        
        ### Formal Definition
        Given a complete weighted graph $G = (V, E, w)$, find a **Hamiltonian cycle**
        with the minimum total weight.
        """, unsafe_allow_html=True)
        st.latex(r"w(v_{\pi(n)}, v_{\pi(1)}) + \sum_{i=1}^{n-1} w(v_{\pi(i)}, v_{\pi(i+1)})")

        st.markdown(r"""
        ### The NP-Complete Problem
        The *optimization* problem is "Find the *shortest* tour."
        The *decision* problem (NP-Complete) is **K-TSP**:
        > "Given $G$, $w$, and $k$, does $G$ have a tour with a total weight less than or equal to $k$?"
        
        ### The P vs. NP vs. Heuristic Tradeoff
        - **<span style='color: #22c55e;'>Verifying a Solution is FAST (in P):</span>**
          Given a path $\pi$, add up the distances ($O(n)$ time).
          
        - **<span style='color: #3b82f6;'>Finding a Heuristic Solution is FAST (in P):</span>**
          The **Nearest Neighbor** heuristic (always go to the closest unvisited city) is $O(n^2)$ time. It provides a "good enough" solution, but no guarantee on its quality.
          
        - **<span style='color: #f43f5e;'>Finding an Optimal Solution is SLOW (in NP):</span>**
          Finding the *shortest* tour is hard. The brute-force way is to check all $(n-1)!$ possible tours ($O(n!)$ time).
        """, unsafe_allow_html=True)
        
        st.divider()
        st.header("Key Concepts")

        st.markdown("**Complete Graph**")
        st.markdown("A graph where every pair of distinct vertices is connected by a unique edge.")

        st.markdown("**Hamiltonian Cycle (or Tour)**")
        st.markdown("A path in a graph that visits each vertex *exactly once* and returns to the starting vertex.")

        st.markdown("**Factorial ($n!$)**")
        st.markdown("The product of all positive integers less than or equal to $n$. ($n! = n \cdot (n-1) \cdot ... \cdot 1$). This function grows *extremely* fast.")
        
        st.markdown("**Heuristic Algorithm**")
        st.markdown("A fast (P-time) algorithm or 'rule of thumb' for solving a problem. Unlike an approximation algorithm, a heuristic gives *no guarantee* on how close its answer is to the optimal one. The 'Nearest Neighbor' heuristic is simple and fast, but can sometimes produce very poor routes.")

    with tab2:
        st.header("City Generators")

        if 'tsp_cities_text' not in st.session_state:
            st.session_state.tsp_cities_text = '{"A": {"x": 0, "y": 0}, "B": {"x": 10, "y": 5}, "C": {"x": 5, "y": 10}, "D": {"x": 15, "y": 0}}'

        col1, col2, col3 = st.columns(3)
        if col1.button("Generate 5 Random Cities", use_container_width=True):
            st.session_state.tsp_cities_text = json.dumps(generate_random_cities(5))
            st.rerun()
        if col2.button("Generate 8 Random Cities", use_container_width=True):
            st.session_state.tsp_cities_text = json.dumps(generate_random_cities(8))
            st.rerun()
        if col3.button("Generate 9 Random Cities", use_container_width=True):
            st.session_state.tsp_cities_text = json.dumps(generate_random_cities(9))
            st.rerun()

        cities_text = st.text_area("Cities JSON:", value=st.session_state.tsp_cities_text, height=150)

        st.header("1. Verify a Tour Length (Polynomial Time)")
        tour_text = st.text_input("Proposed Tour (e.g., A,B,C,D,A):", value="A,C,B,D,A")
        
        if st.button("Verify Tour Length (Fast)", use_container_width=True, type="secondary"):
            try:
                cities = json.loads(cities_text)
                tour = [s.strip() for s in tour_text.split(',') if s.strip()]
                
                if len(tour) < 2:
                    st.error("Tour must have at least 2 cities.")
                    return
                
                total_dist = 0
                op_count = 0
                
                for i in range(len(tour) - 1):
                    op_count += 1
                    city_a_name = tour[i]
                    city_b_name = tour[i+1]
                    if city_a_name not in cities or city_b_name not in cities:
                        st.error(f"Invalid city name in tour: {city_a_name} or {city_b_name}")
                        return
                    total_dist += get_dist(cities[city_a_name], cities[city_b_name])
                
                st.success(f"VERIFICATION COMPLETE!\nTour: {' -> '.join(tour)}\nTotal Length: {total_dist:.2f}\nOperations: ~{op_count} (polynomial)")
            
            except Exception as e:
                st.error(f"Error parsing input: {e}")

        st.header("2. Find a Solution")
        st.markdown("Compare the P-time heuristic with the factorial brute-force solver.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Find Heuristic Tour (Fast)", use_container_width=True, type="secondary"):
                with st.spinner("Finding heuristic tour (Nearest Neighbor)..."):
                    try:
                        cities = json.loads(cities_text)
                        tour, total_dist, op_count, time_taken = solve_tsp_heuristic(cities)
                        st.info(f"HEURISTIC FOUND!\nTour: {' -> '.join(tour)}\nLength: {total_dist:.2f}\nOperations: ~{op_count} (polynomial, $O(n^2)$)\nTime taken: {time_taken:.6f} seconds")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            if st.button("Find Optimal Tour (Very Slow)", use_container_width=True, type="primary"):
                with st.spinner(f"Solving (Brute Force)... This will be very slow if n > 9."):
                    try:
                        cities = json.loads(cities_text)
                        city_names = list(cities.keys())
                        n = len(city_names)

                        if n > 9:
                            st.error(f"Too many cities (n={n}). n > 9 will crash your browser with O(n!) complexity. This is the lesson!")
                            return

                        start_city = city_names[0]
                        other_cities = city_names[1:]
                        
                        shortest_dist = float('inf')
                        shortest_tour = []
                        op_count = 0
                        
                        start_time = time.time()
                        for perm in permutations(other_cities):
                            op_count += 1
                            current_dist = 0
                            current_tour = [start_city] + list(perm) + [start_city]
                            
                            for i in range(len(current_tour) - 1):
                                op_count += 1
                                city_a = cities[current_tour[i]]
                                city_b = cities[current_tour[i+1]]
                                current_dist += get_dist(city_a, city_b)
                            
                            if current_dist < shortest_dist:
                                shortest_dist = current_dist
                                shortest_tour = current_tour
                        
                        end_time = time.time()
                        st.success(f"OPTIMAL SOLVED!\nTour: {' -> '.join(shortest_tour)}\nLength: {shortest_dist:.2f}\nOperations (Permutations): ~{op_count} (factorial)\nTime taken: {end_time - start_time:.4f} seconds")
                    
                    except Exception as e:
                        st.error(f"Error parsing input: {e}")

# --- Module: Open Problems (UPDATED) ---
def show_open_problems():
    st.title("Module 4: Open Problems & The Future")
    st.markdown("""
    Complexity theory is not a "solved" field. It's an active area of research with
    some of the deepest, most difficult questions in all of computer science.
    Click on the terms to learn more.
    """)

    st.header("The 7 Big Questions")

    # --- 1. P vs NP (UPDATED) ---
    st.subheader("1. Is P = NP?")
    st.markdown(r"""
    **The Question:** Is every problem that can be *verified* quickly also *solvable* quickly?
    
    **Formally:** Does $\text{P} = \text{NP}$?
    
    This is the most famous problem in computer science. The consensus is that **P $\neq$ NP**, but proving it remains one of the hardest challenges ever conceived.
    """)
    
    st.info("""
        **Stephen Cook's View (2000):**
        In his paper for the Clay Mathematics Institute (which established the $1M prize),
        Stephen Cook wrote: *"My guess is that the answer is 'no'. ... a proof that P $\neq$ NP
        would be a milestone in mathematics ... a proof that P = NP would be even more
        stunning. It would mean that ... a computer could find a formal proof of any theorem
        which has a proof of reasonable length."*
        
        **Seminal Paper:** Cook, S. A. (2000). *The P vs. NP problem*. Clay Mathematics Institute.
    """)

    # --- 2. NP vs co-NP (UPDATED) ---
    st.subheader("2. Is NP = co-NP?")
    st.markdown(r"""
    **The Question:** If a problem's "yes" answers are easy to verify, are its "no" answers also easy to verify?
    
    **Formally:** Does $\text{NP} = \text{co-NP}$?
    """)
    
    with st.popover("Define: co-NP"):
        st.markdown(r"""
        **co-NP** is the class of problems where "no" instances have a short, verifiable proof (a "counterexample").
        
        **Example: TAUTOLOGY.**
        - **Problem:** Is this Boolean formula *always* true?
        - **"No" Proof:** If the answer is "no," you can provide a counterexample: an assignment of variables that makes the formula *false*.
        - We can verify this counterexample quickly, so TAUTOLOGY is in co-NP.
        """)

    st.markdown(r"""
    If NP $\neq$ co-NP (which is widely believed), it would imply that proving something is *always* true is fundamentally harder than proving it's *sometimes* true (SAT). It would also prove P $\neq$ NP.
    """)
    
    st.info("""
        **Seminal Paper:** Pratt, V. (1975). *Every prime has a succinct certificate*.
        
        This paper famously showed that PRIMES (the problem of checking if a number is prime)
        is in both NP and co-NP, long before it was proven to be in P.
    """)

    # --- 3. ETH (UPDATED) ---
    st.subheader("3. The Exponential Time Hypothesis (ETH)")
    st.markdown(r"""
    **The Question:** Does the 3-SAT problem *really* require exponential time to solve?
    
    **Formally:** The ETH conjectures that there is no algorithm for 3-SAT that runs in *sub-exponential time*.
    """)

    with st.popover("Define: Sub-exponential Time"):
        st.markdown(r"""
        An algorithm that is faster than $O(2^{\delta n})$ for all $\delta > 0$, but still slower than polynomial time.
        
        - **Exponential:** $O(2^n)$
        - **Sub-exponential:** $O(2^{n^{0.5}})$ or $O(2^{\log n})$
        - **Polynomial:** $O(n^2)$
        """)

    st.markdown(r"""
    P vs. NP just asks if there's a *polynomial* algorithm. ETH makes a stronger claim: that 3-SAT *requires* a "strong" exponential-time $O(2^{\delta n})$ algorithm.
    """)
    
    st.info("""
        **Seminal Paper:** Impagliazzo, R., & Paturi, R. (1999). *On the complexity of k-SAT*.
        
        This paper introduced the Exponential Time Hypothesis (ETH) and its variant,
        the Strong Exponential Time Hypothesis (SETH), which are now central assumptions
        in fine-grained complexity.
    """)
    
    # --- 4. P vs PSPACE (UPDATED) ---
    st.subheader("4. Is P = PSPACE?")
    st.markdown(r"""
    **The Question:** Can every problem that is solvable using a *polynomial amount of memory* also be solved using a *polynomial amount of time*?
    
    **Formally:** Does $\text{P} = \text{PSPACE}$?
    """)
    
    with st.popover("Define: PSPACE"):
        st.markdown(r"""
        **PSPACE** is the set of all problems that can be solved by an algorithm using a *polynomial* amount of memory (space), regardless of how much *time* it takes.
        
        Think of solving a chess game. It might take an exponential amount of time to check all moves, but you only need to store the current board state ($O(n^2)$ space) at each step.
        
        We know $\text{P} \subseteq \text{NP} \subseteq \text{PSPACE}$.
        """)
    
    st.markdown(r"""
    We strongly believe $\text{P} \neq \text{PSPACE}$, meaning some problems are *fundamentally* space-efficient but time-inefficient.
    """)
    
    st.info("""
        **Seminal Paper:** Savitch, W. J. (1970). *Relationships between nondeterministic and deterministic tape complexities*.
        
        This paper contains **Savitch's Theorem**, which proves $\text{NSPACE}(S(n)) \subseteq \text{DSPACE}(S(n)^2)$.
        A key corollary is that $\text{PSPACE} = \text{NPSPACE}$, a foundational result.
    """)
    
    # --- 5. BPP vs BQP (UPDATED) ---
    st.subheader("5. Is BPP = BQP? (Quantum Computing)")
    st.markdown(r"""
    **The Question:** Can quantum computers solve problems that classical computers can't?
    
    **Formally:** Does $\text{BPP} = \text{BQP}$?
    """)

    with st.popover("Define: BPP & BQP"):
        st.markdown(r"""
        - **BPP (Bounded-error Probabilistic Polynomial time):**
          The class of problems that a *classical* computer can solve efficiently using randomness (like a Monte Carlo simulation).
          
        - **BQP (Bounded-error Quantum Polynomial time):**
          The class of problems that a *quantum* computer can solve efficiently.
        """)
        
    st.markdown(r"""
    We know $\text{BPP} \subseteq \text{BQP}$. The big question is whether the inclusion is strict. Shor's algorithm for **integer factorization** is in BQP but is *not* believed to be in BPP.
    """)
    
    st.info("""
        **Seminal Papers:**
        1.  Bernstein, E., & Vazirani, U. (1997). *Quantum complexity theory*. (Formally defined BQP).
        2.  Shor, P. W. (1994). *Algorithms for quantum computation: discrete logarithms and factoring*. (Showed Factoring is in BQP).
    """)
    
    # --- 6. Factoring (UPDATED) ---
    st.subheader("6. The Status of Integer Factorization")
    st.markdown(r"""
    **The Question:** Can we factor a large number into its primes in polynomial time (on a classical computer)?
    
    This is the problem that underpins almost all modern cryptography (like RSA). It's in a special class called **NP-Intermediate**.
    """)

    with st.popover("Define: NP-Intermediate"):
        st.markdown(r"""
        Problems that are in **NP**, but are *not* in **P** and are also *not* **NP-Complete**.
        
        They are "hard," but not the "hardest" problems in NP. A fast solver for factoring *would not* mean a fast solver for SAT.
        
        The existence of this class depends on $\text{P} \neq \text{NP}$.
        """)
        
    st.markdown(r"""
    Factoring is in NP (easy to verify) but not believed to be in P. It's also not believed to be NP-Complete.
    """)
    
    st.info("""
        **Seminal Paper:** Shor, P. W. (1994). *Algorithms for quantum computation: discrete logarithms and factoring*.
        
        This paper is so important it's cited twice. It's the key paper for both the power of
        quantum computers (BQP) and the weakness of Factoring.
    """)

    # --- 7. Graph Isomorphism (UPDATED) ---
    st.subheader("7. The Status of Graph Isomorphism")
    st.markdown(r"""
    **The Question:** Can we efficiently determine if two graphs are identical (just with the nodes labeled differently)?
    
    This is another famous "NP-Intermediate" candidate. In 2015, a "quasi-polynomial" time algorithm was announced.
    """)

    with st.popover("Define: Quasi-polynomial Time"):
        st.markdown(r"""
        An algorithm that is *almost* polynomial.
        
        - **Polynomial:** $O(n^k)$
        - **Quasi-polynomial:** $O(2^{(\log n)^c})$
        
        This is *much* faster than exponential ($O(2^n)$) but slightly slower than any true polynomial.
        """)

    st.markdown(r"""
    This breakthrough strongly suggests the problem is *not* NP-Complete, but it's still not known to be in P.
    """)
    
    st.info("""
        **Seminal Paper:** Babai, L. (2015). *Graph Isomorphism in Quasipolynomial Time*.
        
        This paper presented the breakthrough $O(2^{(\log n)^c})$ algorithm,
        a major result that reshaped our understanding of this problem's complexity.
    """)

# --- Module: Reductions ---
def show_reductions():
    st.title("Module 5: Reductions (Independent Set)")
    
    tab1, tab2 = st.tabs(["What is a Reduction?", "Demo: IS ↔ VC"])

    with tab1:
        st.header("What is a Polynomial-Time Reduction?")
        st.markdown(r"""
        A **reduction** is a way to solve one problem using an algorithm for *another* problem.
        A polynomial-time reduction ($L \le_p L'$) is a "fast" transformation that turns an instance
        of problem $L$ into an instance of problem $L'$.
        
        **Why do this?**
        1.  **To Solve Problems:** If you have a "magic" solver for $L'$, you can now solve $L$.
        2.  **To Prove Hardness:** If we know $L$ is "hard" (e.g., NP-Complete), then $L'$ must *also* be "hard." This is how Karp proved 21 problems were NP-Complete.
        
        Formally, $L \le_p L'$ if there's a P-time
        function $f$ that converts any instance $x$ of $L$ into an instance $f(x)$ of $L'$ such that:
        """)
        st.latex(r"x \in L \iff f(x) \in L'")
        
    with tab2:
        st.header("Demo: Independent Set $\leftrightarrow$ Vertex Cover")
        st.markdown(r"""
        Let's show a simple reduction between two problems. This will prove they are
        equally hard.
        
        ### 1. Define: Independent Set (IS)
        An **Independent Set** is a set of vertices $S \subseteq V$ in a graph $G=(V,E)$
        such that no two vertices in $S$ are connected by an edge.
        """)
        st.latex(r"S \subseteq V \text{ s.t. } \forall (u, v) \in E, \{u, v\} \not\subseteq S")
        st.markdown(r"""
        The *decision* problem (NP-Complete) is: "Does $G$ have an independent set of size $k$ or more?"
        
        ### 2. The "Aha!" Moment: The Reduction
        The reduction is based on this theorem:
        """)
        
        st.success(r"""
        **Theorem:** In any graph $G=(V,E)$, a set $S \subseteq V$ is an **Independent Set**
        if and only if its complement, $V \setminus S$, is a **Vertex Cover**.
        """)
        
        st.subheader("Proof (IS $\implies$ VC)")
        st.markdown(r"""
        - **Assume:** $S$ is an Independent Set.
        - **We must show:** Its complement $C = V \setminus S$ is a Vertex Cover.
        - **Proof:** Take any edge $(u, v) \in E$. Since $S$ is an IS, $u$ and $v$ *cannot* both be in $S$.
        - This means at least one of them *must not* be in $S$.
        - If at least one of $\{u, v\}$ is *not* in $S$, then at least one of them *must* be in its complement, $C$.
        - Since this is true for *every* edge, $C = V \setminus S$ is a Vertex Cover.
        """)
        
        st.subheader("Proof (VC $\implies$ IS)")
        st.markdown(r"""
        - **Assume:** $C$ is a Vertex Cover.
        - **We must show:** Its complement $S = V \setminus C$ is an Independent Set.
        - **Proof:** Take any two vertices $u, v \in S$. We must show there is no edge between them.
        - Assume for contradiction that there *is* an edge $(u, v) \in E$.
        - Because $C$ is a Vertex Cover, it must cover this edge. This means at least one of $u$ or $v$ *must* be in $C$.
        - But this is a contradiction! We *chose* $u$ and $v$ from $S$. By definition, neither can be in $C$.
        - The contradiction means our assumption was wrong. There is no edge $(u,v)$.
        - **Therefore, $S = V \setminus C$ is an Independent Set.**
        """)
        
        st.divider()
        st.header("Conclusion")
        st.markdown(r"""
        This proves $\text{Independent Set} \le_p \text{Vertex Cover}$ (and vice-versa).
        
        This also means $|MaxIS| + |MinVC| = |V|$.
        
        The brute-force $O(2^n)$ algorithm in **Module 2** that finds a *minimum vertex cover*
        is, by this reduction, *also* an algorithm for finding a *maximum independent set*.
        """)

# --- Main App ---
def main():
    st.set_page_config(page_title="Complexity Explorer", layout="wide")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", [
        "Home: Start Here", 
        "1. Time Complexity Visualizer", 
        "2. P vs. NP (Vertex Cover)", 
        "3. NP-Complete (TSP)",
        "4. Open Problems & The Future",
        "5. Reductions (IS to VC)"
    ])

    if page == "Home: Start Here":
        show_home()
    elif page == "1. Time Complexity Visualizer":
        show_visualizer()
    elif page == "2. P vs. NP (Vertex Cover)":
        show_vertex_cover()
    elif page == "3. NP-Complete (TSP)":
        show_tsp()
    elif page == "4. Open Problems & The Future":
        show_open_problems()
    elif page == "5. Reductions (IS to VC)":
        show_reductions()

if __name__ == "__main__":
    main()