import streamlit as st
from streamlit_extras import add_vertical_space as avs
import time
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from graphviz import Digraph as d

def rod_cutting(p:list, n:int):
    r = [0]*(n+1)
    s = [0]*(n+1)
    for j in range(1,n+1):
        q = -999999
        for i in range(1,j+1):
            if q < p[i-1] + r[j-i]:
                q = p[i-1] + r[j-i]
                s[j] = i

        r[j] = q

    return q, r, s

def brute_force(p,n):
    # Base case: no length, no revenue
    if n == 0:
        return 0
    
    q = -999999
    
    # Try every possible cut
    for i in range(1, n + 1):
        q = max(q, p[i - 1] + brute_force(p, n - i))
    
    return q

    

def generate_data():
    algos = {"ROD Cutting Buttom UP ":rod_cutting, "ROD Cutting Brute Force ":brute_force}
    data = []
    for i in range(5,25+1,5):
        p = [random.randint(1,30) for i in range(i)]
        print(p)
        data.append(p)

    for i in data:
        n = len(i)
        p = i
        with open(f"data/length-{len(i)}.txt", 'a') as f:
                 f.write(str(p))

        for k in algos.keys():
                start_time = time.perf_counter()
                print(f'for {k}the q is:',algos[k](p,n))
                end_time = time.perf_counter()
                with open(f"data/length-{len(i)}.txt", 'a') as f:
                    a = str(end_time - start_time) + '\n'
                    f.write(a)



def plot2():

    # Data
    n = [5, 10, 15, 20, 25]
    rot_cutting_seconds = [1.3200100511312485e-05, 5.489983595907688e-05, 6.160000339150429e-05,
                        6.910017691552639e-05, 5.6900084018707275e-05]
    brute_force_seconds = [3.6800047382712364e-05, 0.0003822001162916422, 0.009403400123119354, 
                        0.2910243000369519, 9.437798999948427]

    # Convert seconds to milliseconds
    rot_cutting_ms = [time * 1000 for time in rot_cutting_seconds]
    brute_force_ms = [time * 1000 for time in brute_force_seconds]

    # Calculate the time difference in milliseconds
    time_difference_ms = [bf - rc for bf, rc in zip(brute_force_ms, rot_cutting_ms)]

    # Plotting
    avs.add_vertical_space(2)
    st.subheader("5| :blue[Time Difference Between] :green[Bottom Up] :blue[&] :red[Brute Force]")

    plt.figure(figsize=(10, 5))
    plt.plot(n, time_difference_ms, label='Time Difference (Brute Force - Bottom Up)', color='blue', marker='o')

    # Set logarithmic scale for the y-axis
    plt.yscale('log')

    plt.xlabel('Length of Rod (n)')
    plt.ylabel('Time Difference (milliseconds)')
    plt.title('Time Difference with Logarithmic Scale')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(plt.gcf())  # Show the plot in Streamlit
             

def plot1():
    # Data
    n = [5, 10, 15, 20, 25]
    rot_cutting_seconds = [1.3200100511312485e-05, 5.489983595907688e-05, 6.160000339150429e-05,
                        6.910017691552639e-05, 5.6900084018707275e-05]
    brute_force_seconds = [3.6800047382712364e-05, 0.0003822001162916422, 0.009403400123119354, 
                        0.2910243000369519, 9.437798999948427]

    # Convert seconds to milliseconds
    rot_cutting_ms = [time * 1000 for time in rot_cutting_seconds]
    brute_force_ms = [time * 1000 for time in brute_force_seconds]

    # Creating a dataframe
    data = {
        'Rod Length (n)': n,
        'D.P BOTTOM UP (ms)': rot_cutting_ms,
        'BRUTE FORCE (ms)': brute_force_ms
    }

    df = pd.DataFrame(data)

    # Display the dataframe as a table
    st.table(df)

    # Calculate the time difference in milliseconds
    time_difference_ms = [bf - rc for bf, rc in zip(brute_force_ms, rot_cutting_ms)]

    # Plotting
    st.markdown('---')
    st.subheader("4| :blue[Running Time] :green[Bottom Up] :blue[&] :red[Brute Force] :blue[Vs Time]")

    plt.figure(figsize=(10, 5))
    plt.plot(n, rot_cutting_ms, label='D.P BOTTOM UP (ms)', color='green', marker='o')
    plt.plot(n, brute_force_ms, label='BRUTE FORCE (ms)', color='red', marker='o')

    plt.xlabel('Length of Rod (n)')
    plt.ylabel('Time (milliseconds)')
    plt.title('Time Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    st.pyplot(plt.gcf())


def analysis():
    st.subheader("1| :blue[About]")
    st.markdown("This website showcases the difference in time complexities for both approaches for ROD CUTTING, the brute force approach VS the Dynamic Programming Bottom Up approach. ")
    st.markdown("---")
    st.subheader("2| :blue[Generating Data]")
    st.markdown("- For this experement random data was generated and pushed into the two functions.")
    st.markdown("- The range for the data selected was [1, 30] inclusive.")
     #shocase random data
     # Creating a dataframe
    n = [5, 10, 15, 20, 25]
    p = [[25, 27, 1, 29, 24], 
          [29, 7, 10, 14, 18, 2, 25, 14, 7, 27],
          [30, 25, 30, 15, 27, 11, 3, 25, 12, 24, 8, 8, 20, 9, 3]
          ,[19, 25, 29, 24, 26, 3, 20, 8, 6, 21, 28, 21, 28, 21, 13, 30, 1, 13, 4, 7],
          [20, 10, 16, 11, 7, 30, 11, 29, 13, 16, 1, 26, 30, 18, 26, 19, 22, 2, 1, 23, 18, 19, 16, 5, 16]
        ]
    
     
    data = {
        'Rod Length (n)': n,
        'Random Prices Data (p)': p 
    }
     
    df = pd.DataFrame(data)
    st.table(df)
    st.markdown('---')
    avs.add_vertical_space(1)
    st.subheader("3| :blue[Running Time (ms):] :green[Bottom Up] :blue[Vs] :red[Brute Force]")
    st.markdown("Time complexity:")
    st.code("- BOTTOM UP: n^2", language='')
    st.code("- BRUTE FORCE: 2^n", language='')

    plot1()
    plot2()


# Define the rod-cutting recursive function and generate the recursion tree
def rod_cutting_recursion(prices, n, depth=0, parent=None):
    if n == 0:
        return 0
    
    max_val = float('-inf')
    node_label = f"n={n}, depth={depth}"
    current_node = node_label
    if parent:
        dot.node(current_node, label=node_label)
        dot.edge(parent, current_node)
    else:
        dot.node(current_node, label=node_label)

    for i in range(1, n + 1):
        result = prices[i - 1] + rod_cutting_recursion(prices, n - i, depth + 1, current_node)
        max_val = max(max_val, result)
        
    return max_val



# Function to reconstruct the cuts
def reconstruct_cuts(s, n):
    cuts = []
    while n > 0:
        cuts.append(s[n])
        n -= s[n]
    return cuts

def test_input():
    st.subheader("1| :blue[Test] :green[Bottom-Up] :blue[Vs] :red[Brute Force]")
    avs.add_vertical_space(1)
    n = st.slider(label="| Select Rod Length:",min_value=1, max_value=7, step=1)
    avs.add_vertical_space(1)
    st.markdown("| **Randomly generated data for the prices (p):**")
    p = [2, 10, 16, 11, 7, 30, 11, 29, 13, 16, 1, 26, 30, 18, 26, 19, 22, 2, 1, 23, 18, 19, 16, 5, 16]
    p = p[0:n]
    st.code(p[0:n], language='')

    avs.add_vertical_space(1)
    st.markdown("| **Or you can use your own data:** ")
    new_data = st.text_input("Enter an array of prices (p):", placeholder="[1, 2, 5, 8]")

    if  len(new_data)>1:
        p = eval(new_data)
        n = len(p)

    # Generate the recursion tree
    global dot
    dot = d(format="png")
    rod_cutting_recursion(p, n)
    st.graphviz_chart(dot)

    avs.add_vertical_space(2)

    if st.button("CUT THE ROD!!", use_container_width=True):
        st.markdown("---")

        # get buttom up and time
        st_bu = time.perf_counter()
        bu_q, r, s = rod_cutting(p,n)
        et_bu = time.perf_counter()
        bu_time = et_bu - st_bu
        
        #get brute force and time
        st_bf = time.perf_counter()
        bf_q = brute_force(p,n)
        et_bf = time.perf_counter()
        bf_time = et_bf - st_bf

        rot_cutting_ms = [time * 1000 for time in [bu_time]]
        brute_force_ms = [time * 1000 for time in [bf_time]]
        st.subheader("2| :blue[Running Time Analysis]")
        #print table
        data = {
            'Rod Length (n)': n,
            'D.P BOTTOM UP (ms)': rot_cutting_ms,
            'BRUTE FORCE (ms)': brute_force_ms
        }

        df = pd.DataFrame(data)
        st.table(df)

        c1,c2,c3 = st.columns(3)
        with c2:
            if bu_time < bf_time:
                st.markdown("**:blue[WINNER is] :green[D.P BOTTOM UP] :blue[!]**")
            else:
                st.markdown("**:blue[WINNER is] :red[BRUTE FORCE] :blue[!]**")

        avs.add_vertical_space(2)
        st.subheader("3| :blue[Rod Cutting Solution]")
        
        # reconstruct the rod:
        cuts = reconstruct_cuts(s, n)
        df = pd.DataFrame({'Subproblem': range(1, n + 1), 'Optimal Price': r[1:], 'Cut Position': s[1:]})
        st.table(df)

        st.markdown(f"- **Optimal Total Price: {bu_q}**")

        # Display the final cutting solution

        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh(y=0, width=n, height=0.4, color='grey', edgecolor='black')

        current_length = 0
        for cut in cuts:
            current_length += cut
            ax.axvline(x=current_length, color='green', linestyle='--')

        ax.set_yticks([])
        ax.set_xticks(range(n + 1))
        ax.set_xlim(0, n)
        ax.set_title("Rod Cutting Solution (Green lines indicate cuts)")
        st.pyplot(fig)

st.set_page_config("Rod Cutting", page_icon='🔪', layout='centered')
st.title("ROD CUTTING")

t1, t2 = st.tabs(["Bottom-Up Vs Brute Force Analysis", "Try New Input"])
with t1:
    analysis()

with t2:
     test_input()

