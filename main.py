import streamlit as st
from streamlit_extras import add_vertical_space as avs
import time
import random
import os
import matplotlib.pyplot as plt
import numpy as np

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

    return q

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
    algos = {"Rot Cutting Buttom UP ":rod_cutting, "Rot Cutting Brute Force ":brute_force}
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



def plot():

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
    st.subheader("Time Difference Between Brute Force and Rod Cutting Algorithm (in milliseconds)")

    plt.figure(figsize=(10, 5))
    plt.plot(n, time_difference_ms, label='Time Difference (Brute Force - Rod Cutting)', color='red', marker='o')

    # Set logarithmic scale for the y-axis
    plt.yscale('log')

    plt.xlabel('Length of Rod (n)')
    plt.ylabel('Time Difference (milliseconds)')
    plt.title('Time Difference with Logarithmic Scale')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(plt.gcf())  # Show the plot in Streamlit
             

def streamlit_app():
     st.set_page_config("Rot Cutting", page_icon='🔪', layout='centered')
     st.title("ROT CUTTING")
     st.subheader("| :blue[Analysis]")
     st.markdown("---")
     avs.add_vertical_space(1)
     st.subheader("Combined Plot: :blue[Rot Cutting] vs :orange[Brute Force]")
     plot()

streamlit_app()

