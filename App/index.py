import pandas as pd
import numpy as np
import streamlit as st

st.write("""

# It's my first APP
hello *world*
formula 
""")

st.latex(r'''f(x) = \frac{\sin(x)⋅\log(5x)}{x}''')


def f(x):
    return np.sin(x) * np.log(5*x) / x


code = ''' 

import numpy as np
import streamlit as st

st.write("""

# It's my first APP
hello *world*
formula 
""")

st.latex(r\'''f(x) = \frac{\sin(x)⋅\log(5x)}{x}\''')


def f(x):
    return np.sin(x) * np.log(5*x) / x


st.line_chart(f(np.arange(0.5, 100, 0.5)))
'''

st.code(code, language='python')

st.line_chart(f(np.arange(0.5, 100, 0.5)))




