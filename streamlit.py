import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path="diamonds.csv"):
    data = pd.read_csv(file_path)

    data = data[
        (data['price'] > 0) &
        (data['x'] > 0) &
        (data['y'] > 0) &
        (data['z'] > 0) 
    ]
    # Function needs to return in order to function properly
    return data

data = load_data()

if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])


# Sidebar

st.sidebar.header("Filters")

for column in data.columns:
    if pd.api.types.is_numeric_dtype(data[column]):
        min_val = float(data[column].min())
        max_val = float(data[column].max())
        selected_range = st.sidebar.slider(
            f"{column}", min_val, max_val, (min_val, max_val)
        )
        data = data[data[column].between(*selected_range)]
    
    elif pd.api.types.is_datetime64_any_dtype(data[column]):
        data[column] = pd.to_datetime(data[column])  # ensure datetime
        min_date = data[column].min()
        max_date = data[column].max()
        selected_dates = st.sidebar.date_input(
            f"{column}", (min_date, max_date)
        )
        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            data = data[data[column].between(*selected_dates)]

    else:
        unique_vals = data[column].dropna().unique()
        selected_vals = st.sidebar.multiselect(f"{column}", sorted(unique_vals), default=sorted(unique_vals))
        data = data[data[column].isin(selected_vals)]

st.title("We are analyzing diamonds with the intentions of investing, we will be analyzing different combinations and mainly looking at what affects price. Trying to find the best investment. We will be using the following format:")
st.markdown('From this chart we can see...')
st.markdown('The following implications:')
st.markdown('This means that...')

# Graph 1
st.title("Diamonds Size vs Price analysis")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].scatter(data['x'], data['price'], alpha=0.5)
axs[0].set_xlabel('Length (x mm)')
axs[0].set_ylabel('Price (USD)')
axs[0].set_title('Price vs Length')

axs[1].scatter(data['y'], data['price'], alpha=0.5, color='orange')
axs[1].set_xlabel('Width (y mm)')
axs[1].set_ylabel('Price (USD)')
axs[1].set_title('Price vs Width')

axs[2].scatter(data['z'], data['price'], alpha=0.5, color='green')
axs[2].set_xlabel('Depth (z mm)')
axs[2].set_ylabel('Price (USD)')
axs[2].set_title('Price vs Depth')

plt.tight_layout()
st.pyplot(fig)
plt.show()

st.markdown('From this chart, we can see how **length**, **width** and **depth** affects price, where length has the biggest price spread.¶')
st.markdown('The following implications:')
st.markdown('* **Width** and **depth** has roughly the same impact on price, and the impact is not huge.')
st.markdown('* **Length** is the one that affects price the most.')
st.markdown('This means that when looking for investments, **wide and deep diamonds** are the most consistent whereas length is the least consistent.')


# Graph 2
colors = sorted(data['color'].unique())
clarities = sorted(data['clarity'].unique())

grouped = data.groupby(['color', 'clarity'])['price'].mean().unstack()

x = np.arange(len(clarities))
bar_width = 0.1

fig, ax = plt.subplots(figsize=(12, 6))

for i, color in enumerate(colors):
    if color in grouped.index:
        prices = grouped.loc[color]
        ax.bar(x + i * bar_width, prices.values, width=bar_width, label=f'Color {color}')

ax.set_xlabel("Clarity")
ax.set_ylabel("Average Price (USD)")
ax.set_title("Average Diamond Price by Clarity and Color")
ax.set_xticks(x + bar_width * (len(colors) / 2))
ax.set_xticklabels(clarities, rotation=45)
ax.legend(title="Color")
ax.grid(axis='y')
fig.tight_layout()

# Display plot in Streamlit
st.pyplot(fig)
st.markdown('From this chart, we can see prices of different combinations of **Clarity** and **Color**. Notably, **Color D** with the clarity level **IF (Internally Flawless)** is the most expensive.')

st.markdown('**The following implications:**')
st.markdown('- High **clarity** and **color** = highest value')
st.markdown('- **Color** impacts price more at higher clarity levels')
st.markdown('- At lower clarity, **color** matters less')

st.markdown('**This means that:**')
st.markdown('For best investment value, choose diamonds with **top clarity and color**.')
st.markdown('For **budget-friendly options**, prioritize **color** over clarity.')


# Graph 3
avg_price_by_cut = data.groupby('cut')['price'].mean()

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(avg_price_by_cut.index, avg_price_by_cut.values, color='lightgreen')
ax.set_xlabel("Cut")
ax.set_ylabel("Average Price (USD)")
ax.set_title("Average Diamond Price by Cut")
ax.grid(axis='y')
fig.tight_layout()

st.pyplot(fig)

st.markdown('From this chart, we can see the average price of diamonds based on their cut. We can see that **Ideal** cut diamonds on average are the cheapest, and **Premium** cut diamonds are the most expensive.')

st.markdown('**The following implications:**')
st.markdown('- Premium diamonds are the most expensive')
st.markdown('- Ideal cut diamonds are the least expensive')

st.markdown('**This means that when investing, this gives a good overview of which diamonds to invest in based on cut.**')

st.markdown('**Executive Summary:**')
st.markdown(
    'This analysis explores how diamond characteristics such as color, clarity, depth (%), and table (%) influence price. '
    'Key findings show that diamonds with high clarity and top color grades command the highest prices, especially at the upper end of the quality spectrum. '
    'Depth and table percentages also play a role, stones with ideal proportions tend to be more valuable. '
    'For investors, selecting diamonds with excellent clarity, color, and balanced proportions offers the best potential for long-term value.'
)
corr = data[['price', 'x', 'y', 'z']].corr()
print("Correlation matrix:\n", corr)

st.write(data.head())

