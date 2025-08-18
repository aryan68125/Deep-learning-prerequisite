import pandas as pd
import itertools
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_excel("/home/aditya/github/Deep-learning-prerequisite/pandas/dataset/weather_data_5_cities.xlsx")
print(f"df ---> \n {df}")

# Date	Temperature_C	Humidity_%	WindSpeed_kmph	Precipitation_mm
group_by_obj = df.groupby("City")

color_list = ["red","yellow","orange","green","blue","magenta","purple"]
linestyle_list = ["--","-","-.",":",(0,(5,1,3,5)),(0,(5,10))]

color_cycle = itertools.cycle(color_list)
linsestyle_cycle = itertools.cycle(linestyle_list)

cities = list(group_by_obj.groups.keys())

figures, axes = plt.subplots(len(cities),1,figsize=(10,2*len(cities)),sharex=True)
for ax, (city,city_df) in zip(axes,group_by_obj):
    ax.plot(city_df["Date"],city_df["Temperature_C"],color=next(color_cycle),linestyle=next(linsestyle_cycle))
    ax.set_title(f"Temperature of {city}")
    ax.set_ylabel("Temps in deg celcius")
    ax.grid(True)

plt.xlabel("Dates")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
