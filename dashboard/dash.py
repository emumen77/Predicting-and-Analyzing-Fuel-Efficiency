import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import math
import missingno as msn
import streamlit as st


ind_list = []
for i in range(25421):
    ind_list.append(i)


#Importing the excel data files and converting them to csv files

epa_file = pd.DataFrame(pd.read_excel("/home/emumen/Documents/Misk-DSI-2022/Predicting-and-Analyzing-Fuel-Efficiency/data/EPAGreen.xlsx"))
epa_file.to_csv("/home/emumen/Documents/Misk-DSI-2022/Predicting-and-Analyzing-Fuel-Efficiency/data/EPAGreen.csv", index = ind_list, header = True)
epa = pd.read_csv("/home/emumen/Documents/Misk-DSI-2022/Predicting-and-Analyzing-Fuel-Efficiency/data/EPAGreen.csv")
fuel = pd.read_csv("/home/emumen/Documents/Misk-DSI-2022/Predicting-and-Analyzing-Fuel-Efficiency/data/Fuel_Econ.csv")
epa = epa.rename(columns=({'Veh Class':'Veh_Class', "City MPG": "City_MPG", "Cmb MPG": "Cmb_MPG", "Greenhouse Gas Score": "Greenhouse_Score", "Comb CO2": "Comb_CO2", "Unnamed: 0.1": "Indeces", "Hwy MPG": "Hwy_MPG"}))
fuel = fuel.rename(columns=({'# Cyl':'Cyl', "City FE (Guide) - Conventional Fuel": "City_FE", "Comb FE (Guide) - Conventional Fuel": "Comb_FE", "Hwy FE (Guide) - Conventional Fuel": "Hwy_FE", "Annual Fuel1 Cost - Conventional Fuel": "Annual_Cost"}))

st.set_page_config(
    page_title="Analysis of Fuel Efficiency Features",
    page_icon="✅",
    layout="wide",
)

st.title("Analysis of Fuel Efficiency Features")

fuel_filter = st.selectbox("Select Fuel Type", pd.unique(epa["Fuel"]))
epa = epa[epa["Fuel"] == fuel_filter]

Columns = []
for column in epa:
    if epa[column].isna().sum() >= 0.6*26871:
#    if msn.bar(column) == 0:
        Columns.append(column)
    else:
        continue
epa.drop(Columns, axis = 1, inplace = True)
epa.drop(["Cert Region", "Stnd", "Stnd Description", "Underhood ID", "Unnamed: 0", "City_MPG", "Hwy_MPG"], axis = 1, inplace = True)
epa = epa.dropna()

#Comb CO2

a = 0
b = 0
g = 0
Las = []
for ind in epa.index:
    if isinstance(epa["Comb_CO2"][ind], str) == True and "/" in epa["Comb_CO2"][ind]:
        a = epa["Comb_CO2"][ind][0:3]
        e = int(a)
        b = epa["Comb_CO2"][ind][4:]
        f = int(b)
        epa.Comb_CO2[epa.Comb_CO2 == epa["Comb_CO2"][ind]] = (e + f)/2
for ind in epa.index:
    if isinstance(epa["Comb_CO2"][ind], str) == True:
        g = int(epa["Comb_CO2"][ind])
        epa.Comb_CO2[epa.Comb_CO2 == epa["Comb_CO2"][ind]] = g

for ind in epa.index:
    if isinstance(epa["Comb_CO2"][ind], float) == True:
        updt = round(epa["Comb_CO2"][ind])
        updt1 = int(updt)
        epa.Comb_CO2[epa.Comb_CO2 == epa["Comb_CO2"][ind]] = updt1

for ind in epa.index:
    Las.append(epa["Comb_CO2"][ind])


#for ind in epa.index:
#    if isinstance(epa['Comb_CO2'][ind], str) == True:
#        Las.append(epa['Comb_CO2'][ind])

#epa.info()
#print(Las)


#Cmb_MPG

a1 = 0
b1 = 0
g1 = 0
Las1 = []
for ind in epa.index:
    if isinstance(epa["Cmb_MPG"][ind], str) == True and "/" in epa["Cmb_MPG"][ind][0:2]:
        a1 = epa["Cmb_MPG"][ind][0]
        e1 = int(a1)
        b1 = epa["Cmb_MPG"][ind][2:]
        f1 = int(b1)
        epa.Cmb_MPG[epa.Cmb_MPG == epa["Cmb_MPG"][ind]] = (e1 + f1)/2
    elif isinstance(epa["Cmb_MPG"][ind], str) == True and "/" in epa["Cmb_MPG"][ind]:
        a1 = epa["Cmb_MPG"][ind][0:2]
        e1 = int(a1)
        b1 = epa["Cmb_MPG"][ind][3:]
        f1 = int(b1)
        epa.Cmb_MPG[epa.Cmb_MPG == epa["Cmb_MPG"][ind]] = (e1 + f1)/2
for ind in epa.index:
    if isinstance(epa["Cmb_MPG"][ind], str) == True:
        g1 = int(epa["Cmb_MPG"][ind])
        epa.Cmb_MPG[epa.Cmb_MPG == epa["Cmb_MPG"][ind]] = g1

for ind in epa.index:
    if isinstance(epa["Cmb_MPG"][ind], float) == True:
        updt2 = round(epa["Cmb_MPG"][ind])
        updt3 = int(updt2)
        epa.Cmb_MPG[epa.Cmb_MPG == epa["Cmb_MPG"][ind]] = updt3

for ind in epa.index:
    Las1.append(epa["Cmb_MPG"][ind])

l12 = []
for ind in epa.index:
    if epa["Air Pollution Score"][ind] == "Mod":
        epa = epa.drop(ind)

epa['Comb_CO2'] = epa['Comb_CO2'].astype('int')
epa["Cmb_MPG"] = epa["Cmb_MPG"].astype('int')
#epa["City_MPG"] = epa["City_MPG"].astype('int')
epa["Greenhouse_Score"] = epa["Greenhouse_Score"].astype(int)
epa["Air Pollution Score"] = epa["Air Pollution Score"].astype(int)
epa["Drive"] = epa["Drive"].astype(str)
epa["Trans"] = epa["Trans"].astype(str)
epa["Fuel"] = epa["Fuel"].astype(str)
epa["Veh_Class"] = epa["Veh_Class"].astype(str)
epa["Model"] = epa["Model"].astype(str)
epa["SmartWay"] = epa["SmartWay"].astype(str)

#Plotting a correlation matrix in order to compare and find relation among different features, especially those that affect the target (Cmb_MPG)
plt.figure(figsize=(15,12), dpi= 80)
sns.heatmap(epa.corr(), xticklabels=epa.corr().columns, yticklabels=epa.corr().columns, cmap='RdYlGn', center=0, annot=True)



# create two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Correlation Matrix for the Fuel Efficiency Data")
    fig1 = plt.figure(figsize=(15,12), dpi= 80)
    sns.heatmap(epa.corr(), xticklabels=epa.corr().columns, yticklabels=epa.corr().columns, cmap='RdYlGn', center=0, annot=True)
    st.write(fig1)
   
with col2:
    st.markdown("### Correlation Matrix for Conventional Fuel")
    fig2 = fuel.corr().style.background_gradient()

    st.write(fig2)

st.markdown("### Detailed View of the Data")
st.dataframe(epa)


st.markdown("## Extracting the Count for Different Categories")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### Measuring the Count of Each Vehicle Class")
    scat = veh_class_count = epa['Veh_Class'].value_counts()
    #print(veh_class_count)
    x = np.arange(len(veh_class_count.index))
    width = 0.7
    fig, ax = plt.subplots()
    bars = ax.bar(x - width/2, veh_class_count.values, width, color = 'cyan')
    ax.set_ylabel('Count')
    ax.set_title('Count by Vehicle Class')
    ax.set_xticks(x - width/2, veh_class_count.index)
    ax.bar_label(bars)
    plt.xticks(rotation=90)
    plt.show()

with c2:
    st.markdown("### Measuring the Count for 2WD and 4WD Cars")
    drive_count = epa['Drive'].value_counts()
    x = np.arange(len(drive_count.index))
    width = 0.7
    fig, ax = plt.subplots()
    bars = ax.bar(x - width/2, drive_count.values, width, color = 'cyan')
    ax.set_ylabel('Count')
    ax.set_title('Counting 2WD and 4WD')
    ax.set_xticks(x - width/2, drive_count.index)
    ax.bar_label(bars)
    plt.xticks(rotation=90)
    plt.show()