import streamlit as st
import pandas as pd
import pickle
from category_encoders import TargetEncoder

st.set_page_config(page_title='Classificação de Clientes',
    layout='wide')

st.write("""
# Processo Seletivo Liber Capital: Classificação de clientes
""")

st.write('Projeto desenvolvido por Guilherme Yuji Fernandes.')

data = pd.read_csv('dados.csv')

categorical_columns = data.select_dtypes('object').columns

x = data.drop('Credit_Class', axis = 1)
y = data['Credit_Class']

colunas = data.columns

cat_enc = TargetEncoder(cols = categorical_columns).fit(x,y)

model = pickle.load(open('model_tunned.pkl', 'rb'))

st.subheader('Informe os valores para realizar a classicação:')

st.write('')

var1 = st.selectbox('Status of existing checking account:', list(data[colunas[0]].unique()))

var2 = float(st.text_input('Duration in month:', value = 0))

var3 = st.selectbox('Credit history:', list(data[colunas[2]].unique()))

var4 = st.selectbox('Purpose:', list(data[colunas[3]].unique()))

var5 = float(st.text_input('Credit amount:', value = 0))

var6 = st.selectbox('Saving account/bonds:', list(data[colunas[5]].unique()))

var7 = st.selectbox('Present employment since:', list(data[colunas[6]].unique()))

var8 = float(st.text_input('Installment rate in percentage of disposable income:', value = 0))

var9 = st.selectbox('Personal status and sex:', list(data[colunas[8]].unique()))

var10 = st.selectbox('Other debtors/guarantors:', list(data[colunas[9]].unique()))

var11 = float(st.text_input('Present residence since:', value = 0))

var12 = st.selectbox('Property:', list(data[colunas[11]].unique()))

var13 = float(st.text_input('Age', value = 0))

var14 = st.selectbox('Other installment plans:', list(data[colunas[13]].unique()))

var15 = st.selectbox('Housing:', list(data[colunas[14]].unique()))

var16 = float(st.text_input('Number of credits:', value = 0))

var17 = st.selectbox('Housing:', list(data[colunas[16]].unique()))

var18 = float(st.text_input('Number of people being liable to provide maintenance for', value = 0))

var19 = st.selectbox('Telephone:', list(data[colunas[18]].unique()))

var20 = st.selectbox('Foreign worker:', list(data[colunas[19]].unique()))

st.text(' ')

if st.button('Classificar o cliente'):
	vars = [var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14, var15,
	var16, var17, var18, var19, var20]

	df = pd.DataFrame([vars], columns = x.columns)

	df = cat_enc.transform(df)

	result = model.predict_proba(df)

	st.header(f'Probabilidade de ser Classe 1: {result[0][0]}')
	st.header(f'Probabilidade de ser Classe 2: {result[0][1]}')