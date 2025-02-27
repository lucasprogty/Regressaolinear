import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib

lb = LabelEncoder()
scaler = StandardScaler()

data = pd.read_csv('heart_disease_uci.csv')
df = pd.DataFrame(data)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

#entendendo quais colunas possuimos
print(df.columns)

#tratamento de dados
#como ha valores nulos vamos retira-los
df = df.dropna()

#selecionando todas as colunas categoricas
colunas_categorias_idependentes = df.drop(columns=['num']).select_dtypes(include=['object']).columns


#transformando as colunas categoricas em binarios com o get_dummies
df = pd.get_dummies(df, columns=colunas_categorias_idependentes, drop_first=True)

print(df.columns)
# agora vamos comecar colocar nossa variaveis preditoras e nossa variavel alvo

X = df.drop(columns=['num'])
Y = df['num'] #nossa variavel alvo categorica

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#vamos criar e treinar o modelo de regressao linear

modelo_lr = LinearRegression()
modelo_lr.fit(x_train, y_train)

y_prev = modelo_lr.predict(x_test)

#vamos avaliar o modelo de regressao linear

r2 = r2_score(y_test, y_prev)
print(f'Coeficiente de determinacao - R² para a regresao linear: {r2:.2f}')

rmse = sqrt(mean_squared_error(y_test, y_prev))
print(f'Raiz quadrada do erro medio RMSE para a ressao linear: {rmse:.2f}')


#decidi salvar apenas o modelo de regressao linear porque foi o mais assertivo:
joblib.dump(modelo_lr, 'modelo_teste_regressaoLinear')