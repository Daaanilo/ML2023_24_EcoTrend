import tkinter as tk
from tkinter import ttk
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import tkinter.messagebox as messagebox
import mplcursors


dataset_inquinamento = pd.read_csv("PM2.5.csv")

dataset_inquinamento = dataset_inquinamento.drop_duplicates()
dataset_inquinamento = dataset_inquinamento.dropna()

dataset_inquinamento.to_csv("PM2.5_filtered.csv", index=False)


dataset_gdp = pd.read_excel("GDP.xlsx")

dataset_gdp['Year'] = pd.to_datetime(dataset_gdp['Year'], format='%Y', errors='coerce')
dataset_gdp = dataset_gdp[dataset_gdp['Year'].dt.year >= 2010]
dataset_gdp['Year'] = dataset_gdp['Year'].astype(str)
dataset_gdp['Year'] = dataset_gdp['Year'].str.replace('-01-01', '')

dataset_gdp = dataset_gdp.drop_duplicates()
dataset_gdp = dataset_gdp.dropna()

dataset_gdp.to_csv("GDP_filtered.csv", index=False)

paesi_inquinamento = set(dataset_inquinamento['Country Name'].unique())
paesi_gdp = set(dataset_gdp.columns[1:])
anni = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017])
 
def addestra_modelli(dataset_inquinamento, dataset_gdp, model_type):

    modelli_per_paese_inquinamento = {}
    modelli_per_paese_gdp = {}

    for paese in dataset_inquinamento['Country Name'].unique():
        dati_paese = dataset_inquinamento.loc[dataset_inquinamento['Country Name'] == paese, anni.astype(str)]

        if model_type == 'linear_regression':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge()
        elif model_type == 'lasso':
            model = Lasso()
        elif model_type == 'elastic_net':
            model = ElasticNet()
        elif model_type == 'bayesian_ridge':
            model = BayesianRidge()
        else:
            raise ValueError("Tipo di modello non supportato.")

        model.fit(anni.reshape(-1, 1), dati_paese.values.flatten())
        modelli_per_paese_inquinamento[paese] = model

    for paese in dataset_gdp.columns[1:]:
        X_gdp = dataset_gdp['Year'].astype(int).values.reshape(-1, 1)
        y_gdp = dataset_gdp[paese].values

        if model_type == 'linear_regression':
            model_gdp = LinearRegression()
        elif model_type == 'ridge':
            model_gdp = Ridge()
        elif model_type == 'lasso':
            model_gdp = Lasso()
        elif model_type == 'elastic_net':
            model_gdp = ElasticNet()
        elif model_type == 'bayesian_ridge':
            model_gdp = BayesianRidge()
        else:
            raise ValueError("Tipo di modello non supportato.")

        model_gdp.fit(X_gdp, y_gdp)
        modelli_per_paese_gdp[paese] = model_gdp

    return modelli_per_paese_inquinamento, modelli_per_paese_gdp


modelli_per_paese_inquinamento_linear_regression, modelli_per_paese_gdp_linear_regression = addestra_modelli(dataset_inquinamento.copy(), dataset_gdp.copy(), 'linear_regression')

modelli_per_paese_inquinamento_ridge, modelli_per_paese_gdp_ridge = addestra_modelli(dataset_inquinamento.copy(), dataset_gdp.copy(), 'ridge')

modelli_per_paese_inquinamento_lasso, modelli_per_paese_gdp_lasso = addestra_modelli(dataset_inquinamento.copy(), dataset_gdp.copy(), 'lasso')

modelli_per_paese_inquinamento_elastic_net, modelli_per_paese_gdp_elastic_net = addestra_modelli(dataset_inquinamento.copy(), dataset_gdp.copy(), 'elastic_net')

modelli_per_paese_inquinamento_bayesian_ridge, modelli_per_paese_gdp_bayesian_ridge = addestra_modelli(dataset_inquinamento.copy(), dataset_gdp.copy(), 'bayesian_ridge')



def verifica_paese(paese_selezionato):
    if paese_selezionato not in dataset_inquinamento['Country Name'].unique():
        messagebox.showerror("Errore", f"Il paese '{paese_selezionato}' non è presente nel dataset dell'inquinamento.")
        return False
    elif paese_selezionato not in dataset_gdp.columns[1:]:
        messagebox.showerror("Errore", f"Il paese '{paese_selezionato}' non è presente nel dataset del GDP.")
        return False
    return True


def genera_grafico(modelli_per_paese_inquinamento, modelli_per_paese_gdp, tipo_modello_selezionato):
    paese_selezionato = entry_nome_paese.get()

    if not verifica_paese(paese_selezionato):
        return

    fig = None
    plt.figure(figsize=(15, 10))
    canvas = FigureCanvasTkAgg(fig, master=finestra)
    canvas.draw()
    canvas.get_tk_widget().grid(column=0, row=3, columnspan=2, pady=10, padx=5)

    modello_paese_inquinamento = modelli_per_paese_inquinamento[paese_selezionato]

    modello_paese_gdp = modelli_per_paese_gdp[paese_selezionato]

    dati_paese_inquinamento = dataset_inquinamento.loc[dataset_inquinamento['Country Name'] == paese_selezionato, anni.astype(str)]
    anni_paese = dati_paese_inquinamento.columns.astype(int).values
    livelli_inquinamento_paese = dati_paese_inquinamento.values.flatten()

    dati_paese_gdp = dataset_gdp.loc[:, ['Year', paese_selezionato]]
    anni_paese_gdp = dati_paese_gdp['Year'].astype(int).values
    valori_gdp_paese = dati_paese_gdp[paese_selezionato].values

    fig, ax1 = plt.subplots(figsize=(15, 10))
    fig.subplots_adjust(left=0.1, right=0.75, top=0.85, bottom=0.2)

    ax1.set_title(f'Dati e previsioni per l\'inquinamento e il GDP in {paese_selezionato} ({tipo_modello_selezionato})')
    ax1.set_xlabel('Anno')
    ax1.set_ylabel('Livello di inquinamento', color='blue')
    ax1.tick_params('y', colors='blue')

    ax2 = ax1.twinx()

    legenda_listbox = tk.Listbox(finestra)
    legenda_listbox.grid(row=3, column=3, sticky='nsew')

    legenda_listbox.insert(tk.END, 'Dati previsti (Inquinamento)')

    mse_inquinamento = mean_squared_error(livelli_inquinamento_paese, modello_paese_inquinamento.predict(anni.reshape(-1, 1)))
    r2_inquinamento = r2_score(livelli_inquinamento_paese, modello_paese_inquinamento.predict(anni.reshape(-1, 1)))

    legenda_listbox.insert(tk.END, f'MSE: {mse_inquinamento:.4f}')
    legenda_listbox.insert(tk.END, f'R²: {r2_inquinamento:.4f}')

    for anno in range(2018, 2100):  # Parti dall'anno 2018
        previsione = modello_paese_inquinamento.predict(np.array([[anno]]).reshape(1, -1))
        
        str_app = str(f'{previsione[0]:.2f} µg/m3 ({anno})')
        legenda_listbox.insert(tk.END, str_app)

        if previsione[0] <= 0:
            break

    ax1.scatter(anni_paese, livelli_inquinamento_paese, label='Dati reali (Inquinamento)', color='blue')

    ax1.plot(range(2010, anno + 1), modello_paese_inquinamento.predict(np.array([range(2010, anno + 1)]).reshape(-1, 1)), linestyle='dashed', color='green', label='Previsione (Inquinamento)')

    ax2.scatter(anni_paese_gdp, valori_gdp_paese, label='Dati reali (GDP)', color='red')

    ax2.plot(range(2010, anno + 1), modello_paese_gdp.predict(np.array([range(2010, anno + 1)]).reshape(-1, 1)), linestyle='dashed', color='orange', label='Previsione (GDP)')
    
    legenda_listbox.insert(tk.END, 'Dati previsti (GDP)')

    for anno in range(2018, 2100):
        previsione_gdp = modello_paese_gdp.predict(np.array([[anno]]).reshape(1, -1))
        
        str_app = str(f'GDP: {previsione_gdp[0]:.2f} ({anno})')
        legenda_listbox.insert(tk.END, str_app)

        previsione_inquinamento = modello_paese_inquinamento.predict(np.array([[anno]]).reshape(1, -1))
        if previsione_inquinamento[0] <= 0:
            break

    ax2.plot(range(2010, anno + 1), modello_paese_gdp.predict(np.array([range(2010, anno + 1)]).reshape(-1, 1)), linestyle='dashed', color='orange')

    ax2.set_ylabel('Valore GDP', color='red')
    ax2.tick_params('y', colors='red')

    ax1.plot(range(2010, anno + 1), modello_paese_inquinamento.predict(np.array([range(2010, anno + 1)]).reshape(-1, 1)), linestyle='dashed', color='green')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1.0))

    scrollbar_legenda = ttk.Scrollbar(finestra, orient="vertical", command=legenda_listbox.yview)
    scrollbar_legenda.grid(row=3, column=5, sticky='ns')
    legenda_listbox.configure(yscrollcommand=scrollbar_legenda.set)

    canvas = FigureCanvasTkAgg(fig, master=finestra)
    canvas.draw()
    canvas.get_tk_widget().grid(column=0, row=3, columnspan=2, pady=10)

    mplcursors.cursor(hover=True)


finestra = tk.Tk()
finestra.title("EcoTrend - Analisi PM2.5 e GDP")


stile = ttk.Style()
stile.configure("TLabel", font=("Helvetica", 12), foreground="navy")
stile.configure("TButton", font=("Helvetica", 14, "bold"), foreground="white", background="navy")


etichetta_nome_paese = ttk.Label(finestra, text="Nome Paese:")
etichetta_nome_paese.grid(column=0, row=0, padx=10, pady=10)

entry_nome_paese = ttk.Entry(finestra, font=("Helvetica", 12))
entry_nome_paese.grid(column=1, row=0, padx=10, pady=10)

bottone_linear_regression = ttk.Button(finestra, text="Linear Regression", command=lambda: genera_grafico(modelli_per_paese_inquinamento_linear_regression, modelli_per_paese_gdp_linear_regression, "Linear Regression"))
bottone_linear_regression.grid(column=0, row=2, pady=10, padx=5, sticky='nsew')  

bottone_ridge = ttk.Button(finestra, text="Ridge", command=lambda: genera_grafico(modelli_per_paese_inquinamento_ridge, modelli_per_paese_gdp_ridge, "Ridge Regression"))
bottone_ridge.grid(column=1, row=2, pady=10, padx=5, sticky='nsew')  

bottone_lasso = ttk.Button(finestra, text="Lasso", command=lambda: genera_grafico(modelli_per_paese_inquinamento_lasso, modelli_per_paese_gdp_lasso, "Lasso Regression"))
bottone_lasso.grid(column=2, row=2, pady=10, padx=5, sticky='nsew')  

bottone_elastic_net = ttk.Button(finestra, text="Elastic Net", command=lambda: genera_grafico(modelli_per_paese_inquinamento_elastic_net, modelli_per_paese_gdp_elastic_net, "ElasticNet"))
bottone_elastic_net.grid(column=3, row=2, pady=10, padx=5, sticky='nsew')  

bottone_bayesian_ridge = ttk.Button(finestra, text="Bayesian Ridge", command=lambda: genera_grafico(modelli_per_paese_inquinamento_bayesian_ridge, modelli_per_paese_gdp_bayesian_ridge, "Bayesian Ridge Regression"))
bottone_bayesian_ridge.grid(column=4, row=2, pady=10, padx=5, sticky='nsew')  

finestra.mainloop()