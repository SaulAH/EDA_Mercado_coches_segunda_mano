import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
def comunidades_autonomas(x):
    '''
    doc string
    La función devuelve la Comunidad Autónoma a la que pertenece una provincia
    entrada --> provincia (str)
    salida --> Comunidad autónoma
    '''
    galicia = ['A Coruña','Lugo','Pontevedra','Orense']
    asturias =['Asturias']
    cantabria = ['Cantabria']
    pais_vasco = ['Vizcaya','Guipúzcoa','Álava']
    navarra = ['Navarra']
    la_rioja = ['La Rioja']
    aragon = ['Huesca','Zaragoza','Teruel']
    catalunya = ['Lleida','Girona','Barcelona','Tarragona']
    castilla_y_leon = ['León','Palencia','Burgos','Zamora','Valladolid',
                        'Soria','Salamanca','Ávila','Segovia']
    extremadura = ['Badajoz','Cáceres']
    madrid = ['Madrid']
    castilla_la_mancha = ['Guadalajara','Cuenca','Albacete','Ciudad Real',
                        'Toledo']
    valencia = ['Castellón','Valencia','Alicante']
    baleares = ['Baleares']
    murcia = ['Murcia']
    andalucia = ['Huelva','Sevilla','Jaén','Granada','Almería',
                    'Córdoba','Málaga','Cádiz']
    canarias = ['Las Palmas','Tenerife']
    ceuta = ['Ceuta']
    melilla = ['Melilla']
    if x in galicia:
        comunidad = 'Galicia'
    elif x in asturias:
        comunidad = 'Principado_de_Asturias'
    elif x in cantabria:
        comunidad = 'Cantabria'
    elif x in pais_vasco:
        comunidad = 'País_Vasco'
    elif x in navarra:
        comunidad = 'Navarra'
    elif x in la_rioja:
        comunidad = 'La_Rioja'
    elif x in aragon:
        comunidad = 'Aragón'
    elif x in catalunya:
        comunidad = 'Catalunya'
    elif x in castilla_y_leon:
        comunidad = 'Castilla_y_León'
    elif x in extremadura:
        comunidad = 'Extremadura'
    elif x in madrid:
        comunidad = 'Comunidad_de_Madrid'
    elif x in castilla_la_mancha:
        comunidad = 'Castilla_La_Mancha'
    elif x in valencia:
        comunidad = 'Comunidad_Valenciana'
    elif x in baleares:
        comunidad = 'Islas_Baleares'
    elif x in murcia:
        comunidad = 'Región_de_Murcia'
    elif x in andalucia:
        comunidad = 'Andalucía'
    elif x in canarias:
        comunidad = 'Canarias'
    elif x in ceuta:
        comunidad = 'Ceuta'
    elif x in melilla:
        comunidad = 'Melilla'
    else:
        comunidad = 'Desconocida'
    return comunidad
def quitar_acentos(df, columna):
    '''
    doc string
    La función quita los acentos de la columna que indiquemos como parametro
    entradas:
        - df --> dataframe
        - columna --> columna sobre la que quieremos quitar acentos (str)
    '''
    acentos = {'á':'a','é':'e','í':'i','ó':'o','ú':'u',
    'Á':'A','É':'E','Í':'I','Ó':'O','Ú':'U'}
    for k,v in acentos.items():
        df[columna] = df[columna].str.replace(k,v)
def clean_modelo(df,columna_ref,columna_modif):
    '''
    doc string
    La función completa una columna (columna_modif) verificando si los valores
    que tenga en otra de sus columnas (columna_ref) pueden servir como
    información para la columan en cuestión
    entradas:
        - df
        - columna_ref --> columna sobre la que buscamos los posibles valores
        que ajusten en la columna a modificar
        - columna_modif --> columna a modificar si se encuentran los valores
        adecuados
    salida:
        - valor adquirido de la columna_ref
    '''
    for vers in df[columna_ref][df[columna_modif].isnull()]:
        mod = vers.split()
        for i in mod:
            if i in df[columna_modif].unique():
                df[columna_modif][df[columna_ref] == vers] = i
def complete_colum_null(df, column_ref_1,column_ref_2,column_null):
    '''
    doc string:
    la columna completa los valores nulos de una variable numérica con la 
    media de los valores de las observaciones que si que tienen valores
    entradas:
        - df
        - columna_ref_1 --> columna que se utiliza para filtrar las
        observaciones
        - columna_ref_2 --> columna que se utiliza para filtrar las
        observaciones
        - columna_null --> columna númerica que tiene los valores nulos
    salida:
        - media de los valores de las observaciones que no son nulas
    '''
    df_aux = df.loc[df[column_null].isnull(),[column_ref_1,column_ref_2]]
    for mod in df_aux[column_ref_1].value_counts().index:
        for com in df_aux[column_ref_2].value_counts().index:
            df[column_null][
                (df[column_null].isnull()) &
                (df[column_ref_1] == mod) &
                (df[column_ref_2] == com)] = df[column_null][
                    (df[column_null].notnull()) &
                    (df[column_ref_1] == mod) &
                    (df[column_ref_2] == com)].mean();
def outliers(df):
    ''' 
    La función calcula los outliers de las variables numéricas
    entrada --> df
    salida --> df que indica los maximos, mínimos y nº de outliers
    '''
    df_num = df.select_dtypes(exclude=['object','boolean'])
    dic = pd.DataFrame()
    for i in df_num.columns:
            Q_1 = df_num[i].quantile(0.25)
            Q_3 = df_num[i].quantile(0.75)
            RI = Q_3 - Q_1
            lim_inf = Q_1 - 1.5*RI
            lim_sup = Q_3 + 1.5*RI
            outliers = df_num[(df_num[i] < lim_inf) | (df_num[i] > lim_sup )].shape[0]
            dic[i] = [outliers,lim_inf,lim_sup]
    dic.index = ['num_outliers','valor_min','valor_max']
    return dic
def teorema_central_limite(df_ref,cat,var):
    '''
    La función está pensada para las variables de las que no conocemos
    su distribución. Aplica el teorema central del límite sobre una variable
    numércia y luego aplica el test anova de esa variable sobre las
    categorías de una variable categórica una a una
    entadas:
        - df_ref --> dataframe
        - cat --> categoría sobre la que aplicaremos Anova
        - var --> variable numérica
    salida:
        - df indicando cuales de las medias de las categorías son iguales
        y cuales son distintas
    '''
    df_teorema = pd.DataFrame()
    for i in df_ref[cat].value_counts().index:
            df_teorema[f'{i}'] = np.array(
                [np.mean(df_ref[df_ref[cat]==i].sample(30)
                [var].values) for j in range(100 )])            
    # Ahora voy a comprar cada una de las columnas de ese nuevo dataframe
    dic_anova = {}
    for idx_c,col in enumerate(df_teorema.columns):
        dic_anova[col]=[]
        for idx_i,idx in enumerate(df_teorema.columns):
            valor = stats.f_oneway(df_teorema[col],df_teorema[idx])[1]
            if (idx == col) or (idx_c>idx_i):
                dic_anova[col].append('X')
            elif valor > 0.05:
                dic_anova[col].append('Medias iguales')
            else:
                dic_anova[col].append('Distintas')
    df_anova =pd.DataFrame(dic_anova,index=df_teorema.columns)
    return df_anova
def chi2(df,columnas):
    '''
    La función aplica el test chi2 sobre todas las columnas categóricas del df
    entradas:
        - df --> dataframe
        - columnas --> columnas con variables categóricas
    salidas:
        - muestra por pantalla la dependencia o independencia de las variables
    '''
    for i in range(len(df[columnas].columns)-1):
        for j in range(i+1,len(df[columnas].columns)):
            p_valor = stats.chi2_contingency(
                pd.DataFrame(pd.crosstab(
                    df[df[columnas].columns[i]],df[df[columnas].columns[j]])))[1]
            if p_valor>0.05:
                print(f'hay independencia entre las columnas {df[columnas].columns[i]},{df[columnas].columns[j]}')
            else:
                print(f'hay dependencia entre las columnas {df[columnas].columns[i]},{df[columnas].columns[j]}') 