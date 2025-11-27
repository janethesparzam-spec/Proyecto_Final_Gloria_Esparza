import streamlit as st #Para la app
import pandas as pd #Para manejo de datos
import matplotlib.pyplot as plt #Para graficos
from sklearn.cluster import KMeans #Para el clustering
from sklearn.preprocessing import LabelEncoder #Para convertir categorias a numeros

st.title("👨‍⚕️​ Análisis de condiciones médicas más frecuentes en EU")
st.markdown("---")

def cargar_datos():
    try:
        df = pd.read_csv('healthcare_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Error: El archivo 'healthcare_dataset.csv' no se encontró.")
        return None

df = cargar_datos()
if df is not None:
    
    st.header("Información general del conjunto de datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total de pacientes", len(df)) #Para calcular el total de pacientes en el df
    
    with col2:
        st.metric("No. de Condiciones médicas", df['Medical Condition'].nunique())#Para calcular el numero de condiciones unicas en el df
            
    st.markdown("---")

    condiciones_frecuencia = df['Medical Condition'].value_counts() #Con esto obtengo la frecuencia de cada condicion medica

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Condiciones:**")
        for i, (condicion, frecuencia) in enumerate(condiciones_frecuencia.items(), 1): #Con esto itero las condiciones para separarlas y numerarlas
            st.write(f"**{i}. {condicion}**")
    
    with col2:
        st.metric("Condición más común", condiciones_frecuencia.idxmax()) #Para obtener la condicion mas comun
    
    st.markdown("---")
    
    st.header("Análisis general") #Tabla de analisis general
    
    resumen_data = [] #Creo una lista vacia para guardar los datos del resumen
    
    for condicion in condiciones_frecuencia.index: #Pongo un index a las condiciones medicas
    
        df_cond = df[df['Medical Condition'] == condicion] #Filtro el df para cada condicion medica
        
        casos = len(df_cond) #Con el len veo el total de filas que tiene cada condicion medica
        
        genero_pred = df_cond['Gender'].value_counts().idxmax() #Aqui obtengo el genero predominante por condicion medica
        
        edad_prom = df_cond['Age'].mean() #Aqui obtengo la edad promedio por condicion medica
        
        resumen_data.append({ #coloco los datos que va a tener la lista resumen_data
            'Condición médica': condicion,
            'Total de casos': casos,
            'Género predominante': genero_pred,
            'Edad promedio': f"{edad_prom:.1f}",
        })
    
    df_resumen = pd.DataFrame(resumen_data) # Ahora convierto la lista que cree en resumen data en una tabla
    
    st.dataframe(df_resumen, use_container_width=True, hide_index=True) #Esto se hace para que se muestre la tabla en streamlit
    
    st.markdown("---")
    
    st.header("Edad promedio por condición médica")
    
    fig, ax = plt.subplots(figsize=(10, 5))  #grafico de barras
    
    condiciones = []  # Lista para guardar nombres de condiciones
    edades = []  # Lista para guardar las edades promedio
    
    for condicion in condiciones_frecuencia.index: # Iterar las condiciones médicas
    
        df_condicion = df[df['Medical Condition'] == condicion] #filtramos para que muestre solo la condicion medica que queremos
        
        edad_promedio = df_condicion['Age'].mean() # Calcular la edad promedio
        
        # Guardar en las listas
        condiciones.append(condicion)
        edades.append(edad_promedio)
    
    posiciones = range(len(condiciones)) #con esto creo la posicion de las barras de acuerdo a la longitud de condiciones
    
    # Dibujar barras naranjas
    ax.bar(posiciones, edades, color='blue', edgecolor='black', linewidth=1) #barras
    
    ax.set_title('Edad Promedio por Condición', fontsize=14, fontweight='bold') #titulo del grafico
    
    # Etiquetas de los ejes
    ax.set_xlabel('Condición Médica', fontsize=12, fontweight='bold')
    ax.set_ylabel('Edad Promedio (años)', fontsize=12, fontweight='bold')
    
    # Poner nombres de condiciones en eje X
    ax.set_xticks(posiciones)
    ax.set_xticklabels(condiciones, rotation=45, ha='right')
    
    ax.grid(axis='y', alpha=0.3) #Agrego una cuadrícula con transparencia del 30%
    
    # Para colocar los números en cada barra
    for i, edad in enumerate(edades):
        ax.text(i, edad + 0.5, f'{edad:.1f}', ha='center', va='bottom', 
               fontsize=9, fontweight='bold')
    
    st.pyplot(fig) #Mostrar el grafico en streamlit
    
    st.markdown("---")

    st.header("Casos por condición médica") #creacion del grafico de lineas
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    condiciones = condiciones_frecuencia.index #Obtengo las condiciones medicas del df
    
    casos = condiciones_frecuencia.values #Obtengo los valores de frecuencia de cada condicion medica
    
    posiciones = range(len(condiciones)) #Creo las posiciones de las condiciones medicas
    
    # Dibujar la línea conectando los puntos
    # marker='o' significa que pone un círculo en cada punto
    # linewidth=2 es el grosor de la línea
    # markersize=8 es el tamaño de los círculos
    ax.plot(posiciones, casos, marker='o', linewidth=2, markersize=8, color='steelblue')
    
    ax.set_title('Número de Casos por Condición Médica', fontsize=14, fontweight='bold') #Titulo del grafico
    
    ax.set_xlabel('Condición Médica', fontsize=12, fontweight='bold') #Etiquetas eje X
    
    ax.set_ylabel('Número de Casos', fontsize=12, fontweight='bold') #Etiquetas eje Y
    
    ax.set_xticks(posiciones) #posicion de las etiquetas
    ax.set_xticklabels(condiciones, rotation=45, ha='right') # se colocann los nombres de las condiciones y se giran las etiquetas 45 grados 
    ax.grid(True, alpha=0.3) #Agrego una cuadrícula con transparencia del 30%
    
    for i, valor in enumerate(casos): #Para agregar los numeros encima de cada punto
        # i es la posición (0, 1, 2...)
        # valor es el número de casos
        # Se coloca el texto un poco arriba del punto (valor + 20)
        ax.text(i, valor + 5, str(valor), ha='center', fontsize=9, fontweight='bold')
    
    
    
    st.pyplot(fig) #para mostrar el grafico en streamlit
    
    st.markdown("---")
    

    st.header("Agrupación de pacientes")
    
    df_cluster = df.copy() #Creo una copia del df original para no modificarlo
    
    # Como son atributos el genero y la condicion, se transforman a numeros ya que kmeans solo trabaja con numeros
    label_gender = LabelEncoder()
    df_cluster['Gender_Numeric'] = label_gender.fit_transform(df_cluster['Gender'])
    
    label_condition = LabelEncoder()
    df_cluster['Condition_Numeric'] = label_condition.fit_transform(df_cluster['Medical Condition'])
    
    datos_para_clustering = df_cluster[['Age', 'Gender_Numeric', 'Condition_Numeric']]
    
    num_clusters = 3
    
    # Creo el modelo kmeans
    modelo_kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    modelo_kmeans.fit(datos_para_clustering)
    df_cluster['Grupo'] = modelo_kmeans.labels_
    
    st.success(f"Modelo entrenado: {num_clusters} grupos creados")
        
    pacientes_por_grupo = df_cluster['Grupo'].value_counts().sort_index() #Resutado del clustering
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribución de pacientes:**")
        for grupo, cantidad in pacientes_por_grupo.items():
            st.write(f"**Grupo {grupo}:** {cantidad} pacientes")
    
    with col2:
      
        fig, ax = plt.subplots(figsize=(6, 6)) #grafica de pastel
        
        colores = plt.cm.Set3(range(num_clusters))
        
        ax.pie(pacientes_por_grupo.values, 
               labels=[f'Grupo {i}' for i in pacientes_por_grupo.index],
               autopct='%1.1f%%',
               startangle=90,
               colors=colores)
        
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("Características por grupo")
    
    grupo_seleccionado = st.selectbox( #Creo el selector de grupo
        "Selecciona el grupo que deseas visualizar:",
        options=[0, 1, 2],
        format_func=lambda x: f"Grupo {x}"
    )
    
    pacientes_grupo = df_cluster[df_cluster['Grupo'] == grupo_seleccionado] # Filtrar pacientes del grupo seleccionado
    
    st.markdown(f"### 👫​ Grupo {grupo_seleccionado} - {len(pacientes_grupo)} pacientes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad_promedio = pacientes_grupo['Age'].mean()
        st.metric("Edad promedio", f"{edad_promedio:.1f} años")
        
        edad_min = pacientes_grupo['Age'].min()
        edad_max = pacientes_grupo['Age'].max()
        st.write(f"Rango de edades: {edad_min} - {edad_max} años")
    
    with col2:
        genero_comun = pacientes_grupo['Gender'].value_counts().idxmax()
        cantidad_genero = pacientes_grupo['Gender'].value_counts().max()
        porcentaje_genero = (cantidad_genero / len(pacientes_grupo)) * 100
        
        st.metric("Género predominante", genero_comun)
        st.write(f"{porcentaje_genero:.1f}% del grupo son de este género")
    
    with col3:
        condicion_comun = pacientes_grupo['Medical Condition'].value_counts().idxmax()
        cantidad_condicion = pacientes_grupo['Medical Condition'].value_counts().max()
        porcentaje_condicion = (cantidad_condicion / len(pacientes_grupo)) * 100
        
        st.metric("Condición más común", condicion_comun)
        st.write(f"{porcentaje_condicion:.1f}% del grupo padecen esta condición")
    
    st.markdown("---")