import cv2
import numpy as np
import os
import json
import math
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

def extraer_vector_hibrido(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    img_resized = cv2.resize(img, (64, 64)) 
    _, mask = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)
    
    # Hu Moments
    momentos = cv2.moments(mask)
    momentos_hu = cv2.HuMoments(momentos).flatten()
    vector_hu = [-1 * math.copysign(1.0, h) * math.log10(abs(h)) if h != 0 else 0 for h in momentos_hu]

    # HOG
    vector_hog = hog(img_resized, orientations=8, pixels_per_cell=(16, 16),
                     cells_per_block=(1, 1), visualize=False)

    # LBP
    lbp = local_binary_pattern(img_resized, P=8, R=1, method="uniform")
    (vector_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)

    vector_final = np.hstack((vector_hu, vector_hog, vector_lbp))
    return vector_final.tolist()

def generar_dataset_profundo(carpeta_raiz, nombre_archivo_json="dataset_kris_13000.json"):
    registros = []
    id_contador = 1
    
    print(f"[+] Escaneando profundamente la carpeta y subcarpetas en:\n{carpeta_raiz}")
    
    # os.walk entra automáticamente a todas las subcarpetas que haya adentro
    for raiz, directorios, archivos in os.walk(carpeta_raiz):
        for archivo in archivos:
            ruta_img = os.path.join(raiz, archivo)
            
            # Clasificación automática por sufijo
            if archivo.endswith('_temp.jpg'):
                nombre_clase = "Sana"
                id_clase = 0
            elif archivo.endswith('_test.jpg'):
                nombre_clase = "Defectuosa"
                id_clase = 1
            else:
                continue 
                
            vector = extraer_vector_hibrido(ruta_img)
            
            if vector is not None:
                registro = {
                    "ID": id_contador,
                    "Label": nombre_clase,
                    "Target": id_clase,
                    "Features": vector
                }
                registros.append(registro)
                id_contador += 1
                
    with open(nombre_archivo_json, 'w') as f:
        json.dump(registros, f, indent=4)
    print(f"[+] Dataset JSON guardado. Total de registros extraídos: {len(registros)}")
    return nombre_archivo_json

def entrenar_evaluar_graficar(archivo_json):
    with open(archivo_json, 'r') as f:
        registros = json.load(f)

    if len(registros) == 0:
        print("Error: No se extrajeron registros.")
        return

    X = np.array([reg["Features"] for reg in registros])
    Y = np.array([reg["Target"] for reg in registros])
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "SVM": SVC(kernel='linear'),
        "NaiveBayes": GaussianNB()
    }
    
    print("\n[+] ENTRENANDO Y GENERANDO GRÁFICAS (FASE 5):")
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X_test)
        acc = accuracy_score(y_test, predicciones)
        matriz = confusion_matrix(y_test, predicciones)
        
        print("-" * 50)
        print(f"Modelo: {nombre} | Accuracy: {acc * 100:.2f}%")
        
        # Generar gráfica de la matriz de confusión
        plt.figure(figsize=(6, 4))
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Sana (0)', 'Defecto (1)'], 
                    yticklabels=['Sana (0)', 'Defecto (1)'],
                    annot_kws={"size": 14})
        
        plt.title(f'Matriz de Confusión - {nombre}\nAccuracy: {acc * 100:.2f}%', fontsize=14, pad=15)
        plt.ylabel('Etiqueta Real', fontsize=12)
        plt.xlabel('Predicción del Modelo', fontsize=12)
        
        nombre_archivo_img = f"Matriz_Kris_{nombre}.png"
        plt.savefig(nombre_archivo_img, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"-> Gráfica guardada: {nombre_archivo_img}")

if __name__ == "__main__":
    # RUTA APUNTANDO A UN GRUPO DISTINTO PARA QUE TUS DATOS SEAN ÚNICOS
    carpeta_deep_pcb = r"C:\Users\kvela\Documents\Ux\6to semestre\Procesamiento de Imagenes\Parcial 3 (ordinario)\Examen\DeepPCB-master\PCBData\group13000"
    
    if os.path.exists(carpeta_deep_pcb):
        archivo_generado = generar_dataset_profundo(carpeta_deep_pcb)
        entrenar_evaluar_graficar(archivo_generado)
    else:
        print(f"Error: No se encontró la ruta:\n{carpeta_deep_pcb}")