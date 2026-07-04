import cv2
import numpy as np
import os
import json
import math
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
    
    momentos = cv2.moments(mask)
    momentos_hu = cv2.HuMoments(momentos).flatten()
    vector_hu = [-1 * math.copysign(1.0, h) * math.log10(abs(h)) if h != 0 else 0 for h in momentos_hu]

    vector_hog = hog(img_resized, orientations=8, pixels_per_cell=(16, 16),
                     cells_per_block=(1, 1), visualize=False)

    lbp = local_binary_pattern(img_resized, P=8, R=1, method="uniform")
    (vector_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)

    vector_final = np.hstack((vector_hu, vector_hog, vector_lbp))
    return vector_final.tolist()

def generar_dataset_express(carpeta_origen, nombre_archivo_json="dataset_hibrido.json"):
    registros = []
    id_contador = 1
    
    print(f"[+] Escaneando carpeta: {carpeta_origen}")
    archivos = os.listdir(carpeta_origen)
    
    for archivo in archivos:
        ruta_img = os.path.join(carpeta_origen, archivo)
        
        # Clasificación automática por el nombre del archivo
        if archivo.endswith('_temp.jpg'):
            nombre_clase = "Sana"
            id_clase = 0
        elif archivo.endswith('_test.jpg'):
            nombre_clase = "Defectuosa"
            id_clase = 1
        else:
            continue # Ignorar otros archivos
            
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

def entrenar_y_evaluar(archivo_json):
    with open(archivo_json, 'r') as f:
        registros = json.load(f)
        
    if len(registros) == 0:
        print("Error: No se extrajeron registros. Verifica la ruta.")
        return

    X = np.array([reg["Features"] for reg in registros])
    Y = np.array([reg["Target"] for reg in registros])
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    modelos = {
        "KNN (K-Nearest Neighbors)": KNeighborsClassifier(n_neighbors=3),
        "SVM (Support Vector Machine)": SVC(kernel='linear'),
        "Naive Bayes (Gaussian)": GaussianNB()
    }
    
    print("\n[+] RESULTADOS DE EVALUACIÓN (FASE 5):")
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X_test)
        acc = accuracy_score(y_test, predicciones)
        matriz = confusion_matrix(y_test, predicciones)
        
        print("-" * 50)
        print(f"Modelo: {nombre}")
        print(f"Precisión Global (Accuracy): {acc * 100:.2f}%")
        print(f"Tabla de Confusión:\n{matriz}")

if __name__ == "__main__":
    # RUTA DIRECTA Y EXACTA A TUS ARCHIVOS
    carpeta_deep_pcb = r"C:\Users\kvela\Documents\Ux\6to semestre\Procesamiento de Imagenes\Parcial 3 (ordinario)\Examen\DeepPCB-master\PCBData\group00041\00041"
    
    if os.path.exists(carpeta_deep_pcb):
        archivo_generado = generar_dataset_express(carpeta_deep_pcb)
        entrenar_y_evaluar(archivo_generado)
    else:
        print(f"Error: No se encontró la ruta:\n{carpeta_deep_pcb}")