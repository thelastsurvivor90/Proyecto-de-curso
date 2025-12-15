"""
Sistema de Compresion de Imagenes usando SVD
Estudiante: Deyvi Samuel Barrera Rodriguez
Asignatura: Algebra Lineal
Docente: Ruth Mery Gonzales
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

class CompresorSVD:
    """Clase para comprimir imagenes usando Descomposicion en Valores Singulares"""
    
    def __init__(self):
        self.imagen_original = None
        self.imagen_comprimida = None
        self.valores_singulares = None
        
    def cargar_imagen(self, archivo):
        """Carga una imagen desde archivo"""
        img = Image.open(archivo)
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        self.imagen_original = np.array(img)
        return self.imagen_original
    
    def generar_imagen_ejemplo(self, tamano=(300, 400)):
        """Genera una imagen de ejemplo para demostrar el sistema"""
        altura, ancho = tamano
        imagen = np.zeros((altura, ancho, 3), dtype=np.uint8)
        
        # Crear patron de gradientes y formas
        for i in range(altura):
            for j in range(ancho):
                # Gradiente rojo
                imagen[i, j, 0] = int((i / altura) * 255)
                # Gradiente verde
                imagen[i, j, 1] = int((j / ancho) * 255)
                # Patron azul
                imagen[i, j, 2] = int(((i + j) % 256))
        
        # Agregar algunos circulos
        centro_y, centro_x = altura // 2, ancho // 2
        for i in range(altura):
            for j in range(ancho):
                dist = np.sqrt((i - centro_y)**2 + (j - centro_x)**2)
                if 50 < dist < 70:
                    imagen[i, j] = [255, 255, 0]  # Circulo amarillo
                elif dist < 30:
                    imagen[i, j] = [255, 0, 255]  # Centro magenta
        
        self.imagen_original = imagen
        return imagen
    
    def aplicar_svd(self, imagen_canal):
        """Aplica SVD a un canal de la imagen"""
        U, S, Vt = np.linalg.svd(imagen_canal, full_matrices=False)
        return U, S, Vt
    
    def reconstruir_imagen(self, U, S, Vt, k):
        """Reconstruye la imagen usando solo k valores singulares"""
        # Mantener solo los primeros k componentes
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        # Reconstruir
        reconstruida = U_k @ np.diag(S_k) @ Vt_k
        
        # Asegurar que los valores esten en rango valido [0, 255]
        reconstruida = np.clip(reconstruida, 0, 255)
        
        return reconstruida
    
    def comprimir(self, k):
        """Comprime la imagen usando k valores singulares por canal"""
        if self.imagen_original is None:
            raise ValueError("Primero debe cargar una imagen")
        
        altura, ancho, canales = self.imagen_original.shape
        imagen_comprimida = np.zeros_like(self.imagen_original, dtype=np.float64)
        
        # Guardar valores singulares del primer canal para analisis
        self.valores_singulares = []
        
        # Aplicar SVD a cada canal de color
        for c in range(canales):
            canal = self.imagen_original[:, :, c].astype(np.float64)
            U, S, Vt = self.aplicar_svd(canal)
            
            if c == 0:  # Guardar valores singulares del canal rojo
                self.valores_singulares = S
            
            # Reconstruir con k componentes
            imagen_comprimida[:, :, c] = self.reconstruir_imagen(U, S, Vt, k)
        
        self.imagen_comprimida = imagen_comprimida.astype(np.uint8)
        return self.imagen_comprimida
    
    def calcular_metricas(self, k):
        """Calcula metricas de calidad y compresion"""
        if self.imagen_comprimida is None:
            raise ValueError("Primero debe comprimir la imagen")
        
        altura, ancho, canales = self.imagen_original.shape
        
        # Error Cuadratico Medio (MSE)
        mse = np.mean((self.imagen_original.astype(np.float64) - 
                       self.imagen_comprimida.astype(np.float64))**2)
        
        # PSNR (Peak Signal-to-Noise Ratio)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        # Tasa de compresion
        # Original: altura * ancho * canales valores
        # Comprimida: k * (altura + ancho + 1) * canales valores
        datos_originales = altura * ancho * canales
        datos_comprimidos = k * (altura + ancho + 1) * canales
        tasa_compresion = (1 - datos_comprimidos / datos_originales) * 100
        
        return {
            'mse': mse,
            'psnr': psnr,
            'tasa_compresion': tasa_compresion,
            'tamano_original': datos_originales,
            'tamano_comprimido': datos_comprimidos
        }
    
    def visualizar_comparacion(self, k):
        """Visualiza la imagen original vs comprimida"""
        if self.imagen_comprimida is None:
            self.comprimir(k)
        
        metricas = self.calcular_metricas(k)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Imagen original
        axes[0].imshow(self.imagen_original)
        axes[0].set_title('Imagen Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Imagen comprimida
        axes[1].imshow(self.imagen_comprimida)
        axes[1].set_title(f'Imagen Comprimida (k={k})\n' + 
                         f'PSNR: {metricas["psnr"]:.2f} dB\n' +
                         f'Compresion: {metricas["tasa_compresion"]:.1f}%',
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualizar_valores_singulares(self):
        """Grafica los valores singulares"""
        if self.valores_singulares is None:
            raise ValueError("Primero debe comprimir la imagen")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Grafica lineal
        ax1.plot(self.valores_singulares, 'b-', linewidth=2)
        ax1.set_xlabel('Indice', fontsize=12)
        ax1.set_ylabel('Valor Singular', fontsize=12)
        ax1.set_title('Valores Singulares', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Grafica logaritmica
        ax2.semilogy(self.valores_singulares, 'r-', linewidth=2)
        ax2.set_xlabel('Indice', fontsize=12)
        ax2.set_ylabel('Valor Singular (escala log)', fontsize=12)
        ax2.set_title('Valores Singulares (Escala Logaritmica)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analizar_compresion(self, rangos_k):
        """Analiza como varia la calidad con diferentes valores de k"""
        resultados = []
        
        for k in rangos_k:
            self.comprimir(k)
            metricas = self.calcular_metricas(k)
            resultados.append({
                'k': k,
                'psnr': metricas['psnr'],
                'tasa_compresion': metricas['tasa_compresion'],
                'mse': metricas['mse']
            })
        
        # Visualizar resultados
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ks = [r['k'] for r in resultados]
        psnrs = [r['psnr'] for r in resultados]
        tasas = [r['tasa_compresion'] for r in resultados]
        
        # PSNR vs k
        ax1.plot(ks, psnrs, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Numero de Valores Singulares (k)', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('Calidad vs Compresion', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=30, color='r', linestyle='--', label='Umbral Calidad Aceptable')
        ax1.legend()
        
        # Tasa de compresion vs k
        ax2.plot(ks, tasas, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Numero de Valores Singulares (k)', fontsize=12)
        ax2.set_ylabel('Tasa de Compresion (%)', fontsize=12)
        ax2.set_title('Tasa de Compresion vs k', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, resultados


# Funcion principal de demostracion
def demo_completa():
    """Ejecuta una demostracion completa del sistema"""
    print("="*70)
    print("SISTEMA DE COMPRESION DE IMAGENES USANDO SVD")
    print("Estudiante: Deyvi Samuel Barrera Rodriguez")
    print("Asignatura: Algebra Lineal")
    print("Docente: Ruth Mery Gonzales")
    print("="*70)
    print()
    
    # Crear compresor
    compresor = CompresorSVD()
    
    # Generar imagen de ejemplo
    print("Generando imagen de ejemplo...")
    compresor.generar_imagen_ejemplo(tamano=(300, 400))
    altura, ancho, _ = compresor.imagen_original.shape
    print(f"Dimensiones de la imagen: {altura}x{ancho}")
    print()
    
    # Analizar con diferentes valores de k
    print("Analizando diferentes niveles de compresion...")
    k_valores = [10, 30, 50, 80, 120]
    
    for k in k_valores:
        compresor.comprimir(k)
        metricas = compresor.calcular_metricas(k)
        print(f"\nk = {k}:")
        print(f"  PSNR: {metricas['psnr']:.2f} dB")
        print(f"  Tasa de compresion: {metricas['tasa_compresion']:.1f}%")
        print(f"  MSE: {metricas['mse']:.2f}")
    
    print("\n" + "="*70)
    print("Generando visualizaciones...")
    print("="*70)
    
    # Visualizar comparacion con k=50
    fig1 = compresor.visualizar_comparacion(k=50)
    
    # Visualizar valores singulares
    fig2 = compresor.visualizar_valores_singulares()
    
    # Analisis completo
    fig3, resultados = compresor.analizar_compresion([5, 10, 20, 30, 50, 80, 120, 150])
    
    plt.show()
    
    print("\nDemostracion completada exitosamente!")
    return compresor


# Ejecutar demo si se ejecuta directamente
if __name__ == "__main__":
    compresor = demo_completa()