import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sympy as sp
from sympy import Matrix

import tkinter as tk
from tkinter import ttk, messagebox

# Funciones a graficar
x, y = sp.symbols('x y')

# Clase para manejar las funciones
class Funcion:
    def __init__(self, f):
        self.f = f

    def gradiente(self, *args):
        return Matrix([[sp.diff(self.f[i], x) for x in args] for i in range(len(self.f))])

    def hessiana(self, *args):
        return Matrix([[sp.diff(sp.diff(self.f[i], x), y) for x in args] for i in range(len(self.f)) for y in args])
    
    def planoTangente(self, punto, x, y):
        gradx = self.gradiente(x, y)
        return self.f[0].subs(punto) + gradx[0].subs(punto) * (x - punto[x]) + gradx[1].subs(punto) * (y - punto[y])
    
    def taylor(self, punto, grado):
        x, y = punto.keys()
        if(grado == 2):
            return self.planoTangente(punto, x, y)
        else:
            taylor = self.f[0].subs(punto)
            for i in range(1, grado + 1):
                for j in range(i + 1):
                    term = (sp.diff(self.f[0], x, i - j, y, j).subs(punto) * (x - punto[x])**(i - j) * (y - punto[y])**j) / (sp.factorial(i - j) * sp.factorial(j))
                    taylor += term
            return sp.simplify(taylor)

# Funciones ejemplo
ejemplos = [
    {
        'f': [sp.sin(x) * sp.sin(y)],
        'punto': {x: 0.1, y: 0.1},
        'z0' : 3,
        'limX' : 2.5,
    },
    {
        'f': [sp.E ** (x*y)],
        'punto': {x: 0.1, y: 0.1},
        'z0' : 3,
        'limX' : 2,
    },
    {
        'f': [x* sp.E ** y],
        'punto': {x: 0, y: 0},
        'z0' : 3,
        'limX' : 2,
    },
    {
        'f': [sp.E ** x * sp.ln(1 + y)],
        'punto': {x: 0, y: 0},
        'z0' : 3,
        'limX' : 2,
    },
    {
        'f': [x**2 + y**2],
        'punto': {x: 0.0001, y: 0.0001},
        'z0' : 5,
        'limX' : 3,
    },
    {
        'f': [(x**2 + y - 11)**2 + (x + y**2 - 7)**2],
        'punto': {x: -0.27084, y: -0.92304},
        'z0' : 150,
        'limX' : 4,
    },
    {
        'f': [(-20 * sp.E**(-0.2*sp.sqrt((x**2 + y**2)/2)) - sp.E**((sp.cos(2*sp.pi*x) + sp.cos(2*sp.pi*y))/2) + 20 + sp.E).simplify()],
        'punto': {x: 0, y: 0},
        'z0' : 10,
        'limX' : 10,
    }
]

# Función para graficar
def plot_graph():
    try:
        # Obtener los valores de los widgets
        ejemploN = int(combo_ejemplo.current())
        grado = int(combo_grado.get())
        
        x0 = float(entry_punto_x.get())
        y0 = float(entry_punto_y.get())
        z0 = float(entry_punto_z.get())
        
        limX0 = float(entry_limite_x.get())
        limY0 = float(entry_limite_y.get())
    except ValueError:
        # Mostrar un mensaje de error si los valores no son válidos
        messagebox.showerror("Error", "Por favor, seleccione un ejemplo y un grado válidos.")
        return

    # Crear la función y su aproximación de Taylor
    punto = {x: x0, y: y0}
    funcion_str = ejemplos[ejemploN]['f'][0]
    f = Funcion([sp.sympify(funcion_str)])
    PF = sp.sympify(f.taylor(punto, grado))
    
    # Crear los valores de x y y
    limX = limX0
    limY = limY0
    
    x_vals = np.linspace(-limX, limX, 30)
    y_vals = np.linspace(-limY, limY, 30)
    X, Y = np.meshgrid(x_vals, y_vals)

    z1 = sp.lambdify((x, y), f.f[0], 'numpy')(X, Y)
    z2 = sp.lambdify((x, y), PF, 'numpy')(X, Y)

    # Limpiar los ejes y establecer las etiquetas
    ejes.clear()
    ejes.set_xlabel('X')
    ejes.set_ylabel('Y')
    ejes.set_zlabel('Z')
    ejes.set_zlim(-z0, z0)
    
    # Graficar las superficies
    ejes.plot_surface(X, Y, z1, cmap='winter', alpha=0.5)
    ejes.plot_surface(X, Y, z2, cmap='Reds', alpha=0.5)
    
    # Graficar el punto
    z0 = PF.subs(punto)
    ejes.scatter(punto[x], punto[y], z0, color='red', s=100)
    
    # Mostrar la aproximacion de Taylor
    label_taylor_text.config(state=tk.NORMAL)
    label_taylor_text.delete(1.0, tk.END)
    label_taylor_text.insert(tk.END, sp.pretty(PF))
    label_taylor_text.config(state=tk.DISABLED)
    canvas.draw() # Dibujar la grafica

# Crear la ventana principal
root = tk.Tk()
root.title("Aproximación de Taylor")

# Crear un frame para contener los widgets
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

"""  Creación de la interfaz gráfica  """
# Columna uno
ttk.Label(frame, text="Función:").grid(column=0, row=0, sticky=tk.W) #Seleccion de ejemplo
combo_ejemplo = ttk.Combobox(frame, values=[ejemplos[i]['f'][0] for i in range(len(ejemplos))])
combo_ejemplo.grid(column=1, row=0, sticky=(tk.W, tk.E))
combo_ejemplo.current(0)

ttk.Label(frame, text="Grado:").grid(column=0, row=1, sticky=tk.W) #Seleccion de grado
combo_grado = ttk.Combobox(frame, values=list(range(2, 11)))
combo_grado.grid(column=1, row=1, sticky=(tk.W, tk.E))
combo_grado.current(0)

ttk.Button(frame, text="Graficar", command=plot_graph).grid(column=0, row=2, columnspan=2) #Boton de graficar

# Columna dos
ttk.Label(frame, text="X:").grid(column=3, row=0, sticky=tk.W) #Entrada de punto x
entry_punto_x = ttk.Entry(frame)
entry_punto_x.grid(column=4, row=0, sticky=(tk.W, tk.E))

# Función para actualizar las entradas
def update_entries():
    ejemploN = int(combo_ejemplo.current())
    entry_punto_x.delete(0, tk.END)
    entry_punto_x.insert(0, str(ejemplos[ejemploN]['punto'][x]))
    entry_punto_y.delete(0, tk.END)
    entry_punto_y.insert(0, str(ejemplos[ejemploN]['punto'][y]))
    entry_punto_z.delete(0, tk.END)
    entry_punto_z.insert(0, str(ejemplos[ejemploN]['z0']))
    entry_limite_x.delete(0, tk.END)
    entry_limite_x.insert(0, str(ejemplos[ejemploN]['limX']))
    entry_limite_y.delete(0, tk.END)
    entry_limite_y.insert(0, str(ejemplos[ejemploN]['limX']))

# Actualizar las entradas cuando se selecciona un ejemplo
combo_ejemplo.bind("<<ComboboxSelected>>", lambda event: update_entries())

# Actualizar las entradas con los valores iniciales
entry_punto_x.insert(0, str(ejemplos[0]['punto'][x]))

ttk.Label(frame, text="Y:").grid(column=3, row=1, sticky=tk.W) #Entrada de punto y
entry_punto_y = ttk.Entry(frame)
entry_punto_y.grid(column=4, row=1, sticky=(tk.W, tk.E))
entry_punto_y.insert(0, "0")

ttk.Label(frame, text="Z:").grid(column=3, row=2) #Entrada de punto z
entry_punto_z = ttk.Entry(frame)
entry_punto_z.grid(column=4, row=2, sticky=(tk.W, tk.E))
entry_punto_z.insert(0, "10")

# Columna tres
ttk.Label(frame, text="Limite x:").grid(column=5, row=0, sticky=tk.W) #Entrada de limite x
entry_limite_x = ttk.Entry(frame)
entry_limite_x.grid(column=6, row=0, sticky=(tk.W, tk.E))
entry_limite_x.insert(0, "2")

ttk.Label(frame, text="Limite y:").grid(column=5, row=1, sticky=tk.W) #Entrada de limite y
entry_limite_y = ttk.Entry(frame)
entry_limite_y.grid(column=6, row=1, sticky=(tk.W, tk.E))
entry_limite_y.insert(0, "2")

# Columna cuatro
ttk.Label(frame, text="Taylor: ").grid(column=7, row=1, sticky=tk.W) #Texto de aproximacion de Taylor

# Crear un frame para contener el texto y el scrollbar
frame_taylor = ttk.Frame(frame)
frame_taylor.grid(column=8, row=0, columnspan=2, rowspan=4, sticky=(tk.W, tk.E, tk.N, tk.S))

# Crear el texto y el scrollbar
scrollbar_x = ttk.Scrollbar(frame_taylor, orient=tk.HORIZONTAL)
scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

label_taylor_text = tk.Text(frame_taylor, wrap=tk.NONE, xscrollcommand=scrollbar_x.set, height=4, width=80)
label_taylor_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar_x.config(command=label_taylor_text.xview)
label_taylor_text.config(state=tk.NORMAL)
label_taylor_text.config(state=tk.DISABLED)

scrollbar_x.config(command=label_taylor_text.xview)

# Crear la figura y los ejes 3D
fig = plt.figure(figsize=(8, 4))
ejes = fig.add_subplot(111, projection='3d', box_aspect=(3, 3, 2))
fig.subplots_adjust(left=0, right=0.9, top=1, bottom=0.15)

# Ajustar el tamaño de los números de los ejes
ejes.tick_params(axis='both', which='major', labelsize=8)
ejes.tick_params(axis='both', which='minor', labelsize=6)



# Crear el canvas y mostrarlo
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Ejecutar la aplicación
root.mainloop()
