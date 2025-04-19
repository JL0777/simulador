import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import time


class ComponenteMasaResorte:
    """Clase para representar componentes adicionales como masas o resortes"""
    def __init__(self, tipo, datos):
        self.tipo = tipo  # "masa" o "resorte"
        self.datos = datos  # Diccionario con los datos del componente


class SistemaMasaResorte:
    """Clase principal para manejar un sistema masa-resorte individual"""
    def __init__(self, parent, modo, numero):
        self.parent = parent
        self.modo = modo  # "horizontal" o "vertical"
        self.numero = numero
        self.entradas = {}
        self.componentes_adicionales = []
        self.ecuacion_label = None
        self.animacion = None
        self.tiempo_inicio = 0
        self.temporizador_label = None
        self.temporizador_id = None
        self.masa_obj = None
        self.resorte_obj = None
        self.punto_fijo = None
        self.punto_masa = None
        
        # Frame principal para el sistema
        self.frame = ttk.LabelFrame(parent, text=f"Sistema {numero}")
        self.frame.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.BOTH)
        
        # Crear la interfaz del sistema
        self._crear_interfaz()

    def _crear_interfaz(self):
        """Crea los elementos de la interfaz para un sistema"""
        # Frame para parámetros y controles
        panel_control = ttk.Frame(self.frame)
        panel_control.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Frame para los parámetros
        param_frame = ttk.LabelFrame(panel_control, text="Parámetros")
        param_frame.pack(fill=tk.X, pady=5)
        
        # Añadir campos de entrada para los parámetros
        self._crear_entrada(param_frame, "masa", "Masa (kg):", "1.0")
        self._crear_entrada(param_frame, "k", "Constante k (N/m):", "10.0")
        self._crear_entrada(param_frame, "longitud", "Longitud (m):", "0.5")
        self._crear_entrada(param_frame, "amortiguamiento", "Amortiguamiento:", "0.5")
        if self.modo == "vertical":
            self._crear_entrada(param_frame, "gravedad", "Gravedad (m/s²):", "9.8")
        
        # Botones para añadir componentes
        botones_frame = ttk.Frame(panel_control)
        botones_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(botones_frame, text="+ Masa", command=self._agregar_masa).pack(side=tk.LEFT, padx=5)
        ttk.Button(botones_frame, text="+ Resorte", command=self._agregar_resorte).pack(side=tk.LEFT, padx=5)
        
        # Frame para la ecuación diferencial
        ecuacion_frame = ttk.LabelFrame(panel_control, text="Ecuación Diferencial")
        ecuacion_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.ecuacion_label = tk.Label(ecuacion_frame, text="m·d²x/dt² + c·dx/dt + k·x = mg", wraplength=200)
        self.ecuacion_label.pack(pady=10)
        
        # Actualizar la ecuación
        self._actualizar_ecuacion()
        
        # Panel para visualización de la simulación
        visualizacion_frame = ttk.LabelFrame(self.frame, text="Simulación")
        visualizacion_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Crear figura para la simulación
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        
        # Fijar límites para que no se pierdan los objetos
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        # Crear elementos visuales según el modo
        if self.modo == "vertical":
            self.punto_fijo = (0.5, 0.8)  # Punto fijo en la parte superior
            self.punto_masa = (0.5, 0.5)  # Punto inicial de la masa
            # Añadir base fija (soporte)
            self.ax.add_patch(patches.Rectangle((0.4, 0.8), 0.2, 0.05, color='brown'))
            # Añadir masa
            self.masa_obj = self.ax.add_patch(plt.Circle(self.punto_masa, 0.05, color='blue'))
            # Añadir resorte
            resorte_puntos = self._crear_puntos_resorte(self.punto_fijo, self.punto_masa)
            self.resorte_obj, = self.ax.plot(resorte_puntos[0], resorte_puntos[1], 'k-', lw=2)
        else:  # Horizontal
            self.punto_fijo = (0.2, 0.5)  # Punto fijo a la izquierda
            self.punto_masa = (0.5, 0.5)  # Punto inicial de la masa
            # Añadir base fija (soporte)
            self.ax.add_patch(patches.Rectangle((0.15, 0.4), 0.05, 0.2, color='brown'))
            # Añadir masa
            self.masa_obj = self.ax.add_patch(plt.Circle(self.punto_masa, 0.05, color='blue'))
            # Añadir resorte
            resorte_puntos = self._crear_puntos_resorte(self.punto_fijo, self.punto_masa)
            self.resorte_obj, = self.ax.plot(resorte_puntos[0], resorte_puntos[1], 'k-', lw=2)
        
        # Añadir canvas a la interfaz
        self.canvas = FigureCanvasTkAgg(self.fig, master=visualizacion_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Etiqueta para el temporizador
        self.temporizador_label = ttk.Label(visualizacion_frame, text="Tiempo: 0.00 s")
        self.temporizador_label.pack(pady=5)

    def _crear_entrada(self, frame, key, label, valor_default=""):
        """Crea un campo de entrada con etiqueta"""
        fila = ttk.Frame(frame)
        fila.pack(fill=tk.X, pady=2)
        ttk.Label(fila, text=label, width=20).pack(side=tk.LEFT)
        entry = ttk.Entry(fila)
        entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        entry.insert(0, valor_default)
        self.entradas[key] = entry
        # Vincular evento para actualizar ecuación cuando cambie el valor
        entry.bind("<KeyRelease>", lambda e: self._actualizar_ecuacion())

    def _agregar_masa(self):
        """Maneja la adición de una masa adicional"""
        # Diálogo para obtener los datos de la masa
        masa_valor = simpledialog.askfloat("Agregar Masa", "Valor de la masa (kg):", parent=self.parent)
        
        if masa_valor is not None:
            posicion = simpledialog.askfloat("Agregar Masa", 
                                           "Posición relativa (0-1):", 
                                           parent=self.parent,
                                           minvalue=0, maxvalue=1)
            
            if posicion is not None:
                # Añadir la masa al sistema
                componente = ComponenteMasaResorte("masa", {"valor": masa_valor, "posicion": posicion})
                self.componentes_adicionales.append(componente)
                
                # Actualizar visualización y ecuación
                self._actualizar_sistema_visual()
                self._actualizar_ecuacion()
                messagebox.showinfo("Información", f"Masa de {masa_valor}kg añadida en posición {posicion}")

    def _agregar_resorte(self):
        """Maneja la adición de un resorte adicional"""
        # Diálogo para obtener los datos del resorte
        k_valor = simpledialog.askfloat("Agregar Resorte", "Constante k (N/m):", parent=self.parent)
        
        if k_valor is not None:
            posicion = simpledialog.askfloat("Agregar Resorte", 
                                           "Posición de conexión (0-1):", 
                                           parent=self.parent,
                                           minvalue=0, maxvalue=1)
            
            if posicion is not None:
                longitud = simpledialog.askfloat("Agregar Resorte", 
                                               "Longitud del resorte (m):", 
                                               parent=self.parent,
                                               minvalue=0.1)
                
                if longitud is not None:
                    # Añadir el resorte al sistema
                    componente = ComponenteMasaResorte("resorte", {"k": k_valor, "posicion": posicion, "longitud": longitud})
                    self.componentes_adicionales.append(componente)
                    
                    # Actualizar visualización y ecuación
                    self._actualizar_sistema_visual()
                    self._actualizar_ecuacion()
                    messagebox.showinfo("Información", f"Resorte con k={k_valor} N/m añadido en posición {posicion}")

    def _actualizar_sistema_visual(self):
        """Actualiza la visualización del sistema con todos sus componentes"""
        # Limpiar visualización actual
        self.ax.clear()
        self.ax.axis('off')
        
        # Obtener el valor de la masa
        try:
            masa_valor = float(self.entradas["masa"].get())
        except (ValueError, KeyError):
            masa_valor = 1.0
        
        # Recrear elementos visuales
        if self.modo == "vertical":
            self.punto_fijo = (0.5, 0.8)
            self.punto_masa = (0.5, 0.5)
            
            # Añadir base fija
            self.ax.add_patch(patches.Rectangle((0.4, 0.8), 0.2, 0.05, color='brown'))
            
            # Añadir masa principal (radio proporcional a la masa)
            radio_masa = 0.05 * np.power(masa_valor/1.0, 1/3)  # Escala cúbica para el radio
            self.masa_obj = self.ax.add_patch(plt.Circle(self.punto_masa, radio_masa, color='blue'))
            
            # Añadir resorte
            resorte_puntos = self._crear_puntos_resorte(self.punto_fijo, self.punto_masa)
            self.resorte_obj, = self.ax.plot(resorte_puntos[0], resorte_puntos[1], 'k-', lw=2)
        else:
            self.punto_fijo = (0.2, 0.5)
            self.punto_masa = (0.5, 0.5)
            
            # Añadir base fija
            self.ax.add_patch(patches.Rectangle((0.15, 0.4), 0.05, 0.2, color='brown'))
            
            # Añadir masa principal
            radio_masa = 0.05 * np.power(masa_valor/1.0, 1/3)
            self.masa_obj = self.ax.add_patch(plt.Circle(self.punto_masa, radio_masa, color='blue'))
            
            # Añadir resorte
            resorte_puntos = self._crear_puntos_resorte(self.punto_fijo, self.punto_masa)
            self.resorte_obj, = self.ax.plot(resorte_puntos[0], resorte_puntos[1], 'k-', lw=2)
        
        # Añadir componentes adicionales
        for componente in self.componentes_adicionales:
            if componente.tipo == "masa":
                # Añadir masa adicional
                radio = 0.05 * np.power(componente.datos["valor"]/1.0, 1/3)
                pos = componente.datos["posicion"]
                
                if self.modo == "vertical":
                    pos_masa = (0.5, pos)
                else:
                    pos_masa = (pos, 0.5)
                
                self.ax.add_patch(plt.Circle(pos_masa, radio, color='red'))
            
            elif componente.tipo == "resorte":
                # Añadir resorte adicional
                k_valor = componente.datos["k"]
                pos = componente.datos["posicion"]
                longitud = componente.datos["longitud"]
                
                # Ajustar grosor según k
                grosor = 1 + 0.1 * (k_valor/10.0)
                
                if self.modo == "vertical":
                    punto_inicio = (0.5, pos)
                    punto_fin = (0.5 + longitud/2, pos)
                else:
                    punto_inicio = (pos, 0.5)
                    punto_fin = (pos, 0.5 - longitud/2)
                
                resorte_puntos = self._crear_puntos_resorte(punto_inicio, punto_fin)
                self.ax.plot(resorte_puntos[0], resorte_puntos[1], 'r-', lw=grosor)
        
        # Fijar límites para que no se pierdan elementos
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        # Actualizar canvas
        self.canvas.draw()

    def _actualizar_ecuacion(self):
        """Actualiza la ecuación diferencial según los parámetros actuales"""
        try:
            # Obtener valores de los parámetros
            m = float(self.entradas["masa"].get())
            k = float(self.entradas["k"].get())
            c = float(self.entradas["amortiguamiento"].get())
            
            # Crear la ecuación
            if self.modo == "vertical":
                g = float(self.entradas["gravedad"].get())
                ecuacion = f"m·d²x/dt² + c·dx/dt + k·x = mg\n\n{m}·d²x/dt² + {c}·dx/dt + {k}·x = {m*g:.2f}"
            else:
                ecuacion = f"m·d²x/dt² + c·dx/dt + k·x = 0\n\n{m}·d²x/dt² + {c}·dx/dt + {k}·x = 0"
            
            # Añadir componentes adicionales a la ecuación
            if self.componentes_adicionales:
                ecuacion += "\n\nComponentes adicionales:"
                for idx, comp in enumerate(self.componentes_adicionales):
                    if comp.tipo == "masa":
                        ecuacion += f"\nMasa {idx+1}: {comp.datos['valor']} kg"
                    elif comp.tipo == "resorte":
                        ecuacion += f"\nResorte {idx+1}: k={comp.datos['k']} N/m"
            
            # Actualizar etiqueta
            self.ecuacion_label.config(text=ecuacion)
        except Exception as e:
            self.ecuacion_label.config(text=f"Error: {str(e)}")

    def get_datos(self):
        """Obtiene todos los datos de entrada como valores numéricos"""
        datos = {}
        for key, entry in self.entradas.items():
            try:
                datos[key] = float(entry.get())
            except ValueError:
                datos[key] = 0.0  # Valor por defecto si hay error
        return datos

    def iniciar_simulacion(self):
        """Inicia la simulación del sistema"""
        # Detener simulación anterior si existe
        if self.animacion:
            self.animacion.event_source.stop()
            if self.temporizador_id:
                self.parent.after_cancel(self.temporizador_id)
        
        # Obtener parámetros
        datos = self.get_datos()
        
        m = datos.get("masa", 1.0)
        k = datos.get("k", 10.0)
        c = datos.get("amortiguamiento", 0.5)
        g = datos.get("gravedad", 9.8) if self.modo == "vertical" else 0.0
        
        # Actualizar visualización según los parámetros actuales
        self._actualizar_sistema_visual()
        
        # Reiniciar temporizador
        self.tiempo_inicio = time.time()
        self._actualizar_temporizador()
        
        # Función para el sistema de ecuaciones diferenciales
        def sistema(t, y):
            x, v = y
            # Sumar fuerzas de componentes adicionales
            fuerza_adicional = 0
            for comp in self.componentes_adicionales:
                if comp.tipo == "resorte":
                    # Fuerza del resorte adicional F = -k(x-x0)
                    k_extra = comp.datos["k"]
                    pos = comp.datos["posicion"]
                    fuerza_adicional -= k_extra * (x - pos)
            
            if self.modo == "vertical":
                dvdt = (m * g - c * v - k * x + fuerza_adicional) / m
            else:
                dvdt = (-c * v - k * x + fuerza_adicional) / m
                
            return [v, dvdt]
        
        # Condiciones iniciales y resolución
        y0 = [0.1, 0.0]  # Desplazamiento inicial y velocidad inicial
        t_span = (0, 20)
        t_eval = np.linspace(t_span[0], t_span[1], 500)
        
        # Resolver el sistema de ecuaciones diferenciales
        sol = solve_ivp(sistema, t_span, y0, t_eval=t_eval)
        
        tiempos = sol.t
        posiciones = sol.y[0]
        
        # Función de actualización para la animación
        def actualizar(i):
            pos = posiciones[i]
            
            # Actualizar la posición del objeto según el modo
            if self.modo == "vertical":
                # En modo vertical, el desplazamiento afecta la coordenada y
                nueva_pos = (0.5, self.punto_masa[1] - pos)
                self.masa_obj.set_center(nueva_pos)
                
                # Actualizar el resorte
                resorte_pts = self._crear_puntos_resorte(self.punto_fijo, nueva_pos)
                self.resorte_obj.set_data(resorte_pts[0], resorte_pts[1])
            else:
                # En modo horizontal, el desplazamiento afecta la coordenada x
                nueva_pos = (self.punto_masa[0] + pos, 0.5)
                self.masa_obj.set_center(nueva_pos)
                
                # Actualizar el resorte
                resorte_pts = self._crear_puntos_resorte(self.punto_fijo, nueva_pos)
                self.resorte_obj.set_data(resorte_pts[0], resorte_pts[1])
            
            return self.masa_obj, self.resorte_obj
        
        # Crear la animación
        self.animacion = FuncAnimation(
            self.fig, actualizar, frames=len(tiempos), 
            interval=20, blit=True, repeat=True
        )
        
        # Obligar a redibujar el canvas
        self.canvas.draw()

    def detener_simulacion(self):
        """Detiene la simulación actual"""
        if self.animacion:
            self.animacion.event_source.stop()
        
        if self.temporizador_id:
            self.parent.after_cancel(self.temporizador_id)

    def reanudar_simulacion(self):
        """Reanuda la simulación si está detenida"""
        if self.animacion:
            self.animacion.event_source.start()
            self._actualizar_temporizador()

    def _actualizar_temporizador(self):
        """Actualiza la etiqueta del temporizador"""
        tiempo_actual = time.time() - self.tiempo_inicio
        self.temporizador_label.config(text=f"Tiempo: {tiempo_actual:.2f} s")
        self.temporizador_id = self.parent.after(50, self._actualizar_temporizador)

    def _crear_puntos_resorte(self, inicio, fin):
        """Crea los puntos para dibujar un resorte entre dos puntos"""
        num_vueltas = 10
        puntos_por_vuelta = 10
        total_puntos = num_vueltas * puntos_por_vuelta
        
        if self.modo == "vertical":
            # Para resorte vertical
            x = np.zeros(total_puntos) + inicio[0]
            y = np.linspace(inicio[1], fin[1], total_puntos)
            
            # Añadir zigzag para simular el resorte
            amplitud = 0.03
            for i in range(total_puntos):
                if i % puntos_por_vuelta < puntos_por_vuelta/2:
                    x[i] += amplitud
                else:
                    x[i] -= amplitud
        else:
            # Para resorte horizontal
            x = np.linspace(inicio[0], fin[0], total_puntos)
            y = np.zeros(total_puntos) + inicio[1]
            
            # Añadir zigzag para simular el resorte
            amplitud = 0.03
            for i in range(total_puntos):
                if i % puntos_por_vuelta < puntos_por_vuelta/2:
                    y[i] += amplitud
                else:
                    y[i] -= amplitud
                    
        return x, y


class SimuladorInterfaz:
    """Clase principal que gestiona la interfaz del simulador"""
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador Masa-Resorte")
        self.root.geometry("1200x700")  # Ventana más grande para acomodar ambos sistemas
        self.sistema1 = None
        self.sistema2 = None
        
        # Crear interfaz de selección inicial
        self._crear_pantalla_inicio()

    def _crear_pantalla_inicio(self):
        """Crea la pantalla inicial con opciones de modo"""
        inicio_frame = ttk.Frame(self.root, padding=20)
        inicio_frame.pack(expand=True)

        ttk.Label(inicio_frame, text="Simulación del sistema masa-resorte", font=("Arial", 14)).pack(pady=10)

        ttk.Button(inicio_frame, text="Simulación Vertical", command=lambda: self._abrir_simulacion("vertical")).pack(pady=5)
        ttk.Button(inicio_frame, text="Simulación Horizontal", command=lambda: self._abrir_simulacion("horizontal")).pack(pady=5)

    def _abrir_simulacion(self, modo):
        """Abre la interfaz principal con los dos sistemas"""
        # Limpiar ventana
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.title(f"Simulación {modo.capitalize()}")

        # Frame principal
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Crear los dos sistemas
        self.sistema1 = SistemaMasaResorte(main_frame, modo, 1)
        self.sistema2 = SistemaMasaResorte(main_frame, modo, 2)

        # Panel de botones en la parte inferior
        panel_botones = ttk.Frame(self.root)
        panel_botones.pack(pady=10, fill=tk.X)
        
        # Botones de control para ambos sistemas
        ttk.Button(panel_botones, text="Iniciar Simulaciones", 
                 command=self.iniciar_ambas_simulaciones).pack(side=tk.LEFT, padx=10)
                 
        ttk.Button(panel_botones, text="Detener Simulaciones", 
                 command=self.detener_ambas_simulaciones).pack(side=tk.LEFT, padx=10)
                 
        ttk.Button(panel_botones, text="Reanudar Simulaciones", 
                 command=self.reanudar_ambas_simulaciones).pack(side=tk.LEFT, padx=10)
                 
        ttk.Button(panel_botones, text="Volver al Inicio", 
                 command=self._volver_inicio).pack(side=tk.RIGHT, padx=10)

    def iniciar_ambas_simulaciones(self):
        """Inicia la simulación en ambos sistemas"""
        if self.sistema1:
            self.sistema1.iniciar_simulacion()
        if self.sistema2:
            self.sistema2.iniciar_simulacion()

    def detener_ambas_simulaciones(self):
        """Detiene la simulación en ambos sistemas"""
        if self.sistema1:
            self.sistema1.detener_simulacion()
        if self.sistema2:
            self.sistema2.detener_simulacion()

    def reanudar_ambas_simulaciones(self):
        """Reanuda la simulación en ambos sistemas"""
        if self.sistema1:
            self.sistema1.reanudar_simulacion()
        if self.sistema2:
            self.sistema2.reanudar_simulacion()
            
    def _volver_inicio(self):
        """Vuelve a la pantalla inicial"""
        # Detener simulaciones si están activas
        self.detener_ambas_simulaciones()
        
        # Limpiar ventana y volver al inicio
        for widget in self.root.winfo_children():
            widget.destroy()
        self._crear_pantalla_inicio()