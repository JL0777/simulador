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
import math


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

        # Fijar límites más amplios para que no se pierdan los objetos durante la oscilación
        if self.modo == "vertical":
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1.5)  # Límite vertical ampliado para movimiento vertical
        else:
            self.ax.set_xlim(0, 1.5)  # Límite horizontal ampliado para movimiento horizontal
            self.ax.set_ylim(0, 1)
        
        # Fijar límites para que no se pierdan elementos
        if self.modo == "vertical":
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1.8)  # Ajustado para acomodar el sistema más arriba
        else:
            self.ax.set_xlim(0, 1.5)
            self.ax.set_ylim(0, 1)
        
        # Crear elementos visuales según el modo
        # Crear elementos visuales según el modo
        if self.modo == "vertical":
            self.punto_fijo = (0.5, 1.2)  # Punto fijo en la parte superior (más arriba)
            self.punto_masa = (0.5, 0.8)  # Punto inicial de la masa (más arriba)
            # Añadir base fija (soporte)
            self.ax.add_patch(patches.Rectangle((0.4, 1.2), 0.2, 0.05, color='brown'))
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

        # Botón para mostrar gráficas
        ttk.Button(visualizacion_frame, text="Ver Gráficas", 
        command=self.mostrar_graficas).pack(pady=5)

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
        # Recrear elementos visuales
        if self.modo == "vertical":
            self.punto_fijo = (0.5, 1.6)  # Movido más arriba
            self.punto_masa = (0.5, 1.2)  # Movido más arriba
            
            # Añadir base fija
            self.ax.add_patch(patches.Rectangle((0.4, 1.6), 0.2, 0.05, color='brown'))
            
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
        if self.modo == "vertical":
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 2.0)  # Ampliado para movimiento vertical
        else:
            self.ax.set_xlim(0, 1.5)  # Ampliado para movimiento horizontal
            self.ax.set_ylim(0, 1)

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
    
    def mostrar_graficas(self):
        """Muestra una ventana emergente con 5 gráficas relacionadas con el sistema"""
        # Obtener parámetros actuales
        datos = self.get_datos()
        
        m = datos.get("masa", 1.0)
        k = datos.get("k", 10.0)
        c = datos.get("amortiguamiento", 0.5)
        g = datos.get("gravedad", 9.8) if self.modo == "vertical" else 0.0
        
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
        t_span = (0, 10)
        t_eval = np.linspace(t_span[0], t_span[1], 500)
        
        # Resolver el sistema de ecuaciones diferenciales
        sol = solve_ivp(sistema, t_span, y0, t_eval=t_eval)
        
        tiempos = sol.t
        posiciones = sol.y[0]
        velocidades = sol.y[1]
        
        # Calcular aceleración, energía cinética y potencial
        aceleraciones = np.zeros_like(tiempos)
        energia_cinetica = np.zeros_like(tiempos)
        energia_potencial = np.zeros_like(tiempos)
        energia_total = np.zeros_like(tiempos)
        
        for i in range(len(tiempos)):
            # Calcular aceleración: a = F/m = (-kx - cv + mg)/m
            if self.modo == "vertical":
                aceleraciones[i] = (m * g - k * posiciones[i] - c * velocidades[i]) / m
            else:
                aceleraciones[i] = (-k * posiciones[i] - c * velocidades[i]) / m
            
            # Energía cinética: Ec = 0.5 * m * v²
            energia_cinetica[i] = 0.5 * m * velocidades[i]**2
            
            # Energía potencial: Ep = 0.5 * k * x²
            energia_potencial[i] = 0.5 * k * posiciones[i]**2
            
            # Energía total: Et = Ec + Ep
            energia_total[i] = energia_cinetica[i] + energia_potencial[i]
        
        # Crear ventana emergente para las gráficas
        graficas_ventana = tk.Toplevel(self.parent)
        graficas_ventana.title(f"Gráficas Sistema {self.numero} ({self.modo})")
        graficas_ventana.geometry("800x600")
        
        # Crear figura con subplots
        fig = Figure(figsize=(10, 8))
        
        # Gráfica 1: Posición vs tiempo
        ax1 = fig.add_subplot(321)
        ax1.plot(tiempos, posiciones, 'b-')
        ax1.set_title('Posición vs Tiempo')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Posición (m)')
        ax1.grid(True)
        
        # Gráfica 2: Velocidad vs tiempo
        ax2 = fig.add_subplot(322)
        ax2.plot(tiempos, velocidades, 'g-')
        ax2.set_title('Velocidad vs Tiempo')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Velocidad (m/s)')
        ax2.grid(True)
        
        # Gráfica 3: Aceleración vs tiempo
        ax3 = fig.add_subplot(323)
        ax3.plot(tiempos, aceleraciones, 'r-')
        ax3.set_title('Aceleración vs Tiempo')
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Aceleración (m/s²)')
        ax3.grid(True)
        
        # Gráfica 4: Energía vs tiempo
        ax4 = fig.add_subplot(324)
        ax4.plot(tiempos, energia_cinetica, 'g-', label='E. Cinética')
        ax4.plot(tiempos, energia_potencial, 'b-', label='E. Potencial')
        ax4.plot(tiempos, energia_total, 'r-', label='E. Total')
        ax4.set_title('Energía vs Tiempo')
        ax4.set_xlabel('Tiempo (s)')
        ax4.set_ylabel('Energía (J)')
        ax4.legend()
        ax4.grid(True)
        
        # Gráfica 5: Diagrama de fase (posición vs velocidad)
        ax5 = fig.add_subplot(325)
        ax5.plot(posiciones, velocidades, 'k-')
        ax5.set_title('Diagrama de Fase')
        ax5.set_xlabel('Posición (m)')
        ax5.set_ylabel('Velocidad (m/s)')
        ax5.grid(True)
        
        # Ajustar layout
        fig.tight_layout()
        
        # Añadir canvas a la ventana
        canvas = FigureCanvasTkAgg(fig, master=graficas_ventana)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Botón para cerrar ventana
        ttk.Button(graficas_ventana, text="Cerrar", 
                command=graficas_ventana.destroy).pack(pady=5)


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
        """Crea la pantalla inicial con representaciones gráficas en lugar de botones simples"""
        # Configurar el estilo
        self.root.configure(bg="#f0f0f5")
        
        # Marco principal con efecto de relieve
        inicio_frame = ttk.Frame(self.root, padding=30, style="Main.TFrame")
        inicio_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Estilo personalizado para el marco
        style = ttk.Style()
        style.configure("Main.TFrame", background="#f0f0f5")
        
        # Título mejorado con un subtítulo
        titulo_frame = ttk.Frame(inicio_frame, style="Main.TFrame")
        titulo_frame.pack(pady=15)
        
        ttk.Label(
            titulo_frame, 
            text="SISTEMA MASA-RESORTE", 
            font=("Helvetica", 20, "bold"),
            foreground="#2c3e50",
            background="#f0f0f5"
        ).pack()
        
        ttk.Label(
            titulo_frame, 
            text="Simulación interactiva de oscilaciones armónicas", 
            font=("Helvetica", 12, "italic"),
            foreground="#34495e",
            background="#f0f0f5"
        ).pack(pady=5)
        
        # Separador decorativo
        separator = ttk.Separator(inicio_frame, orient="horizontal")
        separator.pack(fill="x", pady=15)
        
        # Contenedor para las tres opciones
        opciones_frame = ttk.Frame(inicio_frame, style="Main.TFrame")
        opciones_frame.pack(pady=20)
        opciones_frame.columnconfigure(0, weight=1)
        opciones_frame.columnconfigure(1, weight=1)
        opciones_frame.columnconfigure(2, weight=1)
        
        # Función para crear efecto hover en los marcos
        def on_enter(e, frame):
            frame.config(background="#d1e0e0")
            
        def on_leave(e, frame):
            frame.config(background="#e6f0f0")
        
        # Opción 1: Simulación Vertical
        vertical_frame = tk.Frame(opciones_frame, bg="#e6f0f0", width=180, height=220, bd=1, relief=tk.RAISED)
        vertical_frame.grid(row=0, column=0, padx=15)
        vertical_frame.pack_propagate(False)
        
        tk.Label(vertical_frame, text="Simulación Vertical", font=("Helvetica", 12, "bold"), bg="#e6f0f0").pack(pady=(15, 10))
        
        # Canvas para dibujar el sistema vertical
        canvas_v = tk.Canvas(vertical_frame, width=100, height=120, bg="#e6f0f0", highlightthickness=0)
        canvas_v.pack()
        
        # Dibujar el resorte vertical y la masa
        # Punto de fijación superior
        canvas_v.create_rectangle(45, 10, 55, 20, fill="#555555")
        # Resorte (zigzag)
        for i in range(8):
            if i % 2 == 0:
                canvas_v.create_line(50, 20 + i*10, 40, 25 + i*10, width=2)
            else:
                canvas_v.create_line(40, 25 + i*10, 60, 25 + i*10, width=2)
                canvas_v.create_line(60, 25 + i*10, 50, 30 + i*10, width=2)
        # Masa (círculo)
        canvas_v.create_oval(35, 90, 65, 120, fill="#3498db", outline="#2980b9")
        
        # Hacer que todo el frame sea clickeable
        vertical_frame.bind("<Button-1>", lambda e: self._abrir_simulacion("vertical"))
        vertical_frame.bind("<Enter>", lambda e: on_enter(e, vertical_frame))
        vertical_frame.bind("<Leave>", lambda e: on_leave(e, vertical_frame))
        
        # Opción 2: Simulación Horizontal
        horizontal_frame = tk.Frame(opciones_frame, bg="#e6f0f0", width=180, height=220, bd=1, relief=tk.RAISED)
        horizontal_frame.grid(row=0, column=1, padx=15)
        horizontal_frame.pack_propagate(False)
        
        tk.Label(horizontal_frame, text="Simulación Horizontal", font=("Helvetica", 12, "bold"), bg="#e6f0f0").pack(pady=(15, 10))
        
        # Canvas para dibujar el sistema horizontal
        canvas_h = tk.Canvas(horizontal_frame, width=140, height=100, bg="#e6f0f0", highlightthickness=0)
        canvas_h.pack(pady=10)
        
        # Dibujar el resorte horizontal y la masa
        # Punto de fijación izquierdo
        canvas_h.create_rectangle(10, 45, 20, 55, fill="#555555")
        # Resorte (zigzag)
        for i in range(6):
            if i % 2 == 0:
                canvas_h.create_line(20 + i*10, 50, 25 + i*10, 40, width=2)
            else:
                canvas_h.create_line(25 + i*10, 40, 25 + i*10, 60, width=2)
                canvas_h.create_line(25 + i*10, 60, 30 + i*10, 50, width=2)
        # Masa (círculo)
        canvas_h.create_oval(80, 30, 110, 70, fill="#e74c3c", outline="#c0392b")
        
        # Hacer que todo el frame sea clickeable
        horizontal_frame.bind("<Button-1>", lambda e: self._abrir_simulacion("horizontal"))
        horizontal_frame.bind("<Enter>", lambda e: on_enter(e, horizontal_frame))
        horizontal_frame.bind("<Leave>", lambda e: on_leave(e, horizontal_frame))
        
        # Opción 3: Teoría
        teoria_frame = tk.Frame(opciones_frame, bg="#e6f0f0", width=180, height=220, bd=1, relief=tk.RAISED)
        teoria_frame.grid(row=0, column=2, padx=15)
        teoria_frame.pack_propagate(False)
        
        tk.Label(teoria_frame, text="Fundamentos Teóricos", font=("Helvetica", 12, "bold"), bg="#e6f0f0").pack(pady=(15, 10))
        
        # Canvas para dibujar algo representativo de la teoría
        canvas_t = tk.Canvas(teoria_frame, width=120, height=100, bg="#e6f0f0", highlightthickness=0)
        canvas_t.pack(pady=10)
        
        # Dibujar representación de fórmulas/gráfico
        # Función sinusoidal
        for x in range(120):
            y = 50 + 30 * math.sin(x/15)
            if x > 0:
                canvas_t.create_line(x-1, prev_y, x, y, fill="#27ae60", width=2)
            prev_y = y
        
        # Fórmula
        canvas_t.create_text(60, 80, text="F = -kx", font=("Helvetica", 14, "bold"), fill="#2c3e50")
        
        # Hacer que todo el frame sea clickeable
        teoria_frame.bind("<Button-1>", lambda e: self._mostrar_teoria())
        teoria_frame.bind("<Enter>", lambda e: on_enter(e, teoria_frame))
        teoria_frame.bind("<Leave>", lambda e: on_leave(e, teoria_frame))
        
        # Créditos en la parte inferior
        ttk.Label(
            inicio_frame, 
            text="© Simulador Educativo de Física", 
            font=("Helvetica", 8),
            foreground="#7f8c8d",
            background="#f0f0f5"
        ).pack(side="bottom", pady=10)

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
    
    def _mostrar_teoria(self):
        """Muestra una ventana emergente mejorada con la teoría del sistema masa-resorte"""
        teoria_ventana = tk.Toplevel(self.root)
        teoria_ventana.title("Teoría del Sistema Masa-Resorte")
        # Configuración para pantalla completa en portátil
        teoria_ventana.geometry("1200x800")
        
        # Definir estilos para mejorar la apariencia
        estilo = ttk.Style()
        estilo.configure("TitleLabel.TLabel", font=("Arial", 20, "bold"), foreground="#2C3E50")
        estilo.configure("SubtitleLabel.TLabel", font=("Arial", 14, "bold"), foreground="#34495E")
        estilo.configure("NormalText.TLabel", font=("Arial", 11), foreground="#2C3E50")
        estilo.configure("Formula.TLabel", font=("Arial", 12, "bold"), background="#F8F9F9", padding=10)
        estilo.configure("Section.TFrame", background="#ECF0F1", padding=10)
        
        # Crear un widget con scroll
        main_frame = ttk.Frame(teoria_ventana)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas con scrollbar
        canvas = tk.Canvas(main_frame, bg="#FFFFFF")
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="Section.TFrame")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Hacer que el scroll responda a la rueda del ratón
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=1160)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Título principal
        ttk.Label(scrollable_frame, text="Sistema Masa-Resorte: Teoría y Aplicaciones", 
                style="TitleLabel.TLabel").pack(pady=20, padx=10)
        
        # Función para crear secciones con título y contenido
        def crear_seccion(titulo, contenido_frame):
            seccion_frame = ttk.LabelFrame(scrollable_frame, text=titulo, padding=10)
            seccion_frame.pack(fill=tk.X, padx=20, pady=10, ipady=5)
            contenido_frame(seccion_frame)
            return seccion_frame
        
        # Función para crear fórmulas estilizadas
        def crear_formula(parent, texto, descripcion=None):
            formula_frame = ttk.Frame(parent)
            formula_frame.pack(fill=tk.X, pady=5)
            
            formula_label = ttk.Label(formula_frame, text=texto, style="Formula.TLabel")
            formula_label.pack(pady=5)
            
            if descripcion:
                desc_label = ttk.Label(formula_frame, text=descripcion, style="NormalText.TLabel")
                desc_label.pack(pady=2)
                
            return formula_frame
        
        # ---- SECCIÓN 1: INTRODUCCIÓN ----
        def intro_content(frame):
            ttk.Label(frame, text="Los sistemas masa-resorte son fundamentales en la física y la ingeniería. "
                    "Representan un modelo matemático que describe el movimiento oscilatorio cuando una masa "
                    "está sujeta a una fuerza restauradora proporcional al desplazamiento (Ley de Hooke).",
                    style="NormalText.TLabel", wraplength=1100).pack(pady=5)
                    
            # Añadir una ilustración básica del sistema masa-resorte
            fig_intro = Figure(figsize=(10, 3))
            ax_intro = fig_intro.add_subplot(111)
            ax_intro.set_xlim(0, 10)
            ax_intro.set_ylim(0, 2)
            ax_intro.axis('off')
            
            # Dibujar un resorte y una masa
            x_spring = np.linspace(1, 6, 100)
            y_spring = 0.5 * np.sin(x_spring * 5) + 1
            ax_intro.plot(x_spring, y_spring, 'k-', linewidth=2)
            ax_intro.add_patch(plt.Rectangle((0.5, 0.7), 0.5, 0.6, color='brown'))
            ax_intro.add_patch(plt.Circle((7, 1), 0.5, color='blue'))
            ax_intro.plot([6, 7], [1, 1], 'k-', linewidth=2)
            
            # Añadir etiquetas
            ax_intro.text(3.5, 0.5, "Resorte (k)", ha='center')
            ax_intro.text(7, 1, "m", ha='center', va='center', color='white')
            ax_intro.text(1.5, 1.8, "Sistema Masa-Resorte", ha='center', fontsize=14, fontweight='bold')
            
            fig_intro.tight_layout()
            
            # Añadir la figura al frame
            canvas_intro = FigureCanvasTkAgg(fig_intro, master=frame)
            canvas_intro.draw()
            canvas_intro.get_tk_widget().pack(pady=10)
            
        crear_seccion("Introducción", intro_content)

        # ---- SECCIÓN 2: ECUACIÓN DEL MOVIMIENTO ----
        def ecuacion_content(frame):
            ttk.Label(frame, text="La ecuación diferencial que describe el movimiento de un sistema masa-resorte es:", 
                      style="NormalText.TLabel").pack(pady=5, anchor="w")
            
            crear_formula(frame, "m·(d²x/dt²) + c·(dx/dt) + k·x = F(t)", 
                         "Ecuación general del movimiento para un sistema masa-resorte")
            
            # Explicación de los términos
            terminos_frame = ttk.LabelFrame(frame, text="Términos de la Ecuación")
            terminos_frame.pack(fill=tk.X, pady=10, padx=20)
            
            terminos = [
                ("m·(d²x/dt²)", "Término de inercia: representa la fuerza necesaria para acelerar la masa"),
                ("c·(dx/dt)", "Término de amortiguamiento: representa la fuerza de resistencia proporcional a la velocidad"),
                ("k·x", "Término elástico: representa la fuerza restauradora del resorte (Ley de Hooke)"),
                ("F(t)", "Fuerza externa aplicada al sistema (puede variar con el tiempo)")
            ]
            
            for i, (term, desc) in enumerate(terminos):
                term_frame = ttk.Frame(terminos_frame)
                term_frame.pack(fill=tk.X, pady=5)
                ttk.Label(term_frame, text=term, width=12, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
                ttk.Label(term_frame, text=desc, style="NormalText.TLabel", wraplength=1000).pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            # Casos especiales
            ttk.Label(frame, text="Dependiendo de los parámetros y condiciones, la ecuación puede simplificarse:", 
                      style="SubtitleLabel.TLabel").pack(pady=10, anchor="w", padx=20)
            
            casos_frame = ttk.Frame(frame)
            casos_frame.pack(fill=tk.X, pady=10, padx=20)
            
            # Caso 1: Movimiento libre no amortiguado
            caso1_frame = ttk.LabelFrame(casos_frame, text="Movimiento Libre No Amortiguado")
            caso1_frame.pack(fill=tk.X, pady=5)
            
            crear_formula(caso1_frame, "m·(d²x/dt²) + k·x = 0", 
                         "Sin amortiguamiento ni fuerzas externas (c = 0, F(t) = 0)")
            crear_formula(caso1_frame, "Solución: x(t) = A·cos(ω₀t + φ)", 
                         "Movimiento armónico simple con amplitud A y fase inicial φ")
            
            # Caso 2: Movimiento libre amortiguado
            caso2_frame = ttk.LabelFrame(casos_frame, text="Movimiento Libre Amortiguado")
            caso2_frame.pack(fill=tk.X, pady=5)
            
            crear_formula(caso2_frame, "m·(d²x/dt²) + c·(dx/dt) + k·x = 0", 
                         "Con amortiguamiento pero sin fuerzas externas (F(t) = 0)")
            crear_formula(caso2_frame, "Solución: x(t) = A·e^(-ζω₀t)·cos(ω₀√(1-ζ²)·t + φ)", 
                         "Para el caso subamortiguado (ζ < 1)")
            
            # Caso 3: Movimiento forzado
            caso3_frame = ttk.LabelFrame(casos_frame, text="Movimiento Forzado")
            caso3_frame.pack(fill=tk.X, pady=5)
            
            crear_formula(caso3_frame, "m·(d²x/dt²) + c·(dx/dt) + k·x = F₀·cos(ωt)", 
                         "Con una fuerza armónica externa de amplitud F₀ y frecuencia ω")
            crear_formula(caso3_frame, "Solución en estado estable: x(t) = A·cos(ωt - φ)", 
                         "Donde A y φ dependen de la frecuencia de la fuerza externa")
            
            # Visualización interactiva de las soluciones
            ttk.Label(frame, text="Visualización de diferentes soluciones:", 
                     style="SubtitleLabel.TLabel").pack(pady=10)
            
            # Marco para visualización
            vis_eq_frame = ttk.Frame(frame)
            vis_eq_frame.pack(pady=10, fill=tk.X)
            
            # Figura para la visualización
            fig_eq = Figure(figsize=(10, 6))
            ax_eq = fig_eq.add_subplot(111)
            
            # Configuración inicial
            ax_eq.set_xlabel('Tiempo (s)')
            ax_eq.set_ylabel('Posición (x)')
            ax_eq.set_title('Simulación de Movimiento del Sistema Masa-Resorte')
            ax_eq.grid(True)
            
            # Líneas para los diferentes tipos de movimiento
            t_eq = np.linspace(0, 10, 1000)
            line_libre, = ax_eq.plot([], [], 'b-', linewidth=2, label='No Amortiguado')
            line_amort, = ax_eq.plot([], [], 'r-', linewidth=2, label='Amortiguado')
            line_forz, = ax_eq.plot([], [], 'g-', linewidth=2, label='Forzado')
            
            ax_eq.legend(loc='upper right')
            
            # Canvas para la figura
            canvas_eq = FigureCanvasTkAgg(fig_eq, master=vis_eq_frame)
            canvas_eq.draw()
            canvas_eq.get_tk_widget().pack(fill=tk.X)
            
            # Controles para la visualización
            ctrl_eq_frame = ttk.Frame(vis_eq_frame)
            ctrl_eq_frame.pack(pady=10)
            
            # Parámetros del sistema
            ttk.Label(ctrl_eq_frame, text="Masa (m):").grid(row=0, column=0, padx=5, pady=5)
            masa_eq_var = tk.DoubleVar(value=1.0)
            masa_eq_entry = ttk.Entry(ctrl_eq_frame, width=6, textvariable=masa_eq_var)
            masa_eq_entry.grid(row=0, column=1, padx=5, pady=5)
            
            ttk.Label(ctrl_eq_frame, text="Constante k:").grid(row=0, column=2, padx=5, pady=5)
            k_eq_var = tk.DoubleVar(value=10.0)
            k_eq_entry = ttk.Entry(ctrl_eq_frame, width=6, textvariable=k_eq_var)
            k_eq_entry.grid(row=0, column=3, padx=5, pady=5)
            
            ttk.Label(ctrl_eq_frame, text="Amortiguamiento (c):").grid(row=0, column=4, padx=5, pady=5)
            c_eq_var = tk.DoubleVar(value=0.5)
            c_eq_entry = ttk.Entry(ctrl_eq_frame, width=6, textvariable=c_eq_var)
            c_eq_entry.grid(row=0, column=5, padx=5, pady=5)
            
            # Parámetros de la fuerza externa
            ttk.Label(ctrl_eq_frame, text="Amplitud Fuerza:").grid(row=1, column=0, padx=5, pady=5)
            f0_eq_var = tk.DoubleVar(value=1.0)
            f0_eq_entry = ttk.Entry(ctrl_eq_frame, width=6, textvariable=f0_eq_var)
            f0_eq_entry.grid(row=1, column=1, padx=5, pady=5)
            
            ttk.Label(ctrl_eq_frame, text="Frecuencia Fuerza:").grid(row=1, column=2, padx=5, pady=5)
            omega_eq_var = tk.DoubleVar(value=3.0)
            omega_eq_entry = ttk.Entry(ctrl_eq_frame, width=6, textvariable=omega_eq_var)
            omega_eq_entry.grid(row=1, column=3, padx=5, pady=5)
            
            # Condiciones iniciales
            ttk.Label(ctrl_eq_frame, text="Posición inicial:").grid(row=1, column=4, padx=5, pady=5)
            x0_eq_var = tk.DoubleVar(value=1.0)
            x0_eq_entry = ttk.Entry(ctrl_eq_frame, width=6, textvariable=x0_eq_var)
            x0_eq_entry.grid(row=1, column=5, padx=5, pady=5)
            
            ttk.Label(ctrl_eq_frame, text="Velocidad inicial:").grid(row=1, column=6, padx=5, pady=5)
            v0_eq_var = tk.DoubleVar(value=0.0)
            v0_eq_entry = ttk.Entry(ctrl_eq_frame, width=6, textvariable=v0_eq_var)
            v0_eq_entry.grid(row=1, column=7, padx=5, pady=5)
            
            # Botón para actualizar la simulación
            actualizar_eq_btn = ttk.Button(ctrl_eq_frame, text="Simular")
            actualizar_eq_btn.grid(row=0, column=6, rowspan=1, columnspan=2, padx=20, pady=5)
            
            # Función para actualizar la simulación
            def actualizar_ecuacion():
                try:
                    # Obtener parámetros
                    m = float(masa_eq_var.get())
                    k = float(k_eq_var.get())
                    c = float(c_eq_var.get())
                    f0 = float(f0_eq_var.get())
                    omega_f = float(omega_eq_var.get())
                    x0 = float(x0_eq_var.get())
                    v0 = float(v0_eq_var.get())
                    
                    # Validar valores positivos
                    if m <= 0 or k <= 0 or c < 0:
                        messagebox.showerror("Error", "La masa y k deben ser positivos, y c no puede ser negativo")
                        return
                    
                    # Frecuencia natural
                    omega0 = np.sqrt(k/m)
                    
                    # Factor de amortiguamiento
                    zeta = c/(2*m*omega0) if omega0 > 0 else 0
                    
                    # Movimiento no amortiguado (c = 0)
                    x_libre = x0 * np.cos(omega0*t_eq) + (v0/omega0) * np.sin(omega0*t_eq)
                    
                    # Movimiento amortiguado (depende del valor de zeta)
                    if zeta < 1:  # Subamortiguado
                        omega_d = omega0 * np.sqrt(1 - zeta**2)
                        A = np.sqrt(x0**2 + ((v0 + zeta*omega0*x0)/omega_d)**2)
                        phi = np.arctan2(v0 + zeta*omega0*x0, x0*omega_d)
                        x_amort = A * np.exp(-zeta*omega0*t_eq) * np.cos(omega_d*t_eq - phi)
                    elif zeta == 1:  # Críticamente amortiguado
                        x_amort = (x0 + (v0 + omega0*x0)*t_eq) * np.exp(-omega0*t_eq)
                    else:  # Sobreamortiguado
                        r1 = -omega0 * (zeta + np.sqrt(zeta**2 - 1))
                        r2 = -omega0 * (zeta - np.sqrt(zeta**2 - 1))
                        C1 = (v0 - r2*x0)/(r1 - r2)
                        C2 = (r1*x0 - v0)/(r1 - r2)
                        x_amort = C1*np.exp(r1*t_eq) + C2*np.exp(r2*t_eq)
                    
                    # Movimiento forzado
                    # Para simplificar, solo mostramos la respuesta en estado estable para el caso forzado
                    if omega_f != omega0:  # Evitar resonancia exacta
                        denominator = np.sqrt((k - m*omega_f**2)**2 + (c*omega_f)**2)
                        A_forzado = f0 / denominator if denominator > 0 else 0
                        phi_forzado = np.arctan2(c*omega_f, k - m*omega_f**2)
                        x_forz = A_forzado * np.cos(omega_f*t_eq - phi_forzado)
                        
                        # Agregar respuesta transitoria para el forzado
                        if zeta < 1:  # Solo para el caso subamortiguado
                            x_trans = x0 * np.exp(-zeta*omega0*t_eq) * np.cos(omega_d*t_eq)
                            # Combinar respuestas transitoria y estable
                            x_forz = x_forz + x_trans * np.exp(-t_eq)  # La transitoria se atenúa más rápido
                    else:
                        # En caso de resonancia, evitar división por cero
                        x_forz = np.zeros_like(t_eq)
                    
                    # Actualizar gráficos
                    line_libre.set_data(t_eq, x_libre)
                    line_amort.set_data(t_eq, x_amort)
                    line_forz.set_data(t_eq, x_forz)
                    
                    # Ajustar ejes
                    ax_eq.relim()
                    ax_eq.autoscale_view()
                    ax_eq.set_ylim(-max(2, 1.5*max(abs(np.max(x_libre)), abs(np.min(x_libre)))), 
                                  max(2, 1.5*max(abs(np.max(x_libre)), abs(np.min(x_libre)))))
                    
                    # Actualizar el título con los parámetros actuales
                    ax_eq.set_title(f'Simulación con m={m:.1f}, k={k:.1f}, c={c:.1f}, ω₀={omega0:.2f} rad/s')
                    
                    # Redibujar
                    canvas_eq.draw()
                    
                except ValueError:
                    messagebox.showerror("Error", "Todos los campos deben contener valores numéricos válidos")
                except Exception as e:
                    messagebox.showerror("Error", f"Error en la simulación: {str(e)}")
            
            # Conectar botón
            actualizar_eq_btn.config(command=actualizar_ecuacion)
            
            # Inicializar simulación con valores predeterminados
            actualizar_ecuacion()
            
            # Ejemplos de aplicación de la ecuación
            ttk.Label(frame, text="Ejemplos de Aplicación:", 
                     style="SubtitleLabel.TLabel").pack(pady=10, anchor="w", padx=20)
            
            ejemplos_frame = ttk.Frame(frame)
            ejemplos_frame.pack(fill=tk.X, pady=10, padx=20)
            
            ejemplo1_frame = ttk.LabelFrame(ejemplos_frame, text="Ejemplo 1: Sistema con Fuerza Variable")
            ejemplo1_frame.pack(fill=tk.X, pady=5)
            
            texto_ejemplo1 = """
            Un sistema masa-resorte con m = 0.5 kg, k = 20 N/m y c = 0.8 N·s/m está sujeto a una fuerza 
            externa que varía según F(t) = 2·cos(4t) N.
            
            • Frecuencia natural: ω₀ = √(k/m) = √(20/0.5) = 6.32 rad/s
            • Factor de amortiguamiento: ζ = c/(2·m·ω₀) = 0.8/(2·0.5·6.32) = 0.127
            • Como ζ < 1, el sistema es subamortiguado
            • La frecuencia de la fuerza (ω = 4 rad/s) es menor que la frecuencia natural (ω₀ = 6.32 rad/s)
            • La amplitud de la respuesta en estado estable será: A = 2/√[(20-0.5·4²)² + (0.8·4)²] = 0.115 m
            """
            
            ttk.Label(ejemplo1_frame, text=texto_ejemplo1, style="NormalText.TLabel", 
                    wraplength=1050).pack(pady=10, padx=10)
            
            ejemplo2_frame = ttk.LabelFrame(ejemplos_frame, text="Ejemplo 2: Análisis de un Sismógrafo")
            ejemplo2_frame.pack(fill=tk.X, pady=5)
            
            texto_ejemplo2 = """
            Un sismógrafo puede modelarse como un sistema masa-resorte amortiguado donde:
            
            • La masa está suspendida y relativamente estacionaria durante un terremoto
            • El marco del instrumento se mueve con el suelo
            • Si m = 1 kg, k = 100 N/m y c = 6 N·s/m:
                - Frecuencia natural: ω₀ = 10 rad/s
                - Factor de amortiguamiento: ζ = 0.3
            • El sistema está diseñado para ser subamortiguado (0 < ζ < 1) para poder registrar las oscilaciones
            • La respuesta del sistema a un impulso sísmico permite determinar la magnitud y características del terremoto
            """
            
            ttk.Label(ejemplo2_frame, text=texto_ejemplo2, style="NormalText.TLabel", 
                    wraplength=1050).pack(pady=10, padx=10)
            
        crear_seccion("Ecuación del Movimiento", ecuacion_content)
        

            
           
        
        # ---- SECCIÓN 3: TIPOS DE AMORTIGUAMIENTO ----
        def amortiguamiento_content(frame):
            ttk.Label(frame, text="El comportamiento del sistema depende del factor de amortiguamiento ζ (zeta):", 
                    style="NormalText.TLabel").pack(pady=5, anchor="w")
            
            crear_formula(frame, "ζ = c / (2√(km))", "Factor de amortiguamiento")
            
            # Crear gráfico comparativo
            fig_am = Figure(figsize=(9, 4))
            ax_am = fig_am.add_subplot(111)
            
            # Generar datos para los tres tipos de amortiguamiento
            t = np.linspace(0, 10, 1000)
            omega = 2.0  # Frecuencia angular
            
            # Subamortiguado (ζ = 0.2)
            y1 = np.exp(-0.2*omega*t) * np.cos(omega*np.sqrt(1-0.2**2)*t)
            
            # Críticamente amortiguado (ζ = 1)
            y2 = np.exp(-omega*t) * (1 + omega*t)
            
            # Sobreamortiguado (ζ = 1.5)
            y3 = np.exp(-omega*t) * (np.cosh(omega*np.sqrt(1.5**2-1)*t) + 
                                    1.5/np.sqrt(1.5**2-1) * np.sinh(omega*np.sqrt(1.5**2-1)*t))
            
            # Graficar los tres casos con mejor estilo
            ax_am.plot(t, y1, 'b-', label='Subamortiguado (ζ = 0.2)', linewidth=2)
            ax_am.plot(t, y2, 'r-', label='Crítico (ζ = 1)', linewidth=2)
            ax_am.plot(t, y3, 'g-', label='Sobreamortiguado (ζ = 1.5)', linewidth=2)
            ax_am.set_title('Comparación de Tipos de Amortiguamiento', fontsize=12, fontweight='bold')
            ax_am.set_xlabel('Tiempo (s)', fontsize=10)
            ax_am.set_ylabel('Desplazamiento', fontsize=10)
            ax_am.legend(loc='upper right')
            ax_am.grid(True, linestyle='--', alpha=0.7)
            fig_am.tight_layout()
            
            # Añadir la gráfica al frame
            canvas_am = FigureCanvasTkAgg(fig_am, master=frame)
            canvas_am.draw()
            canvas_am.get_tk_widget().pack(pady=10)
            
            # Descripción de los tipos
            tipos_frame = ttk.Frame(frame)
            tipos_frame.pack(fill=tk.X, pady=5)
            
            # Subamortiguado
            tipo1_frame = ttk.LabelFrame(tipos_frame, text="Subamortiguado (ζ < 1)")
            tipo1_frame.pack(fill=tk.X, pady=5)
            ttk.Label(tipo1_frame, text="El sistema oscila con amplitud decreciente. Es el caso más común en la mayoría "
                    "de los sistemas mecánicos reales.", style="NormalText.TLabel", 
                    wraplength=1050).pack(pady=5, padx=5)
            
            # Críticamente amortiguado
            tipo2_frame = ttk.LabelFrame(tipos_frame, text="Críticamente amortiguado (ζ = 1)")
            tipo2_frame.pack(fill=tk.X, pady=5)
            ttk.Label(tipo2_frame, text="El sistema regresa al equilibrio sin oscilar, en el menor tiempo posible. "
                    "Este punto de inflexión es importante en el diseño de sistemas que deben estabilizarse rápidamente.", 
                    style="NormalText.TLabel", wraplength=1050).pack(pady=5, padx=5)
            
            # Sobreamortiguado
            tipo3_frame = ttk.LabelFrame(tipos_frame, text="Sobreamortiguado (ζ > 1)")
            tipo3_frame.pack(fill=tk.X, pady=5)
            ttk.Label(tipo3_frame, text="El sistema regresa al equilibrio más lentamente que en el caso crítico, "
                    "sin oscilaciones. Útil cuando se quiere evitar cualquier tipo de oscilación.", 
                    style="NormalText.TLabel", wraplength=1050).pack(pady=5, padx=5)
            
        crear_seccion("Tipos de Amortiguamiento", amortiguamiento_content)
        
        # ---- SECCIÓN 4: ENERGÍA EN EL SISTEMA ----
        def energia_content(frame):
            ttk.Label(frame, text="La energía en un sistema masa-resorte se distribuye entre cinética y potencial:", 
                    style="NormalText.TLabel").pack(pady=5, anchor="w")
            
            crear_formula(frame, "Energía Cinética: Ec = ½·m·v²", 
                        "Energía asociada al movimiento de la masa")
            crear_formula(frame, "Energía Potencial: Ep = ½·k·x²", 
                        "Energía almacenada en el resorte deformado")
            crear_formula(frame, "Energía Total: Et = Ec + Ep", 
                        "En sistemas conservativos, la energía total se mantiene constante")
            
            
        
        # ---- SECCIÓN 5: FRECUENCIA NATURAL ----
        def frecuencia_content(frame):
            ttk.Label(frame, text="La frecuencia natural del sistema depende de la masa y la constante del resorte:", 
                    style="NormalText.TLabel").pack(pady=5, anchor="w")
            
            crear_formula(frame, "Frecuencia angular natural: ω₀ = √(k/m)", 
                        "Velocidad angular de oscilación en radianes por segundo")
            crear_formula(frame, "Frecuencia natural: f₀ = ω₀/(2π) = (1/2π)√(k/m)", 
                        "Número de oscilaciones por segundo (Hz)")
            crear_formula(frame, "Período de oscilación: T = 2π·√(m/k)", 
                        "Tiempo necesario para completar una oscilación")
            
            # Panel interactivo para calcular frecuencias
            calc_frame = ttk.LabelFrame(frame, text="Calculadora de Frecuencia")
            calc_frame.pack(fill=tk.X, pady=10, padx=20)
            
            # Fila de entrada
            entrada_frame = ttk.Frame(calc_frame)
            entrada_frame.pack(fill=tk.X, pady=10, padx=10)
            
            # Masa
            ttk.Label(entrada_frame, text="Masa (kg):").grid(row=0, column=0, padx=5, pady=5)
            masa_var = tk.StringVar(value="1.0")
            masa_entry = ttk.Entry(entrada_frame, width=10, textvariable=masa_var)
            masa_entry.grid(row=0, column=1, padx=5, pady=5)
            
            # Constante k
            ttk.Label(entrada_frame, text="Constante k (N/m):").grid(row=0, column=2, padx=5, pady=5)
            k_var = tk.StringVar(value="10.0")
            k_entry = ttk.Entry(entrada_frame, width=10, textvariable=k_var)
            k_entry.grid(row=0, column=3, padx=5, pady=5)
            
            # Botón calcular
            calcular_btn = ttk.Button(entrada_frame, text="Calcular")
            calcular_btn.grid(row=0, column=4, padx=20, pady=5)
            
            # Marco para resultados
            resultados_frame = ttk.Frame(calc_frame)
            resultados_frame.pack(fill=tk.X, pady=10, padx=10)
            
            # Etiquetas para resultados
            lbl_omega = ttk.Label(resultados_frame, text="Frecuencia angular (ω₀): -- rad/s", 
                                style="NormalText.TLabel")
            lbl_omega.grid(row=0, column=0, padx=10, pady=5, sticky="w")
            
            lbl_freq = ttk.Label(resultados_frame, text="Frecuencia natural (f₀): -- Hz", 
                                style="NormalText.TLabel")
            lbl_freq.grid(row=1, column=0, padx=10, pady=5, sticky="w")
            
            lbl_periodo = ttk.Label(resultados_frame, text="Período (T): -- s", 
                                style="NormalText.TLabel")
            lbl_periodo.grid(row=2, column=0, padx=10, pady=5, sticky="w")
            
            # Función para calcular
            def calcular_frecuencia():
                try:
                    # Obtener valores
                    m = float(masa_var.get())
                    k = float(k_var.get())
                    
                    # Validar valores positivos
                    if m <= 0 or k <= 0:
                        messagebox.showerror("Error", "Los valores deben ser positivos")
                        return
                    
                    # Calcular
                    omega = np.sqrt(k/m)
                    freq = omega/(2*np.pi)
                    periodo = (2*np.pi)/omega
                    
                    # Mostrar resultados
                    lbl_omega.config(text=f"Frecuencia angular (ω₀): {omega:.4f} rad/s")
                    lbl_freq.config(text=f"Frecuencia natural (f₀): {freq:.4f} Hz")
                    lbl_periodo.config(text=f"Período (T): {periodo:.4f} s")
                    
                    # Actualizar gráfico
                    actualizar_grafico_frecuencia(m, k)
                    
                except ValueError:
                    messagebox.showerror("Error", "Ingrese valores numéricos válidos")
            
            # Conectar botón
            calcular_btn.config(command=calcular_frecuencia)
            
            # Visualización gráfica
            ttk.Label(frame, text="Visualización de la dependencia de la frecuencia:", 
                    style="SubtitleLabel.TLabel").pack(pady=10)
            
            # Marco para visualización
            vis_frame = ttk.Frame(frame)
            vis_frame.pack(pady=10, fill=tk.X)
            
            # Crear figura para gráfico
            fig_freq = Figure(figsize=(10, 4))
            ax_freq1 = fig_freq.add_subplot(121)
            ax_freq2 = fig_freq.add_subplot(122)
            
            # Configurar ejes
            ax_freq1.set_title('Efecto de la masa (k constante)')
            ax_freq1.set_xlabel('Masa (kg)')
            ax_freq1.set_ylabel('Frecuencia (Hz)')
            ax_freq1.grid(True)
            
            ax_freq2.set_title('Efecto de k (masa constante)')
            ax_freq2.set_xlabel('Constante k (N/m)')
            ax_freq2.set_ylabel('Frecuencia (Hz)')
            ax_freq2.grid(True)
            
            # Líneas iniciales
            line_freq1, = ax_freq1.plot([], [], 'b-', linewidth=2)
            line_freq2, = ax_freq2.plot([], [], 'r-', linewidth=2)
            
            # Puntos para el valor actual
            punto_freq1, = ax_freq1.plot([], [], 'bo', markersize=8)
            punto_freq2, = ax_freq2.plot([], [], 'ro', markersize=8)
            
            # Canvas para la figura
            canvas_freq = FigureCanvasTkAgg(fig_freq, master=vis_frame)
            canvas_freq.draw()
            canvas_freq.get_tk_widget().pack(fill=tk.X)
            
            # Función para actualizar el gráfico
            def actualizar_grafico_frecuencia(masa, k_const):
                # Calcular valores para diferentes masas (k constante)
                masas = np.linspace(0.1, 5.0, 100)
                freq_masas = (1/(2*np.pi)) * np.sqrt(k_const/masas)
                
                # Calcular valores para diferentes k (masa constante)
                ks = np.linspace(1.0, 50.0, 100)
                freq_ks = (1/(2*np.pi)) * np.sqrt(ks/masa)
                
                # Actualizar líneas
                line_freq1.set_data(masas, freq_masas)
                line_freq2.set_data(ks, freq_ks)
                
                # Actualizar puntos
                punto_freq1.set_data([masa], [(1/(2*np.pi)) * np.sqrt(k_const/masa)])
                punto_freq2.set_data([k_const], [(1/(2*np.pi)) * np.sqrt(k_const/masa)])
                
                # Ajustar escalas
                ax_freq1.relim()
                ax_freq1.autoscale_view()
                ax_freq2.relim()
                ax_freq2.autoscale_view()
                
                # Redibujar
                canvas_freq.draw()
            
            # Inicializar con valores predeterminados
            actualizar_grafico_frecuencia(1.0, 10.0)
            
        crear_seccion("Frecuencia Natural y Resonancia", frecuencia_content)
        
        # ---- SECCIÓN 6: RESONANCIA ----
        def resonancia_content(frame):
            ttk.Label(frame, text="La resonancia ocurre cuando una fuerza externa oscila cerca de la frecuencia natural del sistema:", 
                    style="NormalText.TLabel", wraplength=1100).pack(pady=5, anchor="w")
            
            crear_formula(frame, "Amplitud de respuesta: A(ω) = F₀/√[(k-mω²)² + (cω)²]", 
                        "Amplitud en función de la frecuencia angular de la fuerza externa")
            crear_formula(frame, "Frecuencia de resonancia: ωᵣ = √(k/m - (c²/2m²))", 
                        "Frecuencia donde la amplitud alcanza su máximo valor")
            
            # Demostración visual de resonancia
            ttk.Label(frame, text="Demostración de resonancia:", style="SubtitleLabel.TLabel").pack(pady=10)
            
            # Marco para la demostración
            demo_res_frame = ttk.Frame(frame)
            demo_res_frame.pack(pady=10, fill=tk.X)
            
            # Figura para la demostración
            fig_res = Figure(figsize=(10, 5))
            
            # Gráfico de amplitud vs. frecuencia
            ax_res1 = fig_res.add_subplot(121)
            ax_res1.set_title('Curva de Resonancia')
            ax_res1.set_xlabel('Frecuencia / Frecuencia Natural (ω/ω₀)')
            ax_res1.set_ylabel('Amplitud Normalizada')
            ax_res1.grid(True)
            
            # Gráfico de respuesta temporal
            ax_res2 = fig_res.add_subplot(122)
            ax_res2.set_title('Respuesta en el Tiempo')
            ax_res2.set_xlabel('Tiempo')
            ax_res2.set_ylabel('Amplitud')
            ax_res2.grid(True)
            
            # Generar datos para la curva de resonancia
            omega_rel = np.linspace(0, 2, 200)  # ω/ω₀
            
            # Diferentes factores de amortiguamiento
            zeta_valores = [0.1, 0.3, 0.5, 0.7]
            curvas_res = []
            
            for zeta in zeta_valores:
                # Magnitud de respuesta
                amplitud = 1.0 / np.sqrt((1 - omega_rel**2)**2 + (2*zeta*omega_rel)**2)
                line, = ax_res1.plot(omega_rel, amplitud, linewidth=2, label=f'ζ = {zeta}')
                curvas_res.append(line)
            
            ax_res1.legend(loc='upper right')
            
            # Línea vertical para la frecuencia seleccionada
            freq_sel_line = ax_res1.axvline(x=1.0, color='k', linestyle='--')
            
            # Datos para la respuesta temporal
            t_res = np.linspace(0, 10*np.pi, 500)
            resp_line, = ax_res2.plot([], [], 'b-', linewidth=2)
            
            # Canvas para la figura
            canvas_res = FigureCanvasTkAgg(fig_res, master=demo_res_frame)
            canvas_res.draw()
            canvas_res.get_tk_widget().pack(fill=tk.X)
            
            # Controles para la demostración
            ctrl_res_frame = ttk.Frame(demo_res_frame)
            ctrl_res_frame.pack(pady=5)
            
            # Control de frecuencia
            ttk.Label(ctrl_res_frame, text="Frecuencia (ω/ω₀):").grid(row=0, column=0, padx=5, pady=5)
            freq_var = tk.DoubleVar(value=1.0)
            freq_slider = ttk.Scale(ctrl_res_frame, from_=0.1, to=2.0, variable=freq_var, length=200)
            freq_slider.grid(row=0, column=1, padx=5, pady=5)
            
            # Control de amortiguamiento
            ttk.Label(ctrl_res_frame, text="Amortiguamiento (ζ):").grid(row=0, column=2, padx=5, pady=5)
            zeta_res_var = tk.DoubleVar(value=0.1)
            zeta_res_slider = ttk.Scale(ctrl_res_frame, from_=0.01, to=1.0, variable=zeta_res_var, length=200)
            zeta_res_slider.grid(row=0, column=3, padx=5, pady=5)
            
            # Función para actualizar la demostración
            def actualizar_resonancia(event=None):
                # Obtener valores actuales
                freq_rel = freq_var.get()
                zeta = zeta_res_var.get()
                
                # Actualizar línea vertical
                freq_sel_line.set_xdata([freq_rel, freq_rel])
                
                # Calcular respuesta temporal
                omega0 = 1.0  # Frecuencia natural normalizada
                omega = freq_rel * omega0  # Frecuencia de excitación
                
                # Respuesta estable a una excitación armónica
                # Fase
                phi = np.arctan2(2*zeta*freq_rel, 1-freq_rel**2)
                
                # Amplitud
                amplitud = 1.0 / np.sqrt((1 - freq_rel**2)**2 + (2*zeta*freq_rel)**2)
                
                # Respuesta transitoria + estable
                respuesta = (amplitud * np.sin(omega*t_res - phi)) * (1 - np.exp(-zeta*t_res))
                
                # Actualizar línea de respuesta
                resp_line.set_data(t_res, respuesta)
                
                # Ajustar escalas
                ax_res2.relim()
                ax_res2.autoscale_view()
                
                # Redibujar
                canvas_res.draw()
            
            # Conectar controles
            freq_slider.bind("<Motion>", actualizar_resonancia)
            zeta_res_slider.bind("<Motion>", actualizar_resonancia)
            
            # Inicializar con valores predeterminados
            actualizar_resonancia()
            
            # Ejemplos de resonancia
            ejemplos_frame = ttk.LabelFrame(frame, text="Ejemplos de Resonancia en la Vida Real")
            ejemplos_frame.pack(fill=tk.X, pady=10, padx=20)
            
            ejemplos_texto = """
            • Puente de Tacoma Narrows: Colapso en 1940 debido a oscilaciones resonantes inducidas por el viento.
            • Cristales que se rompen al coincidir con su frecuencia natural.
            • Oscilaciones en motores y maquinaria industrial.
            • Circuitos RLC en electrónica.
            • Instrumentos musicales como violines o guitarras.
            """
            
            ttk.Label(ejemplos_frame, text=ejemplos_texto, style="NormalText.TLabel").pack(pady=10, padx=10)
            
        crear_seccion("Resonancia", resonancia_content)
        
        # ---- SECCIÓN 7: APLICACIONES PRÁCTICAS ----
        def aplicaciones_content(frame):
            ttk.Label(frame, text="Los sistemas masa-resorte tienen numerosas aplicaciones en la ingeniería y la vida cotidiana:", 
                    style="NormalText.TLabel", wraplength=1100).pack(pady=5, anchor="w")
            
            # Crear grid para ejemplos
            app_frame = ttk.Frame(frame)
            app_frame.pack(fill=tk.X, pady=10)
            
            # Crear imágenes representativas
            fig_app = Figure(figsize=(10, 8))
            grid_shape = (2, 2)
            axes = fig_app.subplots(*grid_shape)
            
            # Aplicación 1: Suspensión de vehículos
            ax1 = axes[0, 0]
            ax1.set_title("Suspensión de Vehículos")
            ax1.axis('off')
            
            # Dibujar carro con suspensión
            x_car = np.array([0, 1, 1, 2.5, 2.5, 3.5, 3.5, 4, 4, 0])
            y_car = np.array([1, 1, 1.5, 1.5, 1, 1, 1.5, 1.5, 1, 1])
            ax1.plot(x_car, y_car, 'k-', linewidth=2)
            
            # Ruedas
            rueda1 = plt.Circle((1, 0.5), 0.5, fill=False, color='black', linewidth=2)
            rueda2 = plt.Circle((3, 0.5), 0.5, fill=False, color='black', linewidth=2)
            ax1.add_patch(rueda1)
            ax1.add_patch(rueda2)
            
            # Amortiguadores/resortes
            ax1.plot([1, 1], [1, 0.5], 'r-', linewidth=2, zorder=0)
            ax1.plot([3, 3], [1, 0.5], 'r-', linewidth=2, zorder=0)
            
            # Aplicación 2: Sismógrafos
            ax2 = axes[0, 1]
            ax2.set_title("Sismógrafos")
            ax2.axis('off')
            
            # Dibujar sismógrafo simplificado
            ax2.add_patch(plt.Rectangle((0.5, 0.5), 3, 2, facecolor='lightgray', edgecolor='black'))
            ax2.add_patch(plt.Rectangle((1.5, 2.5), 1, 0.5, facecolor='gray', edgecolor='black'))
            
            # Péndulo con masa
            ax2.plot([2, 2], [2.5, 1.5], 'k-', linewidth=2)
            ax2.add_patch(plt.Circle((2, 1.5), 0.3, facecolor='blue', edgecolor='black'))
            
            # Papel de registro
            ax2.plot([0, 4], [1, 1], 'k-', linewidth=1)
            ax2.plot(np.linspace(0, 4, 100), 0.1*np.sin(10*np.linspace(0, 4, 100))+1, 'r-', linewidth=1)
            
            # Aplicación 3: Edificios antisísmicos
            ax3 = axes[1, 0]
            ax3.set_title("Amortiguadores Sísmicos en Edificios")
            ax3.axis('off')
            
            # Dibujar edificio
            ax3.add_patch(plt.Rectangle((0.5, 0), 3, 3, facecolor='lightgray', edgecolor='black'))
            
            # Pisos
            for i in range(1, 3):
                ax3.plot([0.5, 3.5], [i, i], 'k-', linewidth=1)
            
            # Amortiguadores
            ax3.add_patch(plt.Rectangle((1.5, 0.3), 1, 0.4, facecolor='red', edgecolor='black'))
            ax3.add_patch(plt.Circle((2, 1.5), 0.3, facecolor='red', edgecolor='black'))
            ax3.plot([1.8, 2.2], [1.5, 1.5], 'k-', linewidth=2)
            ax3.plot([2, 2], [1.3, 1.7], 'k-', linewidth=2)
            
            # Aplicación 4: Instrumentos musicales
            ax4 = axes[1, 1]
            ax4.set_title("Instrumentos Musicales")
            ax4.axis('off')
            
            # Dibujar guitarra simplificada
            body_x = np.array([1, 3, 3.5, 3, 1, 0.5, 1])
            body_y = np.array([0.5, 0.5, 1.5, 2.5, 2.5, 1.5, 0.5])
            ax4.plot(body_x, body_y, 'brown', linestyle='-', linewidth=2)
            ax4.add_patch(plt.Circle((2, 1.5), 0.5, facecolor='none', edgecolor='black'))
            
            # Mástil
            ax4.add_patch(plt.Rectangle((3, 1.2), 2, 0.6, facecolor='brown', edgecolor='black'))
            
            # Cuerdas
            for i in range(6):
                y = 1.3 + i*0.08
                ax4.plot([1, 5], [y, y], 'k-', linewidth=1)
            
            fig_app.tight_layout()
            
            # Canvas para la figura
            canvas_app = FigureCanvasTkAgg(fig_app, master=app_frame)
            canvas_app.draw()
            canvas_app.get_tk_widget().pack(fill=tk.X)
            
            # Descripción de aplicaciones
            desc_frame = ttk.Frame(frame)
            desc_frame.pack(fill=tk.X, pady=10)
            
            # Estilo para descripciones
            estilo.configure("App.TLabel", font=("Arial", 10), wraplength=540)
            
            # Columna izquierda
            col_izq = ttk.Frame(desc_frame)
            col_izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            ttk.Label(col_izq, text="Suspensión de Vehículos:", font=("Arial", 11, "bold")).pack(anchor="w", pady=(5,0))
            ttk.Label(col_izq, text="Los sistemas de suspensión automotriz utilizan amortiguadores y resortes para absorber impactos, "
                    "mantener las ruedas en contacto con el suelo y proporcionar una conducción suave. El diseño implica "
                    "encontrar un equilibrio entre confort y manejo.", style="App.TLabel").pack(anchor="w", pady=(0,10))
            
            ttk.Label(col_izq, text="Sismógrafos:", font=("Arial", 11, "bold")).pack(anchor="w", pady=(5,0))
            ttk.Label(col_izq, text="Estos instrumentos utilizan un sistema masa-resorte para detectar y registrar terremotos. "
                    "El principio consiste en una masa suspendida que permanece relativamente estacionaria durante "
                    "un evento sísmico mientras que el marco del instrumento se mueve con el suelo.", 
                    style="App.TLabel").pack(anchor="w", pady=(0,10))
            
            # Columna derecha
            col_der = ttk.Frame(desc_frame)
            col_der.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            ttk.Label(col_der, text="Edificios Antisísmicos:", font=("Arial", 11, "bold")).pack(anchor="w", pady=(5,0))
            ttk.Label(col_der, text="Los amortiguadores de masa sintonizados (TMD) se utilizan en rascacielos para contrarrestar "
                    "vibraciones causadas por viento o terremotos. Consisten en una gran masa conectada al edificio "
                    "mediante resortes y amortiguadores, sintonizados para oscilar en contrafase a la estructura principal.", 
                    style="App.TLabel").pack(anchor="w", pady=(0,10))
            
            ttk.Label(col_der, text="Instrumentos Musicales:", font=("Arial", 11, "bold")).pack(anchor="w", pady=(5,0))
            ttk.Label(col_der, text="Las cuerdas de guitarra, piano y otros instrumentos funcionan como sistemas masa-resorte. "
                    "La tensión de la cuerda actúa como la fuerza restauradora, y la frecuencia de vibración determina "
                    "el tono musical producido.", style="App.TLabel").pack(anchor="w", pady=(0,10))
            
        crear_seccion("Aplicaciones Prácticas", aplicaciones_content)
        
        # ---- SECCIÓN 8: CONCLUSIONES ----
        def conclusiones_content(frame):
            ttk.Label(frame, text="Los sistemas masa-resorte son fundamentales en el estudio de la física y la ingeniería:", 
                    style="NormalText.TLabel", wraplength=1100).pack(pady=5, anchor="w")
            
            # Lista de puntos clave
            puntos_frame = ttk.Frame(frame)
            puntos_frame.pack(fill=tk.X, pady=10, padx=20)
            
            puntos = [
                "• Representan un modelo matemático simple pero poderoso para comprender fenómenos oscilatorios.",
                "• La ecuación diferencial del movimiento permite predecir el comportamiento bajo diferentes condiciones.",
                "• El tipo de amortiguamiento determina cómo el sistema regresa al equilibrio: subamortiguado (oscilante), "
                "críticamente amortiguado (retorno más rápido sin oscilación) o sobreamortiguado (lento sin oscilación).",
                "• La resonancia ocurre cuando una fuerza externa oscila a una frecuencia cercana a la natural del sistema, "
                "pudiendo generar amplitudes peligrosamente grandes.",
                "• Las aplicaciones abarcan desde la ingeniería mecánica y civil hasta la acústica, instrumentos musicales "
                "y dispositivos de precisión.",
                "• El estudio de estos sistemas proporciona las bases conceptuales para comprender sistemas oscilatorios "
                "más complejos."
            ]
            
            for punto in puntos:
                ttk.Label(puntos_frame, text=punto, style="NormalText.TLabel", 
                        wraplength=1060).pack(anchor="w", pady=5)
            
            # Referencias adicionales
            ttk.Label(frame, text="Referencias Adicionales:", 
                    style="SubtitleLabel.TLabel").pack(pady=10, anchor="w", padx=20)
            
            ref_text = """
            • French, A. P. (1971). Vibrations and Waves. W. W. Norton & Company.
            • Marion, J. B., & Thornton, S. T. (1995). Classical Dynamics of Particles and Systems. Saunders College Publishing.
            • Tipler, P. A., & Mosca, G. (2007). Physics for Scientists and Engineers. W. H. Freeman.
            • Serway, R. A., & Jewett, J. W. (2018). Physics for Scientists and Engineers. Cengage Learning.
            """
            
            ttk.Label(frame, text=ref_text, style="NormalText.TLabel").pack(pady=5, anchor="w", padx=20)
            
        crear_seccion("Conclusiones", conclusiones_content)
        
        # Botón para cerrar la ventana de teoría
        ttk.Button(scrollable_frame, text="Cerrar Ventana", 
                command=teoria_ventana.destroy, 
                style="danger.TButton").pack(pady=20)
        
        # Configurar estilo para el botón de cerrar
        estilo.configure("danger.TButton", foreground="white", background="#E74C3C", font=("Arial", 12, "bold"))
        estilo.map("danger.TButton", 
                foreground=[('pressed', 'white'), ('active', 'white')],
                background=[('pressed', '#C0392B'), ('active', '#E74C3C')])