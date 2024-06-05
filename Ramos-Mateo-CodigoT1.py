import pulp

# Función para leer los datos del archivo
def LecturaDatosInventario():
    with open("datos-inventario.txt", "r") as file:
        lines = file.readlines()
        ct = list(map(int, lines[0].strip().split()))
        Ft = list(map(int, lines[1].strip().split()))
        ht = list(map(int, lines[2].strip().split()))
        pt = list(map(int, lines[3].strip().split()))
        dt = list(map(int, lines[4].strip().split()))
    return ct, Ft, ht, pt, dt

# Función para definir e implementar el modelo de inventario
def ModeloInventario():
    # Leer los datos
    ct, Ft, ht, pt, dt = LecturaDatosInventario()
    T = len(ct)
    
    # Crear el problema de optimización
    prob = pulp.LpProblem("ModeloInventario", pulp.LpMinimize)
    
    # Definir variables de decisión
    xt = pulp.LpVariable.dicts("xt", range(T), lowBound=0, cat='Continuous')
    It = pulp.LpVariable.dicts("It", range(T), lowBound=0, cat='Continuous')
    Bt = pulp.LpVariable.dicts("Bt", range(T), lowBound=0, cat='Continuous')
    yt = pulp.LpVariable.dicts("yt", range(T), cat='Binary')
    
    # Función objetivo
    prob += pulp.lpSum([ht[t] * It[t] + pt[t] * Bt[t] + Ft[t] * yt[t] + ct[t] * xt[t] for t in range(T)])
    
    # Restricciones
    prob += It[0] == 0, "InitialInventory"
    prob += Bt[0] == 0, "InitialBacklog"
    for t in range(1, T):
        prob += It[t] - Bt[t] == It[t-1] - Bt[t-1] + xt[t] - dt[t], f"InventoryBalance_{t}"
        prob += xt[t] <= 10000 * yt[t], f"BigMConstraint_{t}"  # 10000 is a large number
    
    # Resolver el problema
    prob.solve()
    
    # Imprimir resultados
    print("Status:", pulp.LpStatus[prob.status])
    print("Costo total:", pulp.value(prob.objective))
    for t in range(T):
        print(f"Periodo {t}: xt={pulp.value(xt[t])}, It={pulp.value(It[t])}, Bt={pulp.value(Bt[t])}, yt={pulp.value(yt[t])}")

# Ejecutar el modelo
ModeloInventario()
