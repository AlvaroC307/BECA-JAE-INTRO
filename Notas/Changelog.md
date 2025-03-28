# Versiones Finales 1.0

## Version 1.0  
    Escritura del programa en python y no en distintos Jupyter. Actualmente funciona mediante un modo jerarquico básico que calcula la chirp mass y el ratio entre las masas Q=m_1/m_2, una vez optimizados estos parametros se calculan el parámetro efectivo de spin y el parámetro de Kerr del agujero negro 1. De momento no está implementada la optimiación a sistemas con precesión. 
#### Version 1.01 
    Cambiado los Boundaries del ratio de masas Q. Ahora se fuerza que siempre m_1>m_2. Además se guardan todos los puntos donde se ha hecho una optimizacion para poder observar como se ha desplazado en la computación.
#### Version 1.02
    Cambiada la jerarquía para que en la primera optimización se calculen Q, M_chirp y Eff_Spin. En la segunda jerarquía se cálcula actualmente solo a_2. Ahora el cuarto parámetro libre es el parámetro de Kerr del agujero negro 2, ya que tiene menos masa y por tanto se puede dejar para la segunda optimización y no la primera con Eff_Spin. Reescrito las funciones de optimización para que sean más simples de leer haciendo que una sean funciones de la total. Creado un archivo representation.py para ver el camino que toman las optimizaciones.

# Version 1.1
    Añadido multiprocessing para comenzar con distintos puntos iniciales y encontrar mejor el máximo global. Añadido el resto de modos no dominantes a la computacion. Añadido el parametro inclinacion. Sustituido la optimizacion en 2 pasos por una en 3 pasos. FALTA POR AÑADIR: parametro ideal para la precesion y elegir bien los puntos iniciales de la misma. 