# ProyectoFinal

## Descripción
Proyecto desarrollado en Python que implementa un modelo entrenado. El repositorio contiene el código fuente, el modelo entrenado y los archivos necesarios para ejecutar la aplicación.

## Estructura del repositorio

ProyectoFinal/
│
├── main.py # Script principal para ejecutar el proyecto
├── requirements.txt # Dependencias necesarias para el proyecto
├── src/ # Código fuente del proyecto
├── Trained_model/ # Modelo ya entrenado listo para usar
└── .gitignore # Archivos y carpetas ignorados por Git



## Requisitos

- Python 3.x
- Paquetes listados en `requirements.txt`

## Instalación

1. Clonar el repositorio

git clone https://github.com/angelaperez-21/ProyectoFinal.git
cd ProyectoFinal


2. Crear y activar un entorno virtual (opcional pero recomendado)

python -m venv env
source env/bin/activate # Linux/macOS
env\Scripts\activate # Windows

3. Instalar las dependencias

pip install -r requirements.txt


## Uso

Ejecutar el script principal:

python main.py


Esto iniciará la aplicación o el modelo según la lógica implementada en `main.py`.

## Personalización

- Si deseas entrenar tu propio modelo, revisa el código dentro de la carpeta `src/`.
- Asegúrate de modificar las rutas a los modelos si cambias la estructura de carpetas.
