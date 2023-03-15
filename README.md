# Smoky 2.0 - Detector de Humo de Incendios Forestales

Smoky es un proyecto de detección de humo de incendios forestales mediante el uso de inteligencia artificial. El objetivo principal de este proyecto es ayudar a prevenir incendios forestales mediante la deteccióntemprana del humo. Smoky utiliza el algoritmo YOLOv7 mediante una plataforma web desarrollada en Flask para detectar y clasificar el humo en las imágenes.

## Características

- Detección precisa de humo de incendios forestales mediante inteligencia artificial.
- Interfaz de usuario fácil de usar para cargar y procesar imágenes.
- Alto rendimiento gracias al uso del algoritmo YOLOv7.

## Instalación

Para ejecutar Smoky 2.0, se recomienda utilizar una GPU compatible para un mejor rendimiento.

Para instalar Smoky 2.0, siga los siguientes pasos:

1. Clone el repositorio en su máquina local.
2. Instale los requisitos utilizando el archivo `requirements.txt`.

```
pip install -r requirements.txt
```

3. Ejecute el archivo `app.py`.

```
python 'app.py'.
```

## Uso

Smoky 2.0 cuenta con una interfaz de usuario fácil de usar para cargar y procesar imágenes. Una vez ejecutado, se montará un servidor web al cual podrá acceder en https://localhost:5050/

Para cargar una imagen, simplemente haga clic en el botón "Cargar Imagen" y seleccione la imagen deseada. Smoky 2.0 procesará la imagen y mostrará el resultado en la pantalla.

Asimismo, en la pestaña "cámaras", el software extrae de forma automática imágenes desde cámaras web de acceso público, particularmente de aeródromos en Chile, para detectar columnas de humo de incendios forestales en etapa inicial.

## Contribuir

Si desea contribuir al proyecto, ¡estamos abiertos a sus sugerencias y aportes! Siéntase libre de enviar solicitudes de extracción o informar problemas.

## Créditos

- Desarrollador principal: [Daniel Eldan R.]
- Algoritmo utilizado: YOLOv7
- Aiformankind (August 28, 2020). Wildfire smoke detection research. 
https://github.com/aiformankind/wildfire-smoke-detection-research.
- Wang, Chien-Yao, Bochkovskiy, Alexey and Liao, Hong-Yuan 
Mark (2022). "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. " ("YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real ...") arXiv preprint 
arXiv:2207.02696. https://arxiv.org/abs/2207.02696
- Dwyer, B., Nelson, J. (2022). Roboflow (Version 1.0) [Software]. 
Available from https://roboflow.com. computer vision.

## Paper

- Eldan R. Daniel. (2023). Wildfire Smoke Detection with Computer Vision. arXiv. doi:10.48550/ARXIV.2301.05070
  
  [[Link](https://arxiv.org/abs/2301.05070)]

  **Keywords:** Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences. 
  
  **Copyright:** Creative Commons Attribution Non Commercial Share Alike 4.0 International.